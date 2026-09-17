import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

"""
We provide two implementations of the MSM module.
The above commented out function provides faster speeds, but because more tensors are loaded onto the GPU at once, the memory consumption is higher.
By default, our program uses the following function, which is slower but consumes less GPU memory.
"""

def compute_scores_fast(Z, i, device, topmin_min=0, topmin_max=0.3, gamma=1.0, use_spot_weight=False):
    # speed fast but space large
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    patch2image = torch.tensor([]).to(device)
    Z_ref = torch.cat((Z[:i], Z[i+1:]), dim=0)  # 排除当前样本i，创建参考样本集
    patch2image = torch.cdist(Z[i:i+1], Z_ref.reshape(-1, c)).reshape(patch_num, image_num-1, patch_num)  # 计算欧氏距离矩阵
    patch2image = torch.min(patch2image, -1)[0]  # 计算最小距离
    # interval average 区间平均 取最相似的30%的其他图像的距离
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max

    # Get Top K nearest neighbors
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)

    # Capture nearest neighbor distance for Spot Weighting (Rare Pattern Suppression)
    d_nearest = vals[:, 0]

    # Interval processing: Keep range [k_min, k_max]
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    score = torch.mean(patch2image, dim=1)

    # Spot Weighting: Suppress "Occasional Normal Patterns"
    # If a patch matches VERY well with at least one other image (d_nearest is small),
    # we reduce its score, even if the average distance (score) is high.
    # We use Geometric Mean: sqrt(Interval_Score * Nearest_Score)
    if use_spot_weight:
        score = torch.sqrt(score * d_nearest)

    # Gamma Scaling to suppress secondary anomalies
    if gamma != 1.0:
        # Normalize score to [0, 1] roughly for effective gamma scaling
        # Note: raw scores are distances, so they are not strictly [0,1].
        # However, gamma scaling works best on relative magnitudes.
        # We apply it directly. Larger values will get much larger, smaller values will shrink relatively.
        # To make it safer, we can normalize locally or just apply power.
        # Let's apply power directly as simple suppression.
        score = torch.pow(score, gamma)

    return score


def soft_reference_distribution(
    Z,
    device,
    temperature=0.5,
    reference_chunk_size=2,
):
    """Return a soft preference over every other reference image.

    For each query patch, the closest patch distance is retained separately
    for every reference image, matching the patch-to-image distance used by
    MSM. The resulting distances are standardized across reference images
    before temperature-scaled softmax so that one temperature is meaningful
    across layers, radii and categories.

    Returns:
        Tensor shaped ``(image_num, patch_num, image_num - 1)``. The last
        dimension follows ascending global image ID with the query image
        omitted, and therefore aligns across frequency bands.
    """
    image_num, patch_num, _ = Z.shape
    if image_num < 2:
        raise ValueError("Soft reference distribution requires at least two images.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if reference_chunk_size < 1:
        raise ValueError("reference_chunk_size must be at least 1.")

    all_image_ids = torch.arange(image_num, device=device)
    distributions = []

    for i in range(image_num):
        query = Z[i]
        reference_ids = torch.cat((all_image_ids[:i], all_image_ids[i + 1:]))
        distance_chunks = []

        for id_chunk in reference_ids.split(reference_chunk_size):
            references = Z[id_chunk]
            # Features are L2-normalized by the caller. Maximizing cosine
            # similarity over reference patches is exactly equivalent to
            # minimizing Euclidean distance over those patches.
            best_similarity = torch.matmul(
                references, query.transpose(0, 1)
            ).amax(dim=1).transpose(0, 1).float()
            distances = torch.sqrt(
                (2.0 - 2.0 * best_similarity).clamp_min(0.0)
            )
            distance_chunks.append(distances)

        patch2image = torch.cat(distance_chunks, dim=1)
        # Per-query-patch scale calibration. Softmax is shift-invariant, but
        # centering makes the calibrated distances easier to inspect.
        center = patch2image.mean(dim=-1, keepdim=True)
        scale = patch2image.std(
            dim=-1, unbiased=False, keepdim=True
        ).clamp_min(1e-6)
        calibrated = (patch2image - center) / scale
        distributions.append(
            torch.softmax(-calibrated / float(temperature), dim=-1)
        )

    return torch.stack(distributions, dim=0)


def jensen_shannon_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence normalized to the interval [0, 1]."""
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=-1)
        + (q * (q.log() - midpoint.log())).sum(dim=-1)
    )
    return (divergence / math.log(2.0)).clamp(0.0, 1.0)

def compute_scores_slow(Z, i, device, topmin_min=0, topmin_max=0.3):
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def _resize_patch_scores(scores, target_patch_count):
    """Resize an (N, P) patch score without assuming a fixed backbone grid."""
    if scores.shape[1] == target_patch_count:
        return scores

    source_size = int(math.sqrt(scores.shape[1]))
    target_size = int(math.sqrt(target_patch_count))
    if source_size * source_size != scores.shape[1]:
        raise ValueError(f"Source patch count {scores.shape[1]} is not square.")
    if target_size * target_size != target_patch_count:
        raise ValueError(f"Target patch count {target_patch_count} is not square.")

    return F.interpolate(
        scores.float().reshape(scores.shape[0], 1, source_size, source_size),
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False,
    ).flatten(1)


def _integrated_manifold_score(
    frequency_features,
    device,
    temperature,
    reference_chunk_size,
):
    """Compute LL/detail soft reference-distribution disagreement in MSM.

    ``frequency_features`` contains one layer at one aggregation radius. Each
    tensor is kept on CPU by the caller and transferred band-by-band so the
    four frequency tensors do not have to reside on the GPU simultaneously.
    """
    required_bands = ('ll', 'lh', 'hl', 'hh')
    missing = [name for name in required_bands if name not in frequency_features]
    if missing:
        raise ValueError(
            "Missing frequency features for integrated manifold MSM: "
            + ", ".join(missing)
        )

    reference_distribution = None
    disagreement_sum = None
    expected_shape = None

    for band_name in required_bands:
        band_features = frequency_features[band_name].to(
            device, non_blocking=True
        )
        if expected_shape is None:
            expected_shape = band_features.shape
        elif band_features.shape != expected_shape:
            raise ValueError(
                "All manifold bands must have the same shape, got "
                f"{expected_shape} and {band_features.shape}."
            )

        band_features = F.normalize(band_features, dim=-1, eps=1e-6)
        band_distribution = soft_reference_distribution(
            band_features,
            device,
            temperature=temperature,
            reference_chunk_size=reference_chunk_size,
        )
        del band_features

        if band_name == 'll':
            reference_distribution = band_distribution
            continue

        disagreement = jensen_shannon_divergence(
            reference_distribution, band_distribution
        )
        disagreement_sum = (
            disagreement
            if disagreement_sum is None
            else disagreement_sum + disagreement
        )
        del band_distribution, disagreement

    score = disagreement_sum / 3.0
    del reference_distribution, disagreement_sum
    return score


def MSM(
    Z,
    device,
    topmin_min=0,
    topmin_max=0.3,
    gamma=1.0,
    use_spot_weight=False,
    frequency_features=None,
    manifold_temperature=0.5,
    lambda_m=0.1,
    manifold_normalize=True,
    manifold_ref_chunk_size=2,
    return_manifold_score=False,
):
    """Mutual scoring with optional in-module frequency-manifold regularization.

    By default, frequency features are fused into this layer's MSM result for
    backward compatibility. With ``return_manifold_score=True``, the function
    instead returns ``(base_msm, raw_manifold)`` so the caller can defer
    calibration and fusion until after all MSM layers/radii have been averaged.
    Consequently, ``lambda_m=0`` exactly recovers the original MSM result.
    """
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in tqdm(range(Z.shape[0])):  # 遍历N个样本
    # for i in range(Z.shape[0]):
        anomaly_scores_i = compute_scores_fast(Z, i, device, topmin_min, topmin_max, gamma, use_spot_weight).unsqueeze(0)  # 计算样本i的异常得分（欧氏距离矩阵）
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
    if frequency_features is None or lambda_m == 0:
        if return_manifold_score:
            return anomaly_scores_matrix, None
        return anomaly_scores_matrix

    manifold_score = _integrated_manifold_score(
        frequency_features,
        device=device,
        temperature=manifold_temperature,
        reference_chunk_size=manifold_ref_chunk_size,
    )
    manifold_score = _resize_patch_scores(
        manifold_score, anomaly_scores_matrix.shape[1]
    ).to(anomaly_scores_matrix.dtype)

    if return_manifold_score:
        return anomaly_scores_matrix, manifold_score

    if manifold_normalize:
        base_float = anomaly_scores_matrix.float()
        lower = torch.quantile(base_float, 0.01)
        upper = torch.quantile(base_float, 0.99)
        manifold_scale = (upper - lower).clamp_min(1e-8)
    else:
        manifold_scale = 1.0

    joint_score = (
        anomaly_scores_matrix
        + float(lambda_m) * manifold_scale * manifold_score
    )
    del manifold_score
    return joint_score

if __name__ == "__main__":
    device = 'cuda:0'
    import time
    s_time = time.time()
    Z = torch.rand(200, 1369, 1024).to(device)
    MSM(Z, device)
    e_time = time.time()
    print((e_time-s_time)*1000)
