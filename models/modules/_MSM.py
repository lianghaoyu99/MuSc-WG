import torch
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


def _resolve_neighbor_count(k, reference_count):
    """Convert an absolute/fractional K to a valid reference-image count."""
    if isinstance(k, float) and k < 1:
        k = int(reference_count * k)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must select at least one neighbor, got {k}.")
    return min(k, reference_count)


def KNN(Z, device, k=10, desc=None):
    """Return MuSc-style nearest reference-image IDs for every patch.

    Each query patch is first matched to the closest patch in every other
    image. The K closest reference images are then retained. The result has
    shape (image_num, patch_num, K).
    """
    image_num, patch_num, channels = Z.shape
    if image_num < 2:
        raise ValueError("KNN requires at least two images.")

    neighbor_count = _resolve_neighbor_count(k, image_num - 1)
    all_image_ids = torch.arange(image_num, device=device)
    neighbor_indices = []
    iterator = range(image_num)
    if desc is not None:
        iterator = tqdm(iterator, desc=desc)

    for i in iterator:
        Z_ref = torch.cat((Z[:i], Z[i + 1:]), dim=0)
        patch2image = torch.cdist(
            Z[i:i + 1], Z_ref.reshape(-1, channels)
        ).reshape(patch_num, image_num - 1, patch_num).amin(dim=-1)
        local_indices = torch.topk(
            patch2image.float(), neighbor_count, largest=False, sorted=True
        ).indices
        reference_ids = torch.cat((all_image_ids[:i], all_image_ids[i + 1:]))
        neighbor_indices.append(reference_ids[local_indices])

    return torch.stack(neighbor_indices, dim=0)


def KNN_cosine_chunked(Z, device, k=10, reference_chunk_size=2, desc=None):
    """Memory-efficient MuSc-style KNN for L2-normalized features.

    Cosine similarity and Euclidean distance produce identical rankings for
    L2-normalized vectors. Reference images are processed in small blocks, so
    the full (patch_num, (image_num - 1) * patch_num) matrix is never created.
    """
    image_num, patch_num, _ = Z.shape
    if image_num < 2:
        raise ValueError("KNN requires at least two images.")
    if reference_chunk_size < 1:
        raise ValueError("reference_chunk_size must be at least 1.")

    neighbor_count = _resolve_neighbor_count(k, image_num - 1)
    all_image_ids = torch.arange(image_num, device=device)
    neighbor_indices = []
    iterator = range(image_num)
    if desc is not None:
        iterator = tqdm(iterator, desc=desc)

    for i in iterator:
        query = Z[i]
        reference_ids = torch.cat((all_image_ids[:i], all_image_ids[i + 1:]))
        best_scores = None
        best_ids = None

        for id_chunk in reference_ids.split(reference_chunk_size):
            references = Z[id_chunk]
            # (ref_images, ref_patches, query_patches) ->
            # best reference patch per image and query patch.
            similarities = torch.matmul(
                references, query.transpose(0, 1)
            ).amax(dim=1).transpose(0, 1)
            candidate_ids = id_chunk.unsqueeze(0).expand(patch_num, -1)

            if best_scores is not None:
                similarities = torch.cat((best_scores, similarities), dim=1)
                candidate_ids = torch.cat((best_ids, candidate_ids), dim=1)

            keep_count = min(neighbor_count, similarities.shape[1])
            best_scores, positions = torch.topk(
                similarities, keep_count, dim=1, largest=True, sorted=True
            )
            best_ids = torch.gather(candidate_ids, dim=1, index=positions)

        neighbor_indices.append(best_ids)

    return torch.stack(neighbor_indices, dim=0)


def neighbor_disagreement(reference_indices, band_indices):
    """Compute 1 - |intersection| / K for two neighbor-index tensors."""
    if reference_indices.shape != band_indices.shape:
        raise ValueError(
            "Neighbor index tensors must have identical shapes, got "
            f"{reference_indices.shape} and {band_indices.shape}."
        )

    reference_sorted = torch.sort(reference_indices, dim=-1).values.contiguous()
    positions = torch.searchsorted(reference_sorted, band_indices.contiguous())
    valid = positions < reference_sorted.shape[-1]
    safe_positions = positions.clamp_max(reference_sorted.shape[-1] - 1)
    matched = valid & (
        torch.gather(reference_sorted, dim=-1, index=safe_positions) == band_indices
    )
    return 1.0 - matched.float().mean(dim=-1)

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

def MSM(Z, device, topmin_min=0, topmin_max=0.3, gamma=1.0, use_spot_weight=False):
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in tqdm(range(Z.shape[0])):  # 遍历N个样本
    # for i in range(Z.shape[0]):
        anomaly_scores_i = compute_scores_fast(Z, i, device, topmin_min, topmin_max, gamma, use_spot_weight).unsqueeze(0)  # 计算样本i的异常得分（欧氏距离矩阵）
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
    return anomaly_scores_matrix

if __name__ == "__main__":
    device = 'cuda:0'
    import time
    s_time = time.time()
    Z = torch.rand(200, 1369, 1024).to(device)
    MSM(Z, device)
    e_time = time.time()
    print((e_time-s_time)*1000)
