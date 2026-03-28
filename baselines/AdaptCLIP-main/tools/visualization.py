import os

import cv2
import numpy as np
import torch

from .utils import normalize


def visualizer(img_path, gt_mask, anomaly_map, save_dir, img_size=518, data_dir=None):
    # 兼容跨平台路径提取
    norm_img_path = os.path.normpath(img_path)
    if data_dir is not None:
        norm_data_dir = os.path.normpath(data_dir)
        if norm_data_dir in norm_img_path:
            rel_path = norm_img_path.replace(norm_data_dir, "").lstrip(os.sep)
        else:
            rel_path = os.path.basename(img_path)
    else:
        rel_path = os.path.basename(img_path)
        
    rel_path = rel_path.replace(os.sep, "-").replace("/", "-")     
    base = rel_path.replace(".png", "")
    
    # AdaptCLIP's img_size can be a single int or tuple. If int, handle it.
    if isinstance(img_size, int):
        resize_dims = (img_size, img_size)
    else:
        resize_dims = (img_size[0], img_size[1])
        
    # Read original image directly from path instead of taking it as a tensor
    ori_img = cv2.imread(img_path)
    if ori_img is None:
        print(f"Warning: Could not read image at {img_path}")
        return
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    vis = cv2.resize(ori_img.copy(), resize_dims)  # RGB
    
    # 异常图
    mask = normalize(anomaly_map)
    vis = apply_ad_scoremap(vis, mask)

    # 可视化 GT
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.squeeze().cpu().numpy()
    gt_mask = (gt_mask * 255).astype(np.uint8) if gt_mask.max() <= 1.0 else gt_mask.astype(np.uint8)
    gt_mask = np.ascontiguousarray(gt_mask)
    gt_mask = cv2.resize(gt_mask, resize_dims, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_vis = os.path.join(save_dir, f"{base}_AdaptCLIP.png")
    cv2.imwrite(save_vis, vis)
    
    # 保存一张纯GT的图做对比
    ori_gt = cv2.resize(ori_img.copy(), resize_dims)
    cv2.drawContours(ori_gt, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(save_dir, f"{base}_gt.png"), cv2.cvtColor(ori_gt, cv2.COLOR_RGB2BGR))


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
