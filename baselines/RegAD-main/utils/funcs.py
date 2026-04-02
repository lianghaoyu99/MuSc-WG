import numpy as np
import torch
from utils.utils import print_log
import torch.nn.functional as F
import kornia as K


def embedding_concat(x, y, use_cuda):
    device = torch.device('cuda' if use_cuda else 'cpu')
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def mahalanobis_torch(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x


def rot90_img(x,k):
    # k is 0,1,2,3
    degreesarr = [0., 90., 180., 270., 360]
    degrees = torch.tensor(degreesarr[k])
    x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
    return x

def grey_img(x):
    x = K.color.rgb_to_grayscale(x)
    x = x.repeat(1, 3, 1,1)
    return x


def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class EarlyStop():
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=True, delta=0, save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, log):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print_log((f'EarlyStopping counter: {self.counter} out of {self.patience}'), log)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer, log):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print_log((f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'),
                      log)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss

import os
import cv2

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    else:
        return (pred - min_value) / (max_value - min_value + 1e-8)

def visualizer(img_path, gt_mask, anomaly_map, save_dir, img_size=224, data_dir=None):
    os.makedirs(save_dir, exist_ok=True)
    
    # Use os.path.normpath and string replacement to handle different OS path separators
    norm_img_path = os.path.normpath(img_path)
    if data_dir is not None:
        norm_data_dir = os.path.normpath(data_dir)
        # Extract relative path robustly
        if norm_data_dir in norm_img_path:
            rel_path = norm_img_path.replace(norm_data_dir, "").lstrip(os.sep)
        else:
            rel_path = os.path.basename(img_path)
    else:
        rel_path = os.path.basename(img_path)
        
    rel_path = rel_path.replace(os.sep, "-").replace("/", "-")     
    base = rel_path.replace(".png", "").replace(".jpg", "")
    
    ori = cv2.imread(img_path)
    if ori is None:
        return
    ori = cv2.cvtColor(cv2.resize(ori, (img_size, img_size)), cv2.COLOR_BGR2RGB)
    
    # GT
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.squeeze().cpu().numpy()
        
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]
        
    gt_mask = (gt_mask * 255).astype(np.uint8) if gt_mask.max() <= 1.0 else gt_mask.astype(np.uint8)
    gt_mask = np.ascontiguousarray(gt_mask)
    gt_mask = cv2.resize(gt_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ori_gt = ori.copy()
    cv2.drawContours(ori_gt, contours, -1, (0, 255, 0), 1)
    
    # Anomaly map
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
    if anomaly_map.ndim == 3 and anomaly_map.shape[0] == 1:
        anomaly_map = anomaly_map[0]
    anomaly_map = cv2.resize(anomaly_map, (img_size, img_size))
    anomaly_map = normalize(anomaly_map)
    vis_img = apply_ad_scoremap(ori, anomaly_map)

    save_vis = os.path.join(save_dir, f"{base}_RegAD.png")
    cv2.imwrite(save_vis, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    save_gt = os.path.join(save_dir, f"{base}_gt.png")
    cv2.imwrite(save_gt, cv2.cvtColor(ori_gt, cv2.COLOR_RGB2BGR))

