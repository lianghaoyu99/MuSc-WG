import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyMapOptimizer(nn.Module):
    """
    Morphological Optimization Module for Anomaly Maps.
    Includes:
    1. Morphological Opening (Erosion -> Dilation): Removes small noise (e.g., dots, hair).
    2. Morphological Closing (Dilation -> Erosion): Fills gaps/holes (e.g., LED chips).
    3. Gaussian Smoothing: Removes blocky artifacts caused by MaxPool operations.
    """
    def __init__(self, kernel_size_open=3, kernel_size_close=5, sigma=1.0, smooth_kernel_size=5):
        super().__init__()
        self.kernel_open = kernel_size_open
        self.kernel_close = kernel_size_close
        
        # Gaussian smoothing kernel setup
        self.smooth_kernel_size = smooth_kernel_size
        if smooth_kernel_size > 0:
            # Create a 2D Gaussian kernel
            coords = torch.arange(smooth_kernel_size, dtype=torch.float32)
            coords -= (smooth_kernel_size - 1) / 2.0
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g /= g.sum()
            
            # Outer product to get 2D kernel: (K, K)
            g2d = g.view(-1, 1) * g.view(1, -1)
            # Shape it for depthwise convolution: (OutC, InC/Groups, kH, kW) -> (1, 1, K, K)
            self.register_buffer('gaussian_kernel', g2d.view(1, 1, smooth_kernel_size, smooth_kernel_size))

    def forward(self, x):
        # x: (B, 1, H, W) or (B, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # 1. Morphological Opening to remove small noise (Capsule dots/hair)
        if self.kernel_open > 1:
            x = self.opening(x, self.kernel_open)
            
        # 2. Morphological Closing to fill gaps (LED regions)
        if self.kernel_close > 1:
            x = self.closing(x, self.kernel_close)
            
        # 3. Smoothing to eliminate blocky artifacts from max-pooling
        if self.smooth_kernel_size > 0:
            pad = self.smooth_kernel_size // 2
            # Replicate padding to avoid border edge artifacts
            x = F.pad(x, (pad, pad, pad, pad), mode='replicate')
            # Ensure kernel matches input dtype (e.g. Double)
            kernel = self.gaussian_kernel.to(x.dtype)
            x = F.conv2d(x, kernel, padding=0)
            
        return x

    def opening(self, x, k):
        pad = k // 2
        # Erosion: -Max(-x)
        x = -F.max_pool2d(-x, k, stride=1, padding=pad)
        # Dilation: Max(x)
        x = F.max_pool2d(x, k, stride=1, padding=pad)
        return x

    def closing(self, x, k):
        pad = k // 2
        # Dilation: Max(x)
        x = F.max_pool2d(x, k, stride=1, padding=pad)
        # Erosion: -Max(-x)
        x = -F.max_pool2d(-x, k, stride=1, padding=pad)
        return x
