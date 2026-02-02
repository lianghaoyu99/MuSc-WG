
import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add WTConv-main to path to import wtconv
# Assuming this file is in models/modules/ and WTConv-main is in the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
wtconv_path = os.path.join(project_root, 'WTConv-main')
if wtconv_path not in sys.path:
    sys.path.append(wtconv_path)

try:
    from wtconv.wtconv2d import WTConv2d
except ImportError:
    print(f"Warning: Could not import WTConv2d from {wtconv_path}. Please check the path.")
    # Fallback or dummy for testing if import fails, but we expect it to work
    WTConv2d = None

class WTConvLNAMD(nn.Module):
    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4], r=3, patchstride=1, wt_levels=2):
        """
        WTConvLNAMD: A replacement for LNAMD using Wavelet Convolution.
        
        Args:
            device: torch device
            feature_dim: input feature dimension (channels)
            feature_layer: list of layers to process
            r: kernel size parameter (used to determine kernel_size for WTConv)
               LNAMD uses r=3 for 3x3 patches. WTConv defaults to 5x5.
               We can set kernel_size = r (if odd) or r*2-1 or just fixed 5.
               Here we will use kernel_size=5 as default for WTConv or max(r, 5).
            wt_levels: levels of wavelet transform
        """
        super(WTConvLNAMD, self).__init__()
        self.device = device
        self.feature_layer = feature_layer
        
        if WTConv2d is None:
            raise ImportError("WTConv2d not found")

        # Create a WTConv2d layer for each feature layer
        # Note: WTConv2d(in, out)
        # We assume we want to preserve dimensions, so in=out=feature_dim
        # We use a ModuleList to store them
        self.wt_convs = nn.ModuleList()
        
        # Use a kernel size related to r, or default to 5 (WTConv standard)
        # LNAMD r=3 implies capturing context of 3x3. WTConv captures larger context.
        kernel_size = 5 
        
        for _ in feature_layer:
            self.wt_convs.append(
                WTConv2d(feature_dim, feature_dim, kernel_size=kernel_size, wt_levels=wt_levels, stride=1)
            )
        
        self.to(device)

    def _embed(self, features):
        """
        Args:
            features: List of tensors, one per layer. 
                      Shape: (B, L_tokens, C) where L_tokens includes CLS token usually.
        Returns:
            Aggregated features: (B, L_spatial, num_layers, C)
        """
        B = features[0].shape[0]
        
        processed_features = []
        
        for i, feature in enumerate(features):
            # Move to device
            feature = feature.to(self.device)
            
            # 1. Remove CLS token (assume index 0) like LNAMD
            # LNAMD: feature = feature[:, 1:, :]
            if feature.shape[1] > 1: # Basic check
                feature = feature[:, 1:, :]
            
            # 2. Reshape to spatial (B, C, H, W)
            # L = H * W. Assume square image.
            L = feature.shape[1]
            H = int(math.sqrt(L))
            W = H
            C = feature.shape[2]
            
            # (B, L, C) -> (B, H, W, C) -> (B, C, H, W)
            feature_spatial = feature.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # 3. Apply LayerNorm (LNAMD does this before processing)
            # LNAMD: LayerNorm([C, H, W])
            # We can use InstanceNorm or LayerNorm. LNAMD uses LayerNorm on (C,H,W) dims?
            # LNAMD code: torch.nn.LayerNorm([feature.shape[1], feature.shape[2], feature.shape[3]])
            # which is LayerNorm over C, H, W. Effectively normalizing the whole sample?
            # Let's replicate LNAMD normalization for consistency
            feature_spatial = F.layer_norm(feature_spatial, [C, H, W])
            
            # 4. Apply WTConv
            # Input: (B, C, H, W) -> Output: (B, C, H, W)
            wt_out = self.wt_convs[i](feature_spatial)
            
            # 5. Reshape back to (B, L, C)
            # (B, C, H, W) -> (B, C, L) -> (B, L, C)
            wt_out_flat = wt_out.flatten(2).transpose(1, 2)
            
            processed_features.append(wt_out_flat)

        # Stack layers: (B, L, num_layers, C)
        # LNAMD output: features_layers = features_layers.reshape(B, -1, *features_layers.shape[-2:])
        # LNAMD stacks in Preprocessing -> stack dim 1 -> (B*L, num_layers, C) ?
        # Wait, LNAMD Preprocessing returns stack dim 1.
        # features_layers in LNAMD line 130: reshape(B, -1, *features_layers.shape[-2:])
        # If Preprocessing returns (N, num_layers, C), then reshape -> (B, L, num_layers, C)
        
        output = torch.stack(processed_features, dim=2) # (B, L, num_layers, C)
        
        return output.detach().cpu()
