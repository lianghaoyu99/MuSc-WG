import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# 尝试导入 WTConv-main 中的 wavelet 工具
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
wtconv_path = os.path.join(project_root, 'WTConv-main')
if wtconv_path not in sys.path:
    sys.path.append(wtconv_path)

try:
    from wtconv.util import wavelet
except ImportError:
    wavelet = None

class WTConv2dStatic(nn.Module):
    """
    高分辨率静态小波聚合模块 (Optimized for Ablation Study).
    
    Args:
        in_channels (int): Input channel dimension.
        wt_levels (int): Number of wavelet decomposition levels.
        wt_type (str): Wavelet type ('db1', 'db2', etc.).
        padding_mode (str): Padding mode for convolution ('reflect', 'zeros', 'replicate', etc.).
        include_level0 (bool): Whether to include the original features (Level 0) in the aggregation.
    """
    def __init__(self, in_channels, wt_levels=1, wt_type='db2', padding_mode='reflect', 
                 include_level0=False, use_details=False, detail_start_level=1, keep_ll=True):
        super(WTConv2dStatic, self).__init__()
        if wavelet is None:
            raise ImportError("wtconv.util.wavelet not found. Please install PyWavelets (pip install PyWavelets).")

        self.wt_levels = wt_levels
        self.padding_mode = padding_mode
        self.include_level0 = include_level0
        self.use_details = use_details
        self.detail_start_level = detail_start_level
        self.keep_ll = keep_ll
        
        # 获取基础小波滤波器
        wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        
        self.register_buffer('wt_filter', wt_filter)
        self.register_buffer('iwt_filter', iwt_filter)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        aggregated_components = []
        if self.include_level0 and self.keep_ll:
            aggregated_components.append(x)
            
        curr_x = x
        
        # 提取滤波器 (LL, LH, HL, HH)
        ll_filter = self.wt_filter[0::4]
        lh_filter = self.wt_filter[1::4]
        hl_filter = self.wt_filter[2::4]
        hh_filter = self.wt_filter[3::4]
        
        k_h, k_w = ll_filter.shape[2], ll_filter.shape[3]
        
        for i in range(self.wt_levels):
            # 随层数增加空洞率 (Dilation)
            dilation = 2**i
            
            # 计算 Padding
            eff_k_h = (k_h - 1) * dilation + 1
            eff_k_w = (k_w - 1) * dilation + 1
            
            pad_total_h = eff_k_h - 1
            pad_total_w = eff_k_w - 1
            
            pad_left = pad_total_w // 2
            pad_right = pad_total_w - pad_left
            pad_top = pad_total_h // 2
            pad_bottom = pad_total_h - pad_top
            
            # 1. Padding
            if self.padding_mode == 'zeros':
                x_padded = F.pad(curr_x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            else:
                x_padded = F.pad(curr_x, (pad_left, pad_right, pad_top, pad_bottom), mode=self.padding_mode)
            
            # 2. Convolution (Approximation / Low-Pass)
            next_ll = F.conv2d(x_padded, ll_filter, groups=C, padding=0, dilation=dilation)
            
            if self.keep_ll:
                aggregated_components.append(next_ll)
            
            # 3. Convolution (Details / High-Pass)
            if self.use_details and i >= self.detail_start_level:
                # LH (Horizontal Detail)
                curr_lh = F.conv2d(x_padded, lh_filter, groups=C, padding=0, dilation=dilation)
                # HL (Vertical Detail)
                curr_hl = F.conv2d(x_padded, hl_filter, groups=C, padding=0, dilation=dilation)
                # HH (Diagonal Detail)
                curr_hh = F.conv2d(x_padded, hh_filter, groups=C, padding=0, dilation=dilation)
                
                # Add details to aggregation
                aggregated_components.append(curr_lh)
                aggregated_components.append(curr_hl)
                aggregated_components.append(curr_hh)
            
            # Update current approximation for next level
            curr_x = next_ll
            
        # 聚合策略
        if len(aggregated_components) > 0:
            out = torch.mean(torch.stack(aggregated_components, dim=0), dim=0)
        else:
            out = x # Fallback
            
        return out

class WTConvLNAMDStatic(nn.Module):
    """
    Parameter-free LNAMD replacement using High-Resolution Static Wavelet Convolution.
    Supports ablation study configuration including Band-Pass filtering.
    """
    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4], r=3, wt_levels=None,
                 wt_type='db2', padding_mode='reflect', include_level0=False,
                 use_details=False, detail_start_level=1, keep_ll=True):
        super(WTConvLNAMDStatic, self).__init__()
        self.device = device
        self.feature_layer = feature_layer
        
        if wt_levels is None:
            # 如果不包含 Level 0，我们需要确保至少有一层分解
            min_levels = 1 if not include_level0 else 0
            # 简单映射逻辑
            wt_levels = max(min_levels, r // 2 + 1)
            
        self.wt_convs = nn.ModuleList([
            WTConv2dStatic(feature_dim, wt_levels=wt_levels, wt_type=wt_type, 
                           padding_mode=padding_mode, include_level0=include_level0,
                           use_details=use_details, detail_start_level=detail_start_level, keep_ll=keep_ll) 
            for _ in feature_layer
        ])
        self.to(device)

    def _embed(self, features):
        """
        Args:
            features: List of tensors (B, L_tokens, C)
        Returns:
            Aggregated features: (B, L_spatial, num_layers, C)
        """
        processed_features = []
        
        for i, feature in enumerate(features):
            feature = feature.to(self.device)
            
            if feature.shape[1] > 1:
                feature = feature[:, 1:, :]
            
            B, L, C = feature.shape
            H = int(math.sqrt(L))
            W = H
            
            feature_spatial = feature.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # LayerNorm 
            feature_spatial = F.layer_norm(feature_spatial, [C, H, W])
            
            wt_out = self.wt_convs[i](feature_spatial)
            
            wt_out_flat = wt_out.flatten(2).transpose(1, 2)
            processed_features.append(wt_out_flat)
            
        return torch.stack(processed_features, dim=2).detach()
