
import sys
import os
import torch

try:
    from models.modules.WTConvStatic import WTConvLNAMDStatic, WTConv2dStatic
    print("Successfully imported WTConvLNAMDStatic and WTConv2dStatic")
    
    # Check PyWavelets
    import pywt
    print(f"PyWavelets version: {pywt.__version__}")
    
    # Try initialization
    device = torch.device('cpu')
    model = WTConvLNAMDStatic(device=device, feature_dim=64, feature_layer=[1], r=3)
    print("Successfully initialized WTConvLNAMDStatic")
    
    # Try forward pass with dummy data
    dummy_input = [torch.randn(1, 197, 64)] # (B, L, C)
    output = model._embed(dummy_input)
    print(f"Forward pass output shape: {output.shape}")
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
