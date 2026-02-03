import timm  # noqa
import torchvision.models as models  # noqa
import models.backbone.vision_transformer as vits
import models.backbone.dino_vision_transformer as dino_vits
import torch


_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch8_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch8_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch8_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch8_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch8_224", pretrained=True)',
    "vit_swin_base_win12": 'timm.create_model("swin_base_patch4_window12_384.ms_in22k", pretrained=True)',
    "vit_swin_base_win7": 'timm.create_model("swin_base_patch4_window7_224.ms_in22k", pretrained=True)',
    "vit_swin_large_win12": 'timm.create_model("swin_large_patch4_window12_384.ms_in22k", pretrained=True)',
    "vit_swin_large_win7": 'timm.create_model("swin_large_patch4_window7_224.ms_in22k", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)'
}


def load(name):
    url = []
    patch_size = 8
    if name == "dino_deitsmall16":
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        patch_size = 16
    elif name == "dino_deitsmall8_300ep":
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
    elif name == "dino_vitbase16":
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        patch_size = 16
    elif name == "dino_vitbase8":
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    elif name=="dinov2_vits14":
        url = "dinov2_vits14/dinov2_vits14_pretrain.pth"
        patch_size = 14
    elif name=="dinov2_vitb14":
        url = "dinov2_vitb14/dinov2_vitb14_pretrain.pth"
        patch_size = 14
    elif name=="dinov2_vitl14":
        url = "dinov2_vitl14/dinov2_vitl14_pretrain.pth"
        patch_size = 14
    elif name=="dinov3_vitl16":
        # Correct URL with forward slashes for DINOv3
        url = "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        patch_size = 16
        
    if 'dinov2' in url or 'dinov3' in url:
        repo = 'facebookresearch/dinov2' if 'dinov2' in url else 'facebookresearch/dinov3'
        
        # Patch for torch._dynamo.config.accumulated_cache_size_limit issue in DINOv3
        patch_applied = False
        original_config = None
        try:
            if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
                if not hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
                    original_config = torch._dynamo.config
                    
                    class ConfigProxy:
                        def __init__(self, config):
                            self.__dict__['_config'] = config
                        def __getattr__(self, name):
                            return getattr(self._config, name)
                        def __setattr__(self, name, value):
                            if name == 'accumulated_cache_size_limit':
                                return
                            setattr(self._config, name, value)
                            
                    torch._dynamo.config = ConfigProxy(original_config)
                    patch_applied = True
        except Exception as e:
            print(f"Warning: Failed to apply dinov3 patch: {e}")

        try:
            if 'dinov3' in name:
                # Handle DINOv3 special loading to avoid Windows path backslash issues and 403 errors
                # We attempt to download manually or use the weights argument
                import os
                try:
                    from torch.hub import get_dir
                    hub_dir = get_dir()
                except:
                    hub_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub')
                
                checkpoints_dir = os.path.join(hub_dir, 'checkpoints')
                os.makedirs(checkpoints_dir, exist_ok=True)
                
                # Extract filename from the correct URL
                filename = os.path.basename(url)
                local_file = os.path.join(checkpoints_dir, filename)
                
                # 1. Try to download if missing
                if not os.path.exists(local_file):
                    print(f"Attempting to download DINOv3 weights from: {url}")
                    try:
                        torch.hub.download_url_to_file(url, local_file)
                    except Exception as e:
                        print(f"\nWarning: Automatic download failed: {e}")
                        print("DINOv3 weights might require a signed URL or manual download.")
                        print(f"Please manually download the weights from the official source or the link above.")
                        print(f"And save it to: {local_file}")
                        print("Continuing... assuming you might have a local file or want to try default loading.\n")

                # 2. Pass explicit weights path/url to override internal default
                if os.path.exists(local_file):
                    print(f"Loading local DINOv3 weights: {local_file}")
                    model = torch.hub.load(repo, name, weights=local_file)
                else:
                    # Try passing the corrected URL directly
                    print(f"Trying to load DINOv3 with URL: {url}")
                    model = torch.hub.load(repo, name, weights=url)
            else:
                model = torch.hub.load(repo, name)
        finally:
            if patch_applied and original_config is not None:
                torch._dynamo.config = original_config
                
        return model

    elif len(url)>0:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # build model
        # vit_tiny, vit_small, vit_base, patch_size=8, 16
        model = vits.__dict__['vit_base'](patch_size=patch_size, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)

        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
        return model
    return eval(_BACKBONES[name])
