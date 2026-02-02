"""
    PatchMaker, Preprocessing and MeanMapper are copied from https://github.com/amazon-science/patchcore-inspection.
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import math

class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize  # 假设r=3
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):  # 输入 features: [4, 1024, 37, 37]
        padding = int((self.patchsize - 1) / 2)   # 计算填充: r=3 → padding = (3-1)/2 = 1
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)  # 执行 Unfold: [4, 1024, 37, 37] → [4, 1024×3×3, 37×37]
        # 计算 patch 网格大小 结果: [37, 37]
        number_of_total_patches = []
        for s in features.shape[-2:]:  # 遍历 37, 37
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )  # 重塑形状: [4, 1024, 3, 3, 1369]
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)  # 改变形状顺序: [4, 1369, 1024, 3, 3]

        if return_spatial_info:
            return unfolded_features, number_of_total_patches  # 展平之后的局部邻域特征，对应的patch形状
        return unfolded_features


class Preprocessing(torch.nn.Module):
    def __init__(self, input_layers, output_dim):
        super(Preprocessing, self).__init__()
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_layer in input_layers:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []   # features 是包含4个张量的列表，每个形状 [5476, 1024, 3, 3]
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))  # 调用 MeanMapper 处理每个特征层
        return torch.stack(_features, dim=1)  # 堆叠所有层的特征，[5476, 4, 1024]


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):  # 输入 features: [5476, 1024, 3, 3]
        features = features.reshape(len(features), 1, -1)  # 展平邻域维度: [5476, 1024, 3, 3] → [5476, 1, 1024×9]
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)  # 自适应平均池化到指定维度并移除维度1: [5476, 1, 1024×9] → [5476, 1024]


class LNAMD(torch.nn.Module):
    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4], r=3, patchstride=1):
        super(LNAMD, self).__init__()
        self.device = device
        self.r = r
        self.patch_maker = PatchMaker(r, stride=patchstride)  # 邻域提取器，将特征图划分为重叠的局部邻域块
        self.LNA = Preprocessing(feature_layer, feature_dim)  # 特征处理器，将不同层的特征统一到相同维度

    def _embed(self, features):  # features(l,b,p,d)[4, (4, 1370, 1024)]
        B = features[0].shape[0]  # batch size = 4

        features_layers = []
        for feature in features:  # 遍历4个特征层
            # reshape and layer normalization
            feature = feature[:, 1:, :] # remove the cls token 移除CLS token: [4, 1370, 1024] → [4, 1369, 1024]
            # 重塑为空间格式: 1369 = 37×37 (518/14=37)
            # [4, 1369, 1024] → [4, 37, 37, 1024]
            feature = feature.reshape(feature.shape[0],
                                      int(math.sqrt(feature.shape[1])),  # 37
                                      int(math.sqrt(feature.shape[1])),  # 37
                                      feature.shape[2])  # 1024
            feature = feature.permute(0, 3, 1, 2)  # 通道在前格式: [4, 1024, 37, 37]
            # 层归一化: 对通道、高度、宽度归一化
            feature = torch.nn.LayerNorm([feature.shape[1], feature.shape[2],
                                          feature.shape[3]]).to(self.device)(feature)
            features_layers.append(feature)  # 得到4个层的特征 [4, (4, 1024, 37, 37)]

        if self.r != 1:
            # divide into patches 使用 PatchMaker 提取每层的局部邻域特征
            features_layers = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features_layers]  # 用Unfold进行局部邻域提取
            patch_shapes = [x[1] for x in features_layers]  # patch形状：4个 [37, 37]
            features_layers = [x[0] for x in features_layers]  # 展平后的局部邻域特征：4个 [4, 1369, 1024, 3, 3]
        else:  # self.r == 1
            patch_shapes = [f.shape[-2:] for f in features_layers]
            features_layers = [f.reshape(f.shape[0], f.shape[1], -1, 1, 1).permute(0, 2, 1, 3, 4) for f in features_layers]

        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features_layers)):  # 对齐不同层的空间分辨率，由于ViT所有层的空间分辨率相同（37×37），对齐步骤被跳过。
            patch_dims = patch_shapes[i]
            if patch_dims[0] == ref_num_patches[0] and patch_dims[1] == ref_num_patches[1]:
                continue
            _features = features_layers[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features_layers[i] = _features
        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]  # 将 [4, 1369, 1024, r, r] 展平为 [4×1369, 1024, r, r]
        
        # aggregation 多层特征聚合
        features_layers = self.LNA(features_layers)  # 用自适应平均池化聚合不同层的特征，输出形状[5476, 4, 1024]
        features_layers = features_layers.reshape(B, -1, *features_layers.shape[-2:])   # (B, L, layer, C) → [4, 1369, 4, 1024]

        return features_layers.detach().cpu()


if __name__ == "__main__":
    import time
    device = 'cuda:0'
    LNAMD_r = LNAMD(device=device, r=3, feature_dim=1024, feature_layer=[1,2,3,4])
    B = 32
    patch_tokens = [torch.rand((B, 1370, 1024)), torch.rand((B, 1370, 1024)), torch.rand((B, 1370, 1024)), torch.rand((B, 1370, 1024))]
    patch_tokens = [f.to('cuda:0') for f in patch_tokens]
    s = time.time()
    features = LNAMD_r._embed(patch_tokens)
    e = time.time()
    print((e-s)*1000/32)