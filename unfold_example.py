import torch
import torch.nn as nn

# 1. 创建一个简单的输入张量
# 形状: [Batch=1, Channel=1, Height=4, Width=4]
# 为了演示清晰，我们用 1 到 16 的数字填充
input_tensor = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)

print("=== 输入张量 (4x4) ===")
print(input_tensor)
print(f"Shape: {input_tensor.shape}\n")

# 2. 定义 Unfold 操作
# kernel_size=2: 提取 2x2 的滑窗
# stride=1: 每次移动 1 格
# padding=0: 不填充
unfold = nn.Unfold(kernel_size=2, stride=1, padding=0)

# 3. 执行 Unfold
output = unfold(input_tensor)

print("=== Unfold 参数 ===")
print("Kernel size: 2x2")
print("Stride: 1")
print("\n=== 输出张量 ===")
print(output)
print(f"Output Shape: {output.shape}")

# 4. 解释输出维度
# Output Shape: [Batch, Channel * Kernel_H * Kernel_W, Number_of_Patches]
# [1, 1 * 2 * 2, 9] -> [1, 4, 9]
# 4x4 的图，用 2x2 的核滑动 (stride=1)，横向能滑 3 次，纵向能滑 3 次，共 3*3=9 个块
print("\n=== 解释 ===")
print(f"输出维度 [1, 4, 9] 的含义:")
print("1: Batch size")
print("4: 展平后的块大小 (Channel=1 * 2 * 2 = 4)")
print("9: 提取出的块(Patch)总数量 (3 * 3 = 9)")

print("\n=== 第一个 Patch (对应输入左上角 2x2) ===")
# 输出的第一列对应第一个 patch
first_patch_flattened = output[0, :, 0]
print(f"展平的值: {first_patch_flattened.tolist()}")
print("对应输入矩阵中的:")
print(input_tensor[0, 0, 0:2, 0:2])
