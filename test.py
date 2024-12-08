import torch

# 定义 A 和 B
A = torch.randn(10, 3)       # A 的形状: (batch_size, filter_weights)
B = torch.randn(3, 10, 101)  # B 的形状: (filter_count, channels, length)

# 通过扩展 A 的维度，使其形状与 B 匹配
A_expanded = A.unsqueeze(-1).unsqueeze(-1)  # A 的形状变为 (batch_size, filter_weights, 1, 1)

# 进行广播相乘
result = (A_expanded * B).sum(dim=1)  # 对 filter 权重维度求和，结果形状为 (batch_size, channels, length)

print("结果形状:", result.shape)  # 应输出 (10, 10, 101)
