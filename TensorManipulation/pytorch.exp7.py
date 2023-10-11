import torch

x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.0))
print(x)
print(x.mul_(2.0))  # 기존 값 덮어쓰기
print(x)
