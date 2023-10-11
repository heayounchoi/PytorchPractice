import torch

m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])  # [3] => [3, 3]
print(m1 + m2)

m1 = torch.FloatTensor([[1, 2]])  # [1, 2] => [[1, 2], [1, 2]]
m2 = torch.FloatTensor([[3], [4]])  # [3], [4] => [[3, 3], [4, 4]]
print(m1 + m2)


m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1.shape)
print(m2.shape)
print(m1.matmul(m2))  # 행렬의 곱셈
print(m1 * m2)  # element-wise 곱셈
print(m1.mul(m2))  # element-wise 곱셈

t = torch.FloatTensor([1, 2])
print(t.mean())
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.mean())
print(t.mean(dim=0))  # 첫번째 차원(행) 제거 / 열의 차원 보존
print(t.mean(dim=1))
print(t.mean(dim=-1))
print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))

print(t.max())
print(t.max(dim=0))  # Returns two valus: max and argmax(최대값을 가진 인덱스)
print(t.max(dim=0)[0])  # max
print(t.max(dim=0)[1])  # argmax
print(t.max(dim=1))
print(t.max(dim=-1))
