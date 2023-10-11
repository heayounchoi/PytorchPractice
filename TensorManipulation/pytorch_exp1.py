import torch

t = torch.FloatTensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(t)
print(t.dim())  # 차원
print(t.shape)  # 크기
print(t.size())  # 크기
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

t = torch.FloatTensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
)
print(t)
print(t.dim())
print(t.size())
print(t[:, 1])  # 첫번째 차원을 전체 선택한 상황에서 두번째 차원의 첫번째 것만 가져오기
print(t[:, 1].size())
print(t[:, :-1])
