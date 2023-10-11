import torch
import torch.nn.functional as F

z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim=0)
print(hypothesis)
print(hypothesis.sum())

z = torch.rand(3, 5, requires_grad=True)
print(z)
# 각 샘플에 대해서 소프트맥스 함수를 적용하여야 하므로 두번째 차원에 대해서 적용한다는 의미로 dim을 1로 지정
hypothesis = F.softmax(z, dim=1)
print(hypothesis)  # 각 행의 원소들의 합이 1임

# 0부터 4까지 (5 미포함) 범위에서 3개의 난수 생성. 왜 3개만 생성하면 중복되는가?
y = torch.randint(5, (5,)).long()[:3]
y = torch.randperm(3)
print(y)
y_one_hot = torch.zeros_like(hypothesis)
# scatter : 텐서의 특정 위치에 값을 설정하는데 사용 (_는 inplace operation)
# 첫번째 인자 '1': 열 방향으로 값을 설정하겠다는 의미
# 두번째 인자 'y.unsqueeze(1)': 설정할 열의 위치를 나타내는 인덱스. y가 1차원 벡터이기 때문에 2차원으로 바꾸기
# 세번째 인자 '1': 설정할 값
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)

print(torch.log(F.softmax(z, dim=1)))  # low level
print(F.log_softmax(z, dim=1))  # high level

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
cost = (y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()
print(cost)
# 원-핫 벡터 필요 없음 / nll: Negative Log Likelihood
cost = F.nll_loss(F.log_softmax(z, dim=1), y)
print(cost)
cost = F.cross_entropy(z, y)  # high level
print(cost)
