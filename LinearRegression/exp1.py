import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)  # random num 동일한 난수로 고정
# 보편적으로 입력은 x, 출력은 y를 사용하여 표기
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# print(x_train)
# print(x_train.shape)
# print(y_train)
# print(y_train.shape)

# requires_grad = True : 자동 미분 기능 적용
# requires_grad = True가 적용된 텐서에 연산을 하면, 계산 그래프가 생성되며 backward 함수를 호출하면 그래프로부터 자동으로 미분이 계산됨
W = torch.zeros(1, requires_grad=True)  # 값 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시
# print(W)

b = torch.zeros(1, requires_grad=True)
# print(b)

optimizer = optim.SGD([W, b], lr=0.01)  # SGD(경사하강법의 일종) / lr(learning rate)

nb_epochs = 2000  # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W + b
    # print(hypothesis)

    cost = torch.mean((hypothesis - y_train) ** 2)
    # print(cost)

    optimizer.zero_grad()
    # gradient(미분을 통해 얻은 기울기)를 0으로 초기화 /
    # 파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있음
    # 기울기를 초기화해야만 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있음
    cost.backward()  # 비용 함수를 미분하여 gradient 계산
    optimizer.step()  # W와 b를 업데이트

    if epoch % 100 == 0:
        print(
            "Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}".format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            )
        )
