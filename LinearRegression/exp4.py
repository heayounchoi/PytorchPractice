import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

print(x_train.shape)
print(y_train.shape)

# 행렬의 곱셈이 성립되려면 곱셈의 좌측에 있는 행렬의 열의 크기와 우측에 있는 행렬의 행의 크기가 일치해야 함
W = torch.zeros((3, 1), requires_grad=True) # 트레이닝 데이터의 크기가 (5 x 3)이라 벡터의 크기는 (3 x 1)이어야 함 
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    hypothesis = x_train.matmul(W) + b # 편향 b는 브로드 캐스팅되어 각 샘플에 더해짐

    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # detach: 연산 기록으로부터 분리하여 이후 연산들이 추적되는 것을 방지
    # detach를 사용하지 않으면 그래디언트를 계산하지 않는 텐서를 출력하더라도 그래디언트 계산이 활성화되어 있어 출력이 복잡해질 수 있음(출력 단순화)
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))