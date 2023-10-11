import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1,1) # (input_dim(입력의 차원), output_dim(출력의 차원)) / 단순 선형 회귀이므로 1차원으로 설정
print(list(model.parameters())) # 첫번째 값이 W, 두번째 값이 b / 랜덤 초기화 상태

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    hypothesis = model(x_train)

    cost = F.mse_loss(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward() # backward 연산: 비용 함수를 미분하여 기울기를 구하는 것
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

new_var =  torch.FloatTensor([[4.0]]) 
pred_y = model(new_var) # forward 연산: H(x) 식에 입력 x로부터 예측된 y를 얻는 것
print("훈련 후 입력이 4일 때의 예측값 :", pred_y)
print(list(model.parameters()))