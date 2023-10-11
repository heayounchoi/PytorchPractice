import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속받는 파이썬 클래스
    def __init__(self): 
        super().__init__() # nn.Module 클래스의 속성들로 초기화
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = LinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

new_var =  torch.FloatTensor([[4.0]]) 
pred_y = model(new_var)
print("훈련 후 입력이 4일 때의 예측값 :", pred_y)
print(list(model.parameters()))