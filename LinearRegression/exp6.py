import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3,1)
print(list(model.parameters()))

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    prediction = model(x_train) # model(x_train)은 model.forward(x_train)와 동일

    cost = F.mse_loss(prediction, y_train) 

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

new_var =  torch.FloatTensor([[73, 80, 75]]) 
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
print(list(model.parameters()))