import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

# print(mnist.data[0])
# print(mnist.target[0])

mnist.target = mnist.target.astype(np.int8)
X = mnist.data / 255  # 0-255값을 [0,1] 구간으로 정규화
Y = mnist.target
# print(X[0])
# print(Y[0])

# plt.imshow(X[0].reshape(28, 28), cmap='gray')
# plt.show()
# print("이 이미지 데이터의 레이블은 {:.0f}이다".format(Y[0]))

# test_size: 테스트 데이터의 비율을 나타내는 매개변수로, 전체 데이터 중 테스트 데이터로 사용할 비율을 지정 / random_state: 데이터를 무작위로 섞는 데 사용되는 시드(seed) 값으로, 동일한 시드를 사용하면 항상 동일한 데이터가 생성
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/7, random_state=0)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

ds_train = TensorDataset(X_train, Y_train)
ds_test = TensorDataset(X_test, Y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28*1, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))

# print(model)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()  # 신경망을 학습 모드로 전환
    # Dropout 및 Batch Normalization 활성화: 드롭아웃은 학습 중에 일부 뉴런을 무작위로 비활성화하여 과적합을 줄임, 배치 정규화는 학습 중에 각 레이어의 입력을 정규화하여 학습을 안정화시킴
    # autograd 활성화
    # 모델의 파라미터(가중치 및 편향)가 업데이트 가능한 상태가 됨
    # model.train()을 호출하지 않으면 모델은 기본적으로 평가(inference) 모드로 간주 됨. 이 모드에서는 드롭아웃이 비활성화되고 그래디언트 계산이 비활성화되므로 모델의 가중치가 업데이트되지 않음

    for data, targets in loader_train:

        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    print("epoch{}：완료\n".format(epoch))


def test():
    model.eval()  # 신경망을 추론 모드로 전환
    correct = 0

    with torch.no_grad():
        for data, targets in loader_test:

            outputs = model(data)

            # 첫 번째 반환 값은 최댓값을 포함하는 새로운 텐서, 두 번째 반환 값은 최댓값의 인덱스를 포함하는 새로운 텐서
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data.view_as(predicted)).sum()

    data_num = len(loader_test.dataset)
    print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct,
                                                         data_num, 100. * correct / data_num))


test()

for epoch in range(3):
    train(epoch)

test()

index = 2018

model.eval()
data = X_test[index]
output = model(data)
_, predicted = torch.max(output.data, 0)

print("예측 결과 : {}".format(predicted))

X_test_show = (X_test[index]).numpy()
plt.imshow(X_test_show.reshape(28, 28), cmap='gray')
plt.show()
print("이 이미지 데이터의 정답 레이블은 {:.0f}입니다".format(Y_test[index]))
