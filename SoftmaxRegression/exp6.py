import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# GPU: 구글 Colab '런타임 - 런타임 유형 변경 - 하드웨어 가속기 - GPU'
# CPU: '하드웨어 가속기 - None'
USE_CUDA = torch.cuda.is_available()  # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", device)

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)  # 여러 개의 GPU가 있는 경우, 모든 GPU

training_epochs = 5
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',  # 경로
                          train=True,  # True: 훈련 데이터, False: 테스트 데이터
                          transform=transforms.ToTensor(),  # 현재 데이터 파이토치 텐서로 변환
                          download=True)  # 해당 경로에 데이터가 없다면 다운받겠다는 의미

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)  # 마지막 배치 버림 / 다른 미니 배치보다 개수가 적은 마지막 배치를 경사 하강법에 사용하여 마지막 배치가 상대적으로 과대 평가되는 현상을 막아줌

# to(): 연산을 어디서 수행할지를 정함 / 모델의 매개변수를 지정한 장치의 메모리로 보냄
linear = nn.Linear(784, 10, bias=True).to(device)

# torch.nn.functional.cross_entropy()랑 동일
criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    # '%04d' % (epoch + 1) : 문자열 포맷칭을 사용하여 정수를 형식화 / 0 : 출력 문자열에서 남는 공간 0으로 채우기 / 4 : 출력 문자열의 총 길이 4자리로 만들기 / d: 정수 값
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

with torch.no_grad():  # 연산을 추적하지 않도록 함 / 아래에서의 연산은 역전파를 수행하지 않으며, 그래디언트를 계산하지 않음
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)  # 예측 결과를 나타내는 텐서. 각 열은 각 클래스에 해당
    # torch.argmax(input, dim, keepdim=False): 주어진 텐서 'input'에서 최댓값을 가지는 요소의 인덱스를 반환함. dim을 1로 지정하면 각 행마다 최댓값 계산
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r +
                                         1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28,
               28), cmap='Greys', interpolation='nearest')
    plt.show()
