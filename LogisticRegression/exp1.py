import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))  # default: W가 1이고, b가 0인 그래프


x = np.arange(-5.0, 5.0, 0.1)  # -5.0부터 5.0까지 0.1씩 증가하는 리스트 생성
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--')  # W의 값이 0.5일때
plt.plot(x, y2, 'g')  # W의 값이 1일때
plt.plot(x, y3, 'b', linestyle='--')  # W의 값이 2일때
plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()  # 선형 회귀에서 가중치 W는 직선의 기울기를 의미하며, 로지스틱 회귀에서 가중치 W는 그래프의 경사도를 의미

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--')  # x + 0.5
plt.plot(x, y2, 'g')  # x + 1
plt.plot(x, y3, 'b', linestyle='--')  # x + 1.5
plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()  # b의 값에 따라서 그래프가 좌, 우로 이동
