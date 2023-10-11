import numpy as np

t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(t)
print("Rank of t: ", t.ndim)  # 차원
print("Shape of t: ", t.shape)  # 크기
print("t[0] t[1] t[-1] = ", t[0], t[1], t[-1])
print(t[2:5], t[4:-1])  # 끝 번호 포함하지 않음
print(t[:2], t[3:])

t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
print(t)
print("Rank of t: ", t.ndim)  # 차원
print("Shape of t: ", t.shape)  # 크기
