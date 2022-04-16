# %%%
import numpy as np
import matplotlib.pyplot as plt
import torch

#%% 只用numpy实现二分类
data = np.array([[50, 50, 1], [99, 400, 1], [200, 100, 1],
                 [250, 80, -1], [450, 300, -1], [400, 200, -1], [350, 450, -1], [300, 350, -1]])

x = [i[0] for i in data]
y = [i[1] for i in data]

s = 3
# plt.plot(x, y, "ro")

plt.scatter(x[:s], y[:s], 40, 'c')
plt.scatter(x[s:], y[s:], 40, 'm', 'x')

y_final = [i[2] for i in data]
# 权重向量
W = np.zeros(3)

# 学习率
alpha = 0.02


def sign(x):
    return 1 if x > 0 else -1


# 直线函数
f = lambda x: -(x * W[1] + W[0]) / W[2]

num = 0  # 循环次数
i = 0
while True:

    num += 1

    xi = np.array([1, x[i], y[i]])

    # print(f'{i}: {x} . {W} = {np.matmul(W, x)}')

    # 预测值
    y_ = sign(np.matmul(W, xi))

    if y_ != y_final[i]:
        W = W + y_final[i] * xi * alpha
        print(f'W: {W}')
        i = 0

        # plt.plot([0, 10], [f(0), f(10)])

        continue

    i += 1

    if i == len(data):
        break

print(f'W: {W} | num: {num}')

plt.plot([0, 450], [f(0), f(450)])

print(f(0), f(10))
plt.show()

# %%　使用pytorch

print(torch.rand(5, 3))
print(torch.cuda.is_available())

