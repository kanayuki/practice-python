import numpy as np
import math
import matplotlib.pyplot as plt


def bda(num, max_iter, dimension, cost_function):
    food_fitness = math.inf
    food_pos = np.zeros((1, dimension))

    enemy_fitness = -math.inf
    enemy_pos = np.zeros((1, dimension))

    # 初始化x 和 delta_x 向量
    x = np.random.randint(0, 2, size=(num, dimension))
    # for xi in np.nditer(x[math.floor(num / 2):], op_flags=["readwrite"]):
    #     xi[...] = 1 if xi == 0 else 0

    x[-math.floor(num / 2):] = 1 - x[-math.floor(num / 2):]
    # x[-math.floor(num / 2):] = 1 - x[:math.floor(num / 2)]

    delta_x = np.random.randint(0, 2, size=(num, dimension))

    # fitness = np.zeros((1, num))

    convergence_curve = []

    for iteration in range(1, max_iter + 1):

        w = 0.9 - iteration * ((0.9 - 0.4) / max_iter)

        my_c = 0.1 - iteration * ((0.1 - 0) / (max_iter / 2))
        if my_c < 0:
            my_c = 0

        s = 2 * np.random.random() * my_c  # 分离权重
        a = 2 * np.random.random() * my_c  # 对齐权重
        c = 2 * np.random.random() * my_c  # 聚集权重
        f = 2 * np.random.random()  # 向食物吸引权重
        e = my_c  # 躲避敌人权重

        if iteration > max_iter * 3 / 4:
            e = 0

        # 计算目标函数
        for i in range(num):
            fitness = cost_function(x[i])
            if fitness < food_fitness:
                food_fitness = fitness
                food_pos = np.copy(x[i])

            if fitness > enemy_fitness:
                enemy_fitness = fitness
                enemy_pos = np.copy(x[i])

        convergence_curve.append(food_fitness)

        # 迭代每一个解
        for i in range(num):
            # neighbours_no = 0
            neighbours_delta_x = []
            neighbours_x = []

            # 查找邻近解
            for j in range(num):
                if i != j:
                    # neighbours_no = neighbours_no + 1
                    neighbours_delta_x.append(delta_x[j])
                    neighbours_x.append(x[j])

            neighbours_x = np.array(neighbours_x)
            neighbours_delta_x = np.array(neighbours_delta_x)

            # 分离
            S = -np.sum(neighbours_x - x[i], axis=0)

            # 对齐
            A = np.mean(neighbours_delta_x, axis=0)

            # 内聚
            C = np.mean(neighbours_x) - x[i]

            # 吸引向食物
            F = food_pos - x[i]

            # 躲避敌人
            E = enemy_pos + x[i]

            delta_x[i] = s * S + a * A + c * C + f * F + e * E + w * delta_x[i]
            delta_x[i] = np.clip(delta_x[i], -6, 6)

            for r in range(delta_x[i].size):
                dx = delta_x[i, r]
                # if np.random.random() < abs(dx) / math.sqrt(1 + dx ** 2):
                if np.random.random() < dx ** 2 / (1 + dx ** 2):
                    x[i, r] = 1 if x[i, r] == 0 else 0
        # print(iteration)

    return food_fitness, food_pos, convergence_curve


def tt():
    score = []
    for i in range(10):
        best_score, _, _ = bda(50, 100, 30, lambda x: np.sum(x))
        score.append(best_score)

    mean = np.mean(score)
    print("结果：", score)
    print('平均值：', mean)


def run():
    best_score, best_pos, convergence_curve = bda(50, 100, 30, lambda x: np.sum(x))
    print(best_pos)
    print(best_score)
    plt.plot(convergence_curve)
    plt.show()


if __name__ == '__main__':
    tt()
    # run()
