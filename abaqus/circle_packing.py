import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def circle_packing_unconstraint(R, r, num_circles, initial_positions):
    # 定义目标函数
    def objective(positions):
        R = positions[-1]
        print("R:", R)

        positions = positions[:-1].reshape((num_circles, 2))
        penalty = R ** 0.1  # 初始罚分，惩罚大圆的半径

        # 计算小圆之间的重叠罚分
        for i in range(num_circles):
            for j in range(i + 1, num_circles):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 2 * r:
                    penalty += (2 * r - dist) ** 2

        # 计算小圆在大圆外的罚分
        for i in range(num_circles):
            dist_to_center = np.linalg.norm(positions[i])
            if dist_to_center > R - r:
                penalty += (dist_to_center - (R - r)) ** 2

        return penalty

    # 优化
    x0 = np.concatenate((initial_positions.flatten(), [R]))
    print("Initial x0:", x0)
    result = minimize(objective, x0, method='L-BFGS-B')
    print("Final res:", result)

    final_positions = result.x[:-1].reshape((num_circles, 2))
    final_R = result.x[-1]
    return final_positions, final_R


def circle_packing(r, initial_positions):
    num_circles = len(initial_positions)
    # 定义目标函数

    def objective(positions):
        positions = positions.reshape((num_circles, 2))
        R = r + np.max([np.linalg.norm(p) for p in positions])
        # print("R:", R)
        return R

    cons = []
    # 计算小圆之间的重叠约束
    for i in range(num_circles):
        for j in range(i + 1, num_circles):

            def con_fun(positions, i=i, j=j):
                positions = positions.reshape((num_circles, 2))
                dist = np.linalg.norm(positions[i] - positions[j])
                return dist - 2 * r

            cons.append({'type': 'ineq', 'fun': con_fun})

    bounds = [(-10, 10)] * (num_circles * 2)

    # 优化
    # result = minimize(objective, initial_positions.flatten(),
    #                   method='L-BFGS-B', constraints=cons, bounds=bounds)
    result = minimize(objective, initial_positions.flatten(),
                      method='SLSQP', constraints=cons, bounds=bounds)
    print("Final res:", result)

    final_positions = result.x.reshape((num_circles, 2))
    final_R = result.fun
    return final_positions, final_R


def plot_circles(R, r, final_R, final_positions):
    # 绘图
    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), R, color='b', fill=False)
    ax.add_artist(circle)
    circle = plt.Circle((0, 0), final_R, color='purple', fill=False)
    ax.add_artist(circle)

    for pos in final_positions:
        c = plt.Circle(pos, r, color='r', fill=True)
        ax.add_artist(c)

    ax.set_xlim(-R-1, R+1)
    ax.set_ylim(-R-1, R+1)
    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    # 大圆的半径
    R = 10
    # 小圆的数量和半径
    num_circles = 50
    r = 1

    # 初始化小圆的位置（随机）
    np.random.seed(52)
    initial_positions = np.random.uniform(-R + r, R - r, (num_circles, 2))

    t = time.time()
    # final_positions, final_R  = circle_packing_unconstraint(R, r, num_circles, initial_positions)
    final_positions, final_R = circle_packing(r, initial_positions)
    print("Final R:", final_R)
    print("Final positions:", final_positions)
    print("Time:", time.time() - t)

    plot_circles(R, r, final_R, final_positions)
