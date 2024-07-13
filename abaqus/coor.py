import numpy as np
import matplotlib.pyplot as plt


def calculate_position(row=5, col=5, space=1):
    # 计算每个格子的位置
    x = np.linspace(0, w := row * space, row)-w/2
    y = np.linspace(0, h := col * space, col)-h/2
    return np.meshgrid(x, y)


def plot_line():
    # 绘制线
    row = 5
    col = 5
    diameter = 1
    carrier_distance = 30
    height = 100

    top_x, top_y = calculate_position(row, col, space=diameter)
    bottom_x, bottom_y = calculate_position(row, col, space=carrier_distance)
    top_z = np.full((row, col), height)
    bottom_z = np.full((row, col), 0)

    for x, y, z, x2, y2, z2 in np.nditer([top_x, top_y, top_z, bottom_x, bottom_y, bottom_z]):
        print(a := np.c_[x, y, z, x2, y2, z2])
        print(list(a))
        print(list(a[0]))
        print(type(a))
        print(list((x, y, z)))
        print(np.array([x, y, z]))

    # plt.figure(figsize=(10, 10))
    # plt.plot(top_x, top_y, 'b.')
    # plt.plot(bottom_x,bottom_y, 'r.')
    # plt.show()


def um():
    row = 5
    col = 5
    zero_matrix = np.zeros((row, col))
    one_matrix = np.ones((row, col))
    u1s1 = one_matrix.copy()
    u1s1[:, 0::2] = -1
    u1s1[:, [0, -1]] = 0
    u2s1 = zero_matrix

    u1s2 = zero_matrix
    u2s2 = one_matrix.copy()
    u2s2[0::2, :] = -1
    u2s2[[0, -1], :] = 0


    u1s3 = -u1s1
    u2s3 = zero_matrix

    u1s4 = zero_matrix
    u2s4 = -u2s2

    print(u1s1)
    print(u1s2)
    print(u1s3)
    print(u1s4)
    print(u2s1)
    print(u2s2)
    print(u2s3)
    print(u2s4)
    print(u2s4[(1,2)])
    print(u2s4[(2,3)])
    print(u2s4[(3,4)])
    print(u2s4[(0,1)])
    print([1]+[2]+[3]+[4])
    print(u1s1+u1s3)
    print(np.array(1))
    print(list([np.array(1)]))


if __name__ == "__main__":
    # plot_line()
    # um()
    a=np.array([1,2,3])
    b= (4,5,6)
    print ((1,2,3)+b)
    print ("a"+'b')
