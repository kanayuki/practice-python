import numpy as np

def rotation_matrix_from_vectors(u, v):
    # 将向量归一化
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    
    # 计算旋转轴
    axis = np.cross(u, v)
    axis_norm = np.linalg.norm(axis)
    
    # 计算旋转角度
    angle = np.arccos(np.dot(u, v))
    
    # 处理平行和反平行的情况
    if axis_norm == 0:
        if np.dot(u, v) > 0:
            return np.eye(3)  # 两个向量相同，返回单位矩阵
        else:
            return -np.eye(3)  # 两个向量相反，返回负单位矩阵
    
    axis = axis / axis_norm

    # Rodrigues' 旋转公式
    
    x,y,z= axis
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    
    M = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return M

# 示例向量
u = np.array([1, 3, 7])
v = np.array([2, 4, 3])

# 计算旋转矩阵
R = rotation_matrix_from_vectors(u, v)
print("旋转矩阵:")
print(R)
