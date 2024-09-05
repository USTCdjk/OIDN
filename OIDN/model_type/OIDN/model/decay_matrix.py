import numpy as np

def decay_matrix(size,sigma):
    # 创建一个空的矩阵
    matrix = np.zeros((size, size))

    # 计算矩阵中心点的坐标
    center = (size - 1) // 2


    # 在矩阵中心点设置最大值
    # matrix[center, center] = np.max(weight)

    # 向四周衰减
    for i in range(size):
        for j in range(size):
            matrix[center, center] = 1
            distance = np.sqrt((i-center)**2 + (j-center)**2)
            weight = np.exp(-0.5 * (distance / sigma))
            matrix[i, j] = matrix[center, center] * weight
    return matrix