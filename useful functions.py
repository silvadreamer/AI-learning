import numpy as np

# 初始化矩阵,全为0
matrix = np.zeros((2, 3))
# 获得矩阵的行数
print(matrix.shape[0])
# 矩阵整体的复制
a = np.array([0, 1, 2])

np.tile(a, 2)  # array([0, 1, 2, 0, 1, 2]

np.tile(a, (2, 2))  # array([[0, 1, 2, 0, 1, 2],
                        #    [0, 1, 2, 0, 1, 2]])

np.tile(a, (2, 1, 2))  # array([[[0, 1, 2, 0, 1, 2]],
                        #       [[0, 1, 2, 0, 1, 2]]])
test = np.zeros((1, 20))
for i in range(20):
    test[0, i] = int(1)
    print(test)


mat = np.array([[1, 2], [2, 3]])
print(mat * mat)
