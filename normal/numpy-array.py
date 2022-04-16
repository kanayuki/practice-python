import numpy as np

# %% list 与 ndarray 的区别
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
na = np.array(a)

print(type(a))
print(type(na))

print(na[1:, 1:])
print(a[1:, 1:])
