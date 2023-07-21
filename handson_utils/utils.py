import numpy as np

x = np.array([[1,2,-1], [1, 0, 1], [0, 1, 0]])

det = np.linalg.det(x)

# print(f"determitant = {det}")
x_inverse = np.linalg.inv(x)
# print (f"Inverse of X is : {x_inverse}")

id_matrix = np.array([[1,0,1], [0, 1, 0], [0, 0, 1]])

# print(f"ID mat * inverse of X = {np.matmul(x_inverse, id_matrix)}")

a = np.array([[5,2,3], [-1, -3, 2], [0, 1, -1]])
b = np.array([[1,0,-4], [2, 1, 0], [8, -1, 0]])
a_b = np.matmul(a, b)
print(f"a * b = {a_b}")
det_a_b = np.linalg.det(a_b)
print(f"inverse of ab = {np.linalg.inv(a_b)}")
print(f"det of a_b = {det_a_b}")
print(f"det of inverse of a_b = {np.linalg.det(np.linalg.inv(a_b))}")
