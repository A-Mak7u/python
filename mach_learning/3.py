import numpy as np

N = 3
x = np.array([1, 2, 3])
A = np.random.randint(1, 10, (N, 2 * N))
print(A)

A_left = A[:, :N]       # левая
A_right = A[:, N:]      # правая

A_left_result = A_left * x[:, None]

x_reversed = x[::-1]
A_right_result = A_right * x_reversed

det_left = np.linalg.det(A_left_result)
det_right = np.linalg.det(A_right_result)

result = det_left + det_right

print(A_left_result)
print(A_right_result)

print(f"Определитель левой части: {det_left}")
print(f"Определитель правой части: {det_right}")
print(f"Сумма определителей: {result}")
