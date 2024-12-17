import numpy as np

# 1
matrix = np.random.randint(0, 100, (2, 10))
print("Матрица:")
print(matrix)

# 2
try:
    element = matrix[3, 7]
    print(f"Элемент [3, 7]: {element}")
except IndexError:
    print("нет его")

# 3
print("вторая строка:")
print(matrix[1])

# 4
print("каждый второй элемент первой строки:")
print(matrix[0, ::2])

# 5
resize_matrix = matrix.reshape(5, 4)
print("(5x4):")
print(resize_matrix)

# 6
st = 2
st_matrix = resize_matrix ** st
print(f"в степень {st}:")
print(st_matrix)

# 7
min_in_columns = np.min(resize_matrix, axis=0)
print("минимальный в каждом столбце:")
print(min_in_columns)

# 8
max_in_last_row = np.max(resize_matrix[-1])
print("максимум в полседней строке:")
print(max_in_last_row)

# 9
array = np.random.randint(0, 2, 20)
print("массив:")
print(array)
zeros = any(array[i] == 0 and array[i+1] == 0 for i in range(len(array) - 1))
print("есть 00:", zeros)

# 10
matrix2 = np.random.randint(0, 100, size=(5, 5))

# Вывод сгенерированной матрицы
print("Сгенерированная матрица:")
print(matrix2)

# Результирующий вектор
result_vector = []

# Обработка каждой строки
for row in matrix2:
    max_index = len(row) - 1 - np.argmax(row[::-1])
    result_vector.extend(row[:max_index])

print(list(map(int, result_vector))) 

