import numpy as np

A = np.random.randint(0, 100, size=(4, 5))  # Матрица со случайными числами от 1 до 10

total_sum = np.sum(A)

column_sums = np.sum(A, axis=0)

column_shares = column_sums / total_sum

result_matrix = np.vstack([A, column_shares])


print(A)

print("\nСумма всех элементов матрицы:", total_sum)

print("\nСумма элементов каждого столбца:")
print(column_sums)

print("\nДоля суммы каждого столбца в общей сумме:")
print(column_shares)

print(result_matrix)
