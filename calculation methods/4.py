import numpy as np

def gauss(sole):
    n = len(sole)
    m = len(sole[0])
    x_coords = list(range(n))
    determinant = 1
    row_swaps = 0

    print("расширенная матрица СЛАУ")
    for v in sole:
        print('\t'.join(f"{x:.10f}" for x in v))
    print("номера неизвестных")
    print('\t'.join(str(x) for x in x_coords))
    print()

    eps = 1e-6

    # прямой
    for j in range(m - 1):
        mx = j
        for i in range(j, n):
            if abs(sole[i][j]) > abs(sole[mx][j]):
                mx = i
        if abs(sole[mx][j]) < eps:
            print("Определитель равен 0, система не имеет решения.")
            return []

        if mx != j:
            sole[j], sole[mx] = sole[mx], sole[j]
            x_coords[j], x_coords[mx] = x_coords[mx], x_coords[j]
            row_swaps += 1
            determinant *= -1

        # под главной диагональю 0
        for i in range(j + 1, n):
            d = sole[i][j] / sole[j][j]
            for k in range(m):
                sole[i][k] -= sole[j][k] * d

    print("прямой")
    for v in sole:
        print('\t'.join(f"{x:.10f}" for x in v))
    print("номера неизвестных:")
    print('\t'.join(str(x) for x in x_coords))
    print()

    for i in range(n):
        determinant *= sole[i][i]

    # обратный
    for j in range(n - 1, -1, -1):
        for i in range(j - 1, -1, -1):
            d = sole[i][j] / sole[j][j]
            sole[i][j] -= sole[j][j] * d
            sole[i][m - 1] -= sole[j][m - 1] * d

    print("обратный")
    for v in sole:
        print('\t'.join(f"{x:.10f}" for x in v))
    print()

    s = [0] * n
    for i in range(n):
        s[x_coords[i]] = sole[i][m - 1] / sole[i][i]  # получение значений переменных

    print(f"определитель матрицы: {determinant:.10f}")
    return s

def gauss_with_b(A, B):
    for i in range(len(A)):
        A[i].append(B[i])  # добавление вектора свободных членов к матрице
    return gauss(A)

def solve():
    n = 5
    v = 1
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i][j] = (v + i) / 100
        A[i][i] = v + i  # диагональ

    V = np.array([v + i for i in range(n)])  # переменные
    B = np.zeros(n)  # свободные члены

    for i in range(n):
        B[i] = sum(A[i][j] * V[j] for j in range(n))

    g = gauss_with_b(A.tolist(), B.tolist())
    print("решение")
    print('\t'.join(f"{x:.10f}" for x in g))

if __name__ == "__main__":
    solve()
