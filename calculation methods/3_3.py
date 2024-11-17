import numpy as np


def tridiagonal_algorithm(a, b, c, d):
    n = len(a)
    p = [0] * (n + 1)
    q = [0] * (n + 1)

    #
    for i in range(n):
        denom = b[i] - a[i] * p[i]
        p[i + 1] = c[i] / denom
        q[i + 1] = (a[i] * q[i] - d[i]) / denom

    print("Прямая прогонка:")
    for i in range(1, n):
        print(f"( {p[i]:.4f}, {q[i]:.4f} )")
    print(f"( {q[n]:.4f} )\n")

    ##
    res = [0] * n
    res[n - 1] = q[n]
    for i in range(n - 2, -1, -1):
        res[i] = p[i + 1] * res[i + 1] + q[i + 1]

    return res


def solve():
    n = 5
    v = 1
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(max(0, i - 1), min(i + 2, n)):
            A[i][j] = (v + i) / 100
        A[i][i] = v + i

    V = np.array([v + i for i in range(n)])
    B = np.zeros(n)
    for i in range(n):
        B[i] = sum(A[i][j] * V[j] for j in range(max(0, i - 1), min(i + 2, n)))

    print("Расширенная матрица A|B:")
    for i in range(n):
        print('\t'.join(f"{A[i][j]:.4f}" for j in range(n)), "|", f"{B[i]:.4f}")
    print()

    #A и B в векторы a, b, c и d
    a = [0] * n
    b = [0] * n
    c = [0] * n
    d = B.tolist()

    for i in range(n):
        if i > 0:
            a[i] = A[i][i - 1]  # поддиагональный элемент
        b[i] = -A[i][i]  # главный диагональный элемент
        if i < n - 1:
            c[i] = A[i][i + 1]  # наддиагональный элемент

    X = tridiagonal_algorithm(a, b, c, d)

    print("Решение СЛАУ:")
    print('\t'.join(f"{x:.4f}" for x in X))


if __name__ == "__main__":
    solve()
