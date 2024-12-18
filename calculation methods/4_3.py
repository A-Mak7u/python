import numpy as np
V = 1
eps = 1e-6
def f(x):
    return 4 * x ** 4 - 3 * V * x ** 3 + 6 * x - 2 * V
def p(x):
    return x ** 2
def q(x):
    return x
def check_y(x):
    return x ** 2 * (x - V)
def phi(x, k):
    k += 1
    return (x ** (k + 1)) * (x - V)
def phi1(x, k):
    k += 1
    return (x ** (k + 1)) * (k + 2) - (x ** k) * (k + 1) * V
def phi2(x, k):
    k += 1
    return (x ** k) * (k + 2) * (k + 1) - (x ** (k - 1)) * (k + 1) * k * V
def gauss(A, B):
    n = len(A)
    for i in range(n):
        # макс элемент для устойчивости
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[max_row] = A[max_row], A[i]
        B[i], B[max_row] = B[max_row], B[i]
        for j in range(i + 1, n):
            if abs(A[j][i]) > eps:
                factor = A[j][i] / A[i][i]
                A[j] = [A[j][k] - factor * A[i][k] for k in range(n)]
                B[j] -= factor * B[i]
    # обратк
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (B[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x
def print_results(vx, vy):
    k = 10
    for i in range(0, len(vx), k):
        m = min(i + k, len(vx))
        for j in range(i, m):
            print(f"{vx[j]:.5f}", end=" \t")
        print()
        for j in range(i, m):
            print(f"{vy[j]:.5f}", end=" \t")
        print()
        for j in range(i, m):
            print(f"{check_y(vx[j]):.5f}", end=" \t")
        print()
        for j in range(i, m):
            print(f"{vy[j] - check_y(vx[j]):.5f}", end=" \t")
        print()
        print()
def solve():
    n = 8
    l, r = 0, V
    h = (r - l) / (n + 1)
    # сетк
    vx = [l + h * (i + 1) for i in range(n)]
    A = np.zeros((n, n))
    B = np.zeros(n)
    for j in range(n):
        x = vx[j]
        for k in range(n):
            A[j][k] = phi2(x, k) + p(x) * phi1(x, k) + q(x) * phi(x, k)
        B[j] = f(x)
    va = gauss(A.tolist(), B.tolist())
    # границы и значения
    vx = [0] + vx + [V]
    n = len(vx)
    vy = np.zeros(n)
    for i in range(n):
        for k in range(len(va)):
            vy[i] += va[k] * phi(vx[i], k)
    print_results(vx, vy)
if __name__ == "__main__":
    solve()