import numpy as np
V = 1
eps = 1e-6
def f(x):
    return V * (4 / 3 * x + 1 / 4 * x ** 2 + 1 / 5 * x ** 3)
def check_y(x):
    return V * x
def gauss(A, B):
    n = len(A)
    m = len(A[0])
    x_coords = list(range(n))
    for j in range(m - 1):
        mx = j
        for i in range(j, n):
            if abs(A[i][j]) > abs(A[mx][j]):
                mx = i
        if abs(A[mx][j]) < eps:
            A[j], A[mx] = A[mx], A[j]
        else:
            for i in range(j + 1, n):
                d = A[i][j] / A[j][j]
                A[i][j:] = np.array(A[i][j:]) - np.array(A[j][j:]) * d
    for j in range(n - 1, -1, -1):
        for i in range(j - 1, -1, -1):
            d = A[i][j] / A[j][j]
            A[i][j] -= A[j][j] * d
            B[i] -= B[j] * d
    x = np.zeros(n)
    for i in range(n):
        x[x_coords[i]] = B[i] / A[i][i]
    return x
def solve():
    n = 3
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i][j] = 1 / (i + j + 3)
        A[i][i] += 1
    B = np.zeros(n)
    for i in range(n):
        B[i] = (4 * V / 3 / (i + 3) + V / 4 / (i + 4) + V / 5 / (i + 5))
    vq = gauss(A.tolist(), B.tolist())
    m = 25
    l, r = 0, 1
    vx = np.linspace(l, r, m)
    vy = np.zeros(m)
    for i in range(m):
        vy[i] = f(vx[i])
        for k in range(n):
            vy[i] -= vq[k] * vx[i] ** (k + 1)
    print(f"{'x':<10}{'y(x)':<10}{'y(точн)':<10}{'delta':<10}")
    for i in range(m):
        print(f"{vx[i]:<10.5f}{vy[i]:<10.5f}{check_y(vx[i]):<10.5f}{vy[i] - check_y(vx[i]):<10.5f}")
if __name__ == "__main__":
    solve()