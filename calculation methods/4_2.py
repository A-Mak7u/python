import numpy as np

eps = 1e-6
V = 1


def f(x):
    return 4 * x ** 4 - 3 * V * x ** 3 + 6 * x - 2 * V


#def p(x):
#    return x ** 2


#def q(x):
#    return x


def check_y(x):
    return x * x * (x - V)


def tridiagonal_algorithm(a, b, c, d):
    n = len(a)
    p = np.zeros(n + 1)
    q = np.zeros(n + 1)

    # Прямой ход
    for i in range(n):
        p[i + 1] = c[i] / (b[i] - a[i] * p[i])
        q[i + 1] = (a[i] * q[i] - d[i]) / (b[i] - a[i] * p[i])

    # Обратный ход
    res = np.zeros(n)
    res[n - 1] = q[n]
    for i in range(n - 2, -1, -1):
        res[i] = p[i + 1] * res[i + 1] + q[i + 1]

    return res


def print_results(vx, vy):
    k = 10
    for i in range(0, len(vx), k):
        m = min(i + k, len(vx))
        print("\t".join(f"{val:.10f}" for val in vx[i:m]))
        print("\t".join(f"{val:.10f}" for val in vy[i:m]))
        print("\t".join(f"{check_y(val):.10f}" for val in vx[i:m]))
        print("\t".join(f"{(vy[j] - check_y(vx[j])):.10f}" for j in range(i, m)))
        print()

    max_deviation = max(abs(vy[i] - check_y(vx[i])) for i in range(len(vy)))
    print(f"Максимальное отклонение: {max_deviation:.10f}")


def solve():
    n = 100
    l = 0
    r = V
    h = (r - l) / n

    #
    vx = np.linspace(l, r, n + 1)

    a = np.zeros(n - 1)
    b = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):
        x = vx[i + 1]
        a[i] = (1 / h / h - x * x / (2 * h))
        b[i] = -(-2 / h / h + x)
        c[i] = (1 / h / h + x * x / (2 * h))
        d[i] = f(x)

    a[0] = 0
    c[-1] = 0

    vy = tridiagonal_algorithm(a, b, c, d)

    vy = np.concatenate(([0], vy, [0]))

    print("Результаты:")
    print_results(vx, vy)


if __name__ == "__main__":
    solve()
