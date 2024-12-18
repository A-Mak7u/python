import numpy as np
eps = 1e-6
V = 1
def f(x, y, v):
    return x**3 - x**2 * (3 + v) + 2 * v * x + y
def check_y(x, V):
    return x**2 * (V - x)
# 1
def euler(n, h, x0, y0, v):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = x0, y0
    for i in range(n - 1):
        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + h * f(x[i], y[i], v)
    return x, y
# 2
def ex_euler(n, h, x0, y0, v):
    x = np.zeros(2 * n)
    y = np.zeros(2 * n)
    x[0], y[0] = x0, y0
    for i in range(2 * n - 1):
        x[i + 1] = x[i] + h / 2
        if i % 2:
            y[i + 1] = y[i - 1] + h * f(x[i], y[i], v)
        else:
            y[i + 1] = y[i] + (h / 2) * f(x[i], y[i], v)
    return x[::2], y[::2]
def print_results(vx, vy, V):
    print("x:", "\t".join(f"{val:.5f}" for val in vx))
    print("y:", "\t".join(f"{val:.5f}" for val in vy))
    exact_y = [check_y(x, V) for x in vx]
    print("Точное значение y:", "\t".join(f"{val:.5f}" for val in exact_y))
    deviation = [vy[i] - exact_y[i] for i in range(len(vy))]
    print("Отклонение:", "\t".join(f"{val:.5f}" for val in deviation))
    max_deviation = max(abs(dev) for dev in deviation)
    print(f"Максимальное отклонение: {max_deviation:.5f}")
def solve():
    n = 13
    h = 0.1
    x0 = 1
    y0 = 0
    v = 1.0
    print("Метод Эйлера:")
    vx, vy1 = euler(n, h, x0, y0, v)
    print_results(vx, vy1, V)
    print()
    print("Расширенный метод Эйлера:")
    _, vy2 = ex_euler(n, h, x0, y0, v)
    print_results(vx, vy2, V)
    print()
if __name__ == "__main__":
    solve()