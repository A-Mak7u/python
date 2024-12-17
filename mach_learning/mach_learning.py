import numpy as np


def f(x):
    return (x + 3) * (x + 1) * (x - 2) * (x - 3) + 2

# Метод кусочно-линейной аппроксимации
def piecewise_linear_approximation(f, a, b, L, tol=1e-6, max_iter=100):

    # Начальные значения
    x_prev = None
    x_min = (a + b) / 2  # начальное приближение минимума
    iter_count = 0

    while iter_count < max_iter:
        iter_count += 1

        # Построение вспомогательной функции g(x, x_min)
        g = lambda x: f(x_min) - L * abs(x - x_min)

        # Вычисляем значения p(x) на концах интервала
        p_a = f(a) - L * (x_min - a)
        p_b = f(b) - L * (b - x_min)

        # Поиск глобального минимума кусочно-линейной функции
        x0 = (f(a) - f(b) + L * (a + b)) / (2 * L)
        p0 = (f(a) + f(b) + L * (b - a)) / 2

        # Если точность достигнута, выходим
        if x_prev is not None and abs(x_min - x_prev) < tol:
            break

        x_prev = x_min
        x_min = x0 if a <= x0 <= b else (a if f(a) < f(b) else b)

        # Обновление границ интервала
        a, b = (a, x_min) if f(a) < f(b) else (x_min, b)

        print(f"Итерация {iter_count}: x_min = {x_min}, f(x_min) = {f(x_min)}")

    return x_min, f(x_min)


a, b = -3, 3
L = 100
tol = 1e-6
x_min, f_min = piecewise_linear_approximation(f, a, b, L, tol)

print(f"Глобальный минимум найден: x = {x_min}, f(x) = {f_min}")
