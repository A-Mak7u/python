import numpy as np
from scipy.optimize import minimize_scalar

def gradient_descent(f, f_grad, x0, alpha=1.0, tol=1e-6, max_iter=1000):

    x = x0.copy()
    f_x = f(x)

    for i in range(max_iter):
        grad = f_grad(x)  # в текущей точке
        grad_norm = np.linalg.norm(grad)  # длина вектора градиента

        # условие остановки
        if grad_norm < tol:
            break

        y = x - alpha * grad
        f_y = f(y)

        if f_y < f_x:
            x = y
            f_x = f_y
        else:
            alpha *= 0.5

    return x, i + 1




def steepest_descent(f, f_grad, x0, tol=1e-6, max_iter=1000):

    x = x0.copy()
    for i in range(max_iter):
        grad = f_grad(x)
        grad_norm = np.linalg.norm(grad)  # длина вектора градиента

        if grad_norm < tol:
            break

        # минимизируем f(x - alpha * grad)
        result = minimize_scalar(lambda alpha: f(x - alpha * grad))
        alpha = result.x

        x -= alpha * grad

    return x, i + 1

def newton_method(f_grad, f_hess, x0, tol=1e-6, max_iter=1000):
    x = x0.copy()

    for i in range(max_iter):
        grad = f_grad(x)
        hess = f_hess(x)

        if np.linalg.norm(grad) < tol:
            break

        hess_inv = np.linalg.inv(hess)
        d = hess_inv @ grad
        x -= d
    return x, i+1



# функция
def func(x):
    x1, x2 = x
    return 99*x1**2 + 126*x1*x2 + 64*x2**2 - 10*x1 + 30*x2 + 13

# градиент
def grad_func(x):
    x1, x2 = x
    df_dx1 = 198*x1 + 126*x2 - 10
    df_dx2 = 126*x1 + 128*x2 + 30
    return np.array([df_dx1, df_dx2])

# матрица Гесса
def hess_func(x):
    return np.array([[198, 126], [126, 128]])

x0 = np.array([0.0, 0.0])

x_gd, iter_gd = gradient_descent(func, grad_func, x0)
print("Градиентный спуск с адаптивным шагом:")
print(f"Минимум: {x_gd}, Итераций: {iter_gd}\n")

x_sd, iter_sd = steepest_descent(func, grad_func, x0)
print("Наискорейший спуск:")
print(f"Минимум: {x_sd}, Итераций: {iter_sd}\n")

x_nm, iter_nm = newton_method(grad_func, hess_func, x0)
print("Метод Ньютона:")
print(f"Минимум: {x_nm}, Итераций: {iter_nm}\n")
