import numpy as np
import matplotlib.pyplot as plt
from task1 import *


def least_squares_1d(x, y, rho, degree):
    n = degree + 1
    N = len(x)

    M = np.zeros((n, n))
    B = np.zeros(n)

    for i in range(N):
        powers = [1.0]
        for k in range(1, 2 * degree + 1):
            powers.append(powers[-1] * x[i])

        for k in range(n):
            for j in range(n):
                M[k][j] += rho[i] * powers[k + j]
            B[k] += rho[i] * y[i] * powers[k]

    coeff, _, _, _ = np.linalg.lstsq(M, B, rcond=None)
    return coeff


def nonlinear_fit(x, y, func_type):
    if func_type == 1:
        # y = a * x^b -> ln y = ln a + b * ln x
        X = np.log(x)
        Y = np.log(y)
        coeff = least_squares_1d(X, Y, np.ones(len(x)), 1)
        a = np.exp(coeff[0])
        b = coeff[1]
        return lambda x: a * (x ** b), (a, b)

    elif func_type == 2:
        # y = a * e^(b*x) -> ln y = ln a + b * x
        X = x
        Y = np.log(y)
        coeff = least_squares_1d(X, Y, np.ones(len(x)), 1)
        a = np.exp(coeff[0])
        b = coeff[1]
        return lambda x: a * np.exp(b * x), (a, b)

    elif func_type == 3:
        # y = a + b/x
        X = 1.0 / x
        Y = y
        coeff = least_squares_1d(X, Y, np.ones(len(x)), 1)
        a = coeff[0]
        b = coeff[1]
        return lambda x: a + b / x, (a, b)

    elif func_type == 4:
        # y = a0 / (a1 + a2*x) -> 1/y = (a1/a0) + (a2/a0)*x
        X = x
        Y = 1.0 / y
        coeff = least_squares_1d(X, Y, np.ones(len(x)), 1)
        A = coeff[0]
        B = coeff[1]
        a0 = 1.0
        a1 = A
        a2 = B
        return lambda x: a0 / (a1 + a2 * x), (a0, a1, a2)



    elif func_type == 5:

        # y = ln(a*x + b)  ->   e^y = a*x + b
        y_min = np.min(y)
        shift = (abs(y_min) + 1.0) if y_min <= 0 else 0.0
        y_shifted = y + shift
        X = x
        Y = np.exp(y_shifted)
        coeff = least_squares_1d(X, Y, np.ones(len(x)), 1)
        a = coeff[1]
        b = coeff[0]
        def log_func(x_val):
            val = a * x_val + b
            if val <= 0:
                return np.nan
            return np.log(val) - shift
        return log_func, (a, b, shift)


def demonstrate_nonlinear():
    print("Демонстрация подбора нелинейных зависимостей")
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    y = np.array([10.5, 1.6, 0.55, 0.26, 0.15, 0.08])

    print("Табличные данные:")
    for i in range(len(x)):
        print(f"  x={x[i]:.1f}, y={y[i]:.3f}")

    functions = [
        (1, "y = a * x^b"),
        (2, "y = a * e^(b*x)"),
        (3, "y = a + b/x"),
        (4, "y = a0 / (a1 + a2*x)"),
        (5, "e^y = a*x + b  ->  y = ln(a*x + b)")
    ]

    results = []

    for func_type, name in functions:
        try:
            f, params = nonlinear_fit(x, y, func_type)
            err = 0
            for i in range(len(x)):
                val = f(x[i])
                if np.isfinite(val):
                    err += (y[i] - val) ** 2
                else:
                    err += 1e10
            err = np.sqrt(err / len(x))

            results.append((func_type, name, params, err))
            print(f"\n{name}")
            print(f"  Параметры: {params}")
            print(f"  Среднеквадратичная ошибка: {err:.6f}")
        except Exception as e:
            print(f"\n{name}: ошибка - {e}")

    best = min(results, key=lambda r: r[3])
    print(f"\n{'=' * 50}")
    print(f"ЛУЧШАЯ ФУНКЦИЯ: {best[1]}")
    print(f"Ошибка: {best[3]:.6f}")

    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, c='k', marker='o', s=100, label='Исходные точки', zorder=5)

    x_plot = np.linspace(0.4, 3.1, 100)
    colors = ['r', 'g', 'b', 'm', 'c']
    markers = ['-', '--', '-.', ':', '-']

    for (func_type, name, params, err), color, marker in zip(results, colors, markers):
        try:
            f, _ = nonlinear_fit(x, y, func_type)
            y_plot = [f(xi) for xi in x_plot]

            y_plot = np.array(y_plot)
            mask = np.isfinite(y_plot)

            plt.plot(np.array(x_plot)[mask], y_plot[mask], color=color,
                     linestyle=marker, linewidth=2, label=f'{name} (err={err:.4f})')
        except:
            pass

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Сравнение нелинейных зависимостей')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    plt.show()
