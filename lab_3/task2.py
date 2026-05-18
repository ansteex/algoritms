import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from task3 import *

def read_data_2d(filename):
    """Чтение двумерных данных из файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
        return None, None, None, None

    x = []
    y = []
    z = []
    rho = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('#'):
            continue

        parts = line.split()

        if len(parts) >= 4:
            z.append(float(parts[0]))
            x.append(float(parts[1]))
            y.append(float(parts[2]))
            rho.append(float(parts[3]))
        elif len(parts) == 3:
            z.append(float(parts[0]))
            x.append(float(parts[1]))
            y.append(float(parts[2]))
            rho.append(1.0)
        else:
            print(f"Пропущена строка: {line}")

    return np.array(x), np.array(y), np.array(z), np.array(rho)


def least_squares_2d_linear(x, y, z, rho):
    """МНК для линейной двумерной аппроксимации"""
    n = 3
    N = len(x)

    M = np.zeros((n, n))
    B = np.zeros(n)

    for i in range(N):
        phi = [1.0, x[i], y[i]]

        for k in range(n):
            for j in range(n):
                M[k][j] += rho[i] * phi[k] * phi[j]
            B[k] += rho[i] * z[i] * phi[k]

    coeff = np.linalg.solve(M, B)
    return coeff


def least_squares_2d_quadratic(x, y, z, rho):
    """МНК для квадратичной двумерной аппроксимации"""
    n = 6
    N = len(x)

    M = np.zeros((n, n))
    B = np.zeros(n)

    for i in range(N):
        phi = [1.0, x[i], y[i], x[i] ** 2, x[i] * y[i], y[i] ** 2]

        for k in range(n):
            for j in range(n):
                M[k][j] += rho[i] * phi[k] * phi[j]
            B[k] += rho[i] * z[i] * phi[k]

    coeff = np.linalg.solve(M, B)
    return coeff


def polyval_2d_linear(coeff, x, y):
    """Вычисление значения линейной поверхности"""
    return coeff[0] + coeff[1] * x + coeff[2] * y


def polyval_2d_quadratic(coeff, x, y):
    """Вычисление значения квадратичной поверхности"""
    return coeff[0] + coeff[1] * x + coeff[2] * y + coeff[3] * x ** 2 + coeff[4] * x * y + coeff[5] * y ** 2


def demonstrate_2d():
    """Демонстрация двумерной аппроксимации"""
    print("Демонстрация двумерной аппроксимации")
    x, y, z, rho = read_data_2d("3.txt")

    if x is None:
        return

    coeff_linear = least_squares_2d_linear(x, y, z, rho)
    coeff_quad = least_squares_2d_quadratic(x, y, z, rho)

    print(f"\nЛинейный полином: z = {coeff_linear[0]:.4f} + {coeff_linear[1]:.4f}x + {coeff_linear[2]:.4f}y")
    print(f"Квадратичный полином: z = {coeff_quad[0]:.4f} + {coeff_quad[1]:.4f}x + {coeff_quad[2]:.4f}y + "
          f"{coeff_quad[3]:.4f}x^2 + {coeff_quad[4]:.4f}xy + {coeff_quad[5]:.4f}y^2")

    # Вычисление ошибок
    err_linear = 0
    err_quad = 0
    for i in range(len(x)):
        z_lin = polyval_2d_linear(coeff_linear, x[i], y[i])
        z_quad = polyval_2d_quadratic(coeff_quad, x[i], y[i])
        err_linear += (z[i] - z_lin) ** 2
        err_quad += (z[i] - z_quad) ** 2

    err_linear = np.sqrt(err_linear / len(x))
    err_quad = np.sqrt(err_quad / len(x))

    print(f"\nСреднеквадратичная ошибка (линейная): {err_linear:.4f}")
    print(f"Среднеквадратичная ошибка (квадратичная): {err_quad:.4f}")

    # Построение графиков
    x_grid = np.linspace(min(x), max(x), 20)
    y_grid = np.linspace(min(y), max(y), 20)
    X, Y = np.meshgrid(x_grid, y_grid)

    Z_linear = np.zeros_like(X)
    Z_quad = np.zeros_like(X)
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            Z_linear[i, j] = polyval_2d_linear(coeff_linear, X[i, j], Y[i, j])
            Z_quad[i, j] = polyval_2d_quadratic(coeff_quad, X[i, j], Y[i, j])

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(x, y, z, c='r', marker='o', alpha=0.7)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Исходные точки')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(x, y, z, c='r', marker='o', alpha=0.3, label='Точки')
    ax2.plot_surface(X, Y, Z_linear, alpha=0.6, cmap='plasma')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Линейная аппроксимация (плоскость)\nОшибка = {err_linear:.4f}')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(x, y, z, c='r', marker='o', alpha=0.3, label='Точки')
    ax3.plot_surface(X, Y, Z_quad, alpha=0.6, cmap='viridis')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'Квадратичная аппроксимация\nОшибка = {err_quad:.4f}')

    plt.tight_layout()
    plt.show()