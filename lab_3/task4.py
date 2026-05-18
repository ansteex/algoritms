import numpy as np
import matplotlib.pyplot as plt
import math
def fit_ode_by_least_squares(m):
    """Решение ОДУ методом Галеркина через МНК (как в main_V.py)"""
    grid = np.linspace(0.0, 1.0, 1200)

    def residual(poly):
        d1 = np.polyder(poly, 1)
        d2 = np.polyder(poly, 2)
        return d2(grid) + grid * d1(grid) + poly(grid) - 2.0 * grid

    u0 = np.poly1d([-1.0, 1.0])
    basis = [np.poly1d([1.0, 0.0]) ** k * np.poly1d([-1.0, 1.0]) for k in range(1, m + 1)]

    r0 = residual(u0)
    residual_basis = [residual(poly) for poly in basis]

    matrix = np.zeros((m, m), dtype=float)
    rhs = np.zeros(m, dtype=float)
    for i in range(m):
        for j in range(m):
            matrix[i, j] = np.trapezoid(residual_basis[i] * residual_basis[j], grid)
        rhs[i] = -np.trapezoid(r0 * residual_basis[i], grid)

    coefficients = np.linalg.solve(matrix, rhs)
    y = u0(grid)
    for coefficient, poly in zip(coefficients, basis):
        y += coefficient * poly(grid)

    total_residual = r0.copy()
    for coefficient, basis_residual in zip(coefficients, residual_basis):
        total_residual += coefficient * basis_residual
    residual_norm = math.sqrt(np.trapezoid(total_residual**2, grid))
    return grid, y, residual_norm


def demonstrate_ode():
    print("Демонстрация решения ОДУ методом Галеркина (через МНК)")

    print("Уравнение: y'' + x*y' + y = 2x")
    print("Краевые условия: y(0)=1, y(1)=0")

    x2, y2, norm2 = fit_ode_by_least_squares(2)
    x3, y3, norm3 = fit_ode_by_least_squares(3)

    print(f"\nРезультаты:")
    print(f"  m=2: норма невязки = {norm2:.6f}")
    print(f"  m=3: норма невязки = {norm3:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(x2, y2, 'b-', linewidth=2.5, label=f"m=2, ||R||={norm2:.6f}")
    plt.plot(x3, y3, 'r--', linewidth=2.5, label=f"m=3, ||R||={norm3:.6f}")
    plt.scatter([0.0, 1.0], [1.0, 0.0], c="black", zorder=4, label="краевые условия")
    plt.title("Приближённое решение краевой задачи")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()