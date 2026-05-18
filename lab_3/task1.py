import numpy as np
import matplotlib.pyplot as plt


def gauss_solve(A, b):
    n = len(A)
    a = A.copy().astype(float)
    b_vec = b.copy().astype(float)

    for k in range(n):
        max_row = k
        max_val = abs(a[k][k])
        for i in range(k + 1, n):
            if abs(a[i][k]) > max_val:
                max_val = abs(a[i][k])
                max_row = i

        if max_val < 1e-12:
            raise ValueError("Матрица вырождена")

        if max_row != k:
            a[[k, max_row]] = a[[max_row, k]]
            b_vec[[k, max_row]] = b_vec[[max_row, k]]

        for i in range(k + 1, n):
            factor = a[i][k] / a[k][k]
            for j in range(k, n):
                a[i][j] -= factor * a[k][j]
            b_vec[i] -= factor * b_vec[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += a[i][j] * x[j]
        x[i] = (b_vec[i] - s) / a[i][i]

    return x


def least_squares_1d(x, y, rho, n, lambda_reg=4e-11 + 2e-12):
    N = len(x)
    n_coeffs = n + 1
    A = np.zeros((n_coeffs, n_coeffs))
    b = np.zeros(n_coeffs)

    for i in range(N):
        for k in range(n_coeffs):
            for j in range(n_coeffs):
                A[k, j] += rho[i] * (x[i] ** (k + j))
            b[k] += rho[i] * y[i] * (x[i] ** k)

    for k in range(n_coeffs):
        A[k, k] += lambda_reg

    coeffs = gauss_solve(A, b)
    return coeffs


def poly_value_1d(coeffs, x_val):
    result = 0
    for k, c in enumerate(coeffs):
        result += c * (x_val ** k)
    return result


def polynomial_to_string(coeffs, variable="x"):
    """Преобразование коэффициентов в строку"""
    terms = []
    for power, coefficient in enumerate(coeffs):
        if abs(coefficient) < 1e-10:
            continue
        if power == 0:
            term = f"{coefficient:.6f}"
        elif power == 1:
            term = f"{coefficient:.6f}*{variable}"
        else:
            term = f"{coefficient:.6f}*{variable}^{power}"
        terms.append(term)
    if terms:
        return " + ".join(terms)
    return "0"


def test_1d_approximation():
    print("=" * 70)
    print("ТЕСТ 1D АППРОКСИМАЦИИ")
    print("=" * 70)

    data = np.loadtxt('test1.txt')
    x = data[:, 0]
    y = data[:, 1]
    rho = data[:, 2]

    print("\nИсходные данные:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"p = {rho}")

    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))

    sizes = 100 + 200 * rho / rho.max() if rho.max() > 0 else 100 * np.ones_like(rho)
    scatter1 = ax1.scatter(x, y, s=sizes, c='red', zorder=5, alpha=0.8, edgecolors='black', linewidth=1.5)

    for i, (xi, yi, ri) in enumerate(zip(x, y, rho)):
        ax1.annotate(f'ρ={ri:.1f}', (xi, yi), textcoords="offset points",
                     xytext=(8, 8), fontsize=9, ha='center',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    x_smooth = np.linspace(x.min(), x.max(), 500)

    print("\n" + "=" * 70)
    print("Полином 100-й степени")
    print("=" * 70)

    coeffs_high = least_squares_1d(x, y, rho, 20)

    if coeffs_high is not None:
        print(f"\nКоэффициенты полинома (от младшей степени к старшей):")
        print("-" * 70)
        for i, coeff in enumerate(coeffs_high):
            if abs(coeff) > 1e-10:
                print(f"  c[{i:2d}] = {coeff: .8e}  (x^{i})")
        print("-" * 70)

        print(f"\nПолином в виде формулы:")
        print(f"  y = {polynomial_to_string(coeffs_high)}")

        y_smooth_high = [poly_value_1d(coeffs_high, xv) for xv in x_smooth]

        y_pred = [poly_value_1d(coeffs_high, xi) for xi in x]
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        print(f"\n  RMSE = {rmse:.6f}")
        print("=" * 70)

        ax1.plot(x_smooth, y_smooth_high, 'purple', linewidth=1.5, alpha=0.7,
                 label=f'Полином 100-й степени')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Аппроксимация методом наименьших квадратов', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.show()


def test_1d_approximation1():
    print("=" * 70)
    print("ТЕСТ 1D АППРОКСИМАЦИИ")
    print("=" * 70)

    data = np.loadtxt('test2.txt')
    x = data[:, 0]
    y = data[:, 1]
    rho = data[:, 2]

    print("\nИсходные данные:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"p = {rho}")

    coeffs_1 = least_squares_1d(x, y, rho, 1)
    a0, a1 = coeffs_1[0], coeffs_1[1]

    coeffs_inv = least_squares_1d(y, x, rho, 1)
    c0, c1 = coeffs_inv[0], coeffs_inv[1]

    sum_rho = np.sum(rho)
    X_center = np.sum(rho * x) / sum_rho
    Y_center = np.sum(rho * y) / sum_rho

    print(f"\nПрямая y = a0 + a1*x:")
    print(f"  y = {a0:.6f} + {a1:.6f}*x")
    print(f"\nОбратная зависимость x = c0 + c1*y:")
    print(f"  x = {c0:.6f} + {c1:.6f}*y")
    print(f"\nЦентр масс (центр тяжести):")
    print(f"  X = {X_center:.6f}")
    print(f"  Y = {Y_center:.6f}")

    y_on_line = a0 + a1 * X_center
    x_on_line = c0 + c1 * Y_center
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

    sizes = 100 + 200 * rho / rho.max() if rho.max() > 0 else 100 * np.ones_like(rho)
    scatter1 = ax1.scatter(x, y, s=sizes, c='red', zorder=5, alpha=0.8,
                           edgecolors='black', linewidth=1.5, label='Исходные точки')

    for i, (xi, yi, ri) in enumerate(zip(x, y, rho)):
        ax1.annotate(f'ρ={ri:.1f}', (xi, yi), textcoords="offset points",
                     xytext=(8, 8), fontsize=9, ha='center',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    x_smooth = np.linspace(x.min() - 0.5, x.max() + 0.5, 500)
    y_line = [a0 + a1 * xv for xv in x_smooth]
    ax1.plot(x_smooth, y_line, 'b-', linewidth=2.5,
             label=f'y = {a0:.4f} + {a1:.4f}·x')

    y_smooth = np.linspace(y.min() - 0.5, y.max() + 0.5, 500)
    x_inv_line = [c0 + c1 * yv for yv in y_smooth]
    ax1.plot(x_inv_line, y_smooth, 'g--', linewidth=2,
             label=f'x = {c0:.4f} + {c1:.4f}·y')

    ax1.scatter([X_center], [Y_center], c='purple', s=250, marker='.',
                zorder=10, label=f'Центр масс ({X_center:.3f}, {Y_center:.3f})', edgecolors='black')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Аппроксимация полиномом 1-й степени\nПрямая y(x), обратная x(y) и центр масс', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

