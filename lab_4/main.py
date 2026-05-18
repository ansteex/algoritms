import numpy as np
import matplotlib.pyplot as plt

def system_equations(vars):
    x, y = vars
    eq1 = 20 * np.log(x - y) - x - y - 6
    eq2 = 20 * np.sin(0.7 * x - 0.7 * y) + 7 * x + 7 * y
    return np.array([eq1, eq2])


def jacobian_matrix(vars):
    x, y = vars
    J11 = 20 / (x - y) - 1
    J12 = -20 / (x - y) - 1
    J21 = 20 * 0.7 * np.cos(0.7 * x - 0.7 * y) + 7
    J22 = -20 * 0.7 * np.cos(0.7 * x - 0.7 * y) + 7
    return np.array([[J11, J12], [J21, J22]])


def newton_system(F, J, x0, tol=1e-6, max_iter=100, verbose=True):
    x = np.array(x0, dtype=float)
    errors = []

    for i in range(max_iter):
        Fx = F(x)
        Jx = J(x)

        try:
            dx = np.linalg.solve(Jx, Fx)
        except np.linalg.LinAlgError:
            if verbose:
                print("Матрица Якоби вырождена!")
            break

        x = x - dx
        error = np.linalg.norm(Fx)
        errors.append(error)

        if verbose:
            print(f"Итерация {i + 1}: x={x[0]:.8f}, y={x[1]:.8f}, невязка={error:.2e}")

        if error < tol:
            break

    return x, errors


def find_good_initial_guess():
    candidates = [[2, 1], [3, 2], [4, 3], [10, 9], [1.5, 0.5], [6, 2]]

    for x0 in candidates:
        try:
            solution, errors = newton_system(system_equations, jacobian_matrix, x0,
                                             tol=1e-6, max_iter=20, verbose=True)
            if not np.any(np.isnan(solution)):
                return solution
        except:
            print("Расходится")
    return None


def task1():
    print("1. Решение системы нелинейных уравнений")
    print("Система:")
    print("  20·ln(x-y) - x - y = 6")
    print("  20·sin(0.7x-0.7y) + 7x + 7y = 0")
    solution = find_good_initial_guess()
    if solution is not None:
        print(f"\nНайденное решение: x = {solution[0]:.10f}, y = {solution[1]:.10f}")


def integrate_trapezoidal(f, a, b, n=1000):
    h = (b - a) / n
    result = (f(a) + f(b)) / 2
    for i in range(1, n):
        result += f(a + i * h)
    return result * h

def laplas_integral(x, n=1000):
    if x < 0:
        return -laplas_integral(-x, n)
    def integrand(t):
        return np.exp(-t ** 2 / 2)
    integral = integrate_trapezoidal(integrand, 0, x, n)
    return (2 / np.sqrt(2 * np.pi)) * integral


def inverse_laplace_bisection(phi_target, a=-5, b=5, tol=1e-6, max_iter=100, n=1000):
    if phi_target < -1 or phi_target > 1:
        raise ValueError("Φ(x) должна быть в диапазоне [-1, 1]")

    for i in range(max_iter):
        x_mid = (a + b) / 2
        phi_mid = laplas_integral(x_mid, n)
        if abs(phi_mid - phi_target) < tol:
            print(f"Сошлось за {i + 1} итераций")
            return x_mid

        if phi_mid < phi_target:
            a = x_mid
        else:
            b = x_mid

    return (a + b) / 2


def task2():
    print("2. Обратная функция Лапласа")
    phi_values = [0.1]
    print("  Φ(x)    |       x (решение)")
    print("-" * 45)

    for phi in phi_values:
        x_numeric = inverse_laplace_bisection(phi, n=1000)

        print(f"  {phi:.3f}     |    {x_numeric:.8f}")

def solve_boundary_problem(N=50, tol=1e-6, max_iter=100):
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    y = np.zeros(N + 1)
    for i in range(N + 1):
        y[i] = 1 + 2 * x[i]  


    for iteration in range(max_iter):
        A = np.zeros(N + 1)
        B = np.zeros(N + 1)
        C = np.zeros(N + 1)
        F = np.zeros(N + 1)

        for n in range(1, N):
            R = (y[n - 1] - 2 * y[n] + y[n + 1]) / h ** 2 - y[n] ** 3 - x[n] ** 2

            A[n] = 1 / h ** 2

            B[n] = -2 / h ** 2 - 3 * y[n] ** 2

            C[n] = 1 / h ** 2
            F[n] = -R
        B[0] = 1
        C[0] = 0
        F[0] = 0

        A[N] = 0
        B[N] = 1
        F[N] = 0

        dy = tridiagonal_solve(A, B, C, F, N + 1)

        y = y + dy

        max_diff = np.max(np.abs(dy))
        if max_diff < tol:
            print(f"Сошлось за {iteration + 1} итераций")
            break

    return x, y


def tridiagonal_solve(A, B, C, F, n):
    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -C[0] / B[0]
    beta[0] = F[0] / B[0]

    for i in range(1, n):
        denom = B[i] + A[i] * alpha[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        alpha[i] = -C[i] / denom
        beta[i] = (F[i] - A[i] * beta[i - 1]) / denom

    u = np.zeros(n)
    u[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        u[i] = alpha[i] * u[i + 1] + beta[i]

    return u


def task3():
    print("3. Решение краевой задачи")
    print("Уравнение: y'' - y^3 = x^2, 0 <= x <= 1")
    print("Краевые условия: y(0) = 1, y(1) = 3")
    for N in [20, 50, 100]:
        print(f"\nСетка: N = {N} (h = {1 / N:.4f})")
        x, y = solve_boundary_problem(N=N, tol=1e-6)

        if N == 50:
            x_final, y_final = x, y

    plt.figure(figsize=(10, 6))
    plt.plot(x_final, y_final, 'b-', linewidth=2)
    plt.scatter([0, 1], [1, 3], c='red', s=100, zorder=5, label='Краевые условия')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Решение краевой задачи y\'\' - y^3 = x^2')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nЗначения решения в нескольких точках:")
    print("   x     |     y(x)")
    print("-" * 25)
    for i in range(0, len(x_final), len(x_final) // 10):
        print(f"  {x_final[i]:.2f}   |   {y_final[i]:.6f}")


def main():
    task1()
    task2()
    task3()


if __name__ == "__main__":
    main()