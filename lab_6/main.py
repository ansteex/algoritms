import math

x = [1, 2, 3, 4, 5, 6]
y = [0.571, 0.889, 1.091, 1.231, 1.333, 1.412]

n = len(x)
h = 1.0
print("1. Численное дифференцирование сеточной функции")

one_sided = [float('nan')] * n
for i in range(n-1):
    one_sided[i] = (y[i+1] - y[i]) / h

central = [float('nan')] * n
for i in range(1, n-1):
    central[i] = (y[i+1] - y[i-1]) / (2*h)

runge = [float('nan')] * n
for i in range(0, n-2):
    y_h = (y[i+1] - y[i]) / h
    y_2h = (y[i+2] - y[i]) / (2*h)
    runge[i] = 2*y_2h - y_h
eta = [1.0/y_i for y_i in y]
xi = [1.0/x_i for x_i in x]

n_points = len(xi)
sum_xi = sum(xi)
sum_eta = sum(eta)
sum_xi_xi = sum(xi_i * xi_i for xi_i in xi)
sum_xi_eta = sum(xi[i] * eta[i] for i in range(n_points))

det = n_points * sum_xi_xi - sum_xi * sum_xi
if abs(det) > 1e-12:
    A = (n_points * sum_xi_eta - sum_xi * sum_eta) / det
    B = (sum_xi_xi * sum_eta - sum_xi * sum_xi_eta) / det
else:
    A, B = 0, 0

y_prim_align = [0.0] * n
for i in range(n):
    y_prim_align[i] = A * (y[i]*y[i]) / (x[i]*x[i])

second_deriv = [float('nan')] * n
for i in range(1, n-1):
    second_deriv[i] = (y[i+1] - 2*y[i] + y[i-1])/h**2

print(f"{'x':>4} | {'y':>10}| {'Одностор.':>10} | {'Центр.':>10} | {'Рунге':>10} | {'Выравн.':>10} | {'y\'\'':>10}")
print("-" * 80)

def fmt(val):
    return "   ----   " if math.isnan(val) else f"{val:10.6f}"

for i in range(n):
    print(f"{x[i]:4d} | {y[i]:10.6f} | {fmt(one_sided[i])} | {fmt(central[i])} | {fmt(runge[i])} | {y_prim_align[i]:10.6f} | {fmt(second_deriv[i])}")

runge_second = [float('nan')] * n
for i in range(1, n - 1):
    y_h = (y[i + 1] - 2 * y[i] + y[i - 1]) / (h * h)


    y_mid_right = (y[i + 1] + y[i]) / 2
    y_mid_left = (y[i] + y[i - 1]) / 2
    y_half = (y_mid_right - 2 * y[i] + y_mid_left) / ((h / 2) * (h / 2))

    runge_second[i] = (4 * y_half - y_h) / 3
print()
print(f"{'x':>4} | {'y':>10} | {'y\'\' (обычная)':>15} | {'y\'\' (Рунге)':>15}")
print("-" * 55)




second_deriv = [float('nan')] * n
for i in range(1, n - 1):
    second_deriv[i] = (y[i + 1] - 2 * y[i] + y[i - 1]) / (h * h)

for i in range(n):
    print(f"{x[i]:4d} | {y[i]:10.6f} | {fmt(second_deriv[i])} | {fmt(runge_second[i])}")


alpha = 0.0
beta = 1.0
gamma = -1.0

def solve_ode_progonka(N, alpha, beta, gamma):
    h = 1.0 / N
    x = [i * h for i in range(N + 1)]

    A = [0.0] * (N + 1)
    B = [0.0] * (N + 1)
    C = [0.0] * (N + 1)
    F = [0.0] * (N + 1)

    for i in range(1, N):
        A[i] = 1 + h * x[i] ** 2
        B[i] = -2 + 4 * h * h
        C[i] = 1 - h * x[i] ** 2
        F[i] = h * h * (2 * x[i] + math.exp(-x[i]))
    N_nodes = N + 1

    ksi = [0.0] * N_nodes
    eta = [0.0] * N_nodes


    A1 = 1 + h * x[1] ** 2
    B1 = -2 + 4 * h * h
    C1 = 1 - h * x[1] ** 2
    F1 = h * h * (2 * x[1] + math.exp(-x[1]))

    B_modified = 4 * A1 / 3 + B1
    C_modified = -A1 / 3 + C1
    F_modified = F1 + 2 * h * alpha * A1 / 3

    if abs(C_modified) < 1e-15:
        ksi[1] = 0
        eta[1] = F_modified / B_modified
    else:
        ksi[1] = -C_modified / B_modified
        eta[1] = F_modified / B_modified

    for i in range(2, N):
        A_i = 1 + h * x[i] ** 2
        B_i = -2 + 4 * h * h
        C_i = 1 - h * x[i] ** 2
        F_i = h * h * (2 * x[i] + math.exp(-x[i]))

        denom = B_i + A_i * ksi[i - 1]
        if abs(denom) < 1e-15:
            ksi[i] = 0
            eta[i] = 0
        else:
            ksi[i] = -C_i / denom
            eta[i] = (F_i - A_i * eta[i - 1]) / denom

    a_coef = ksi[N - 2] * ksi[N - 1] - 4 * ksi[N - 1] + (3 - 2 * h * beta)
    b_coef = ksi[N - 2] * eta[N - 1] + eta[N - 2] - 4 * eta[N - 1] - 2 * h * gamma

    if abs(a_coef) < 1e-15:
        u_N = 0
    else:
        u_N = -b_coef / a_coef

    u = [0.0] * N_nodes
    u[N] = u_N

    for i in range(N - 1, -1, -1):
        u[i] = ksi[i] * u[i + 1] + eta[i]
    u[0] = (4 * u[1] - u[2]) / 3
    return x, u


def newton_method(f, df, x0, tol=1e-10, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, i + 1
        x = x_new
    return x, max_iter

def solve_ode(N, alpha, beta, gamma):
    h = 1.0 / N
    x = [i * h for i in range(N + 1)]

    def solve_ivp(u0, u0_prime):
        u = [0.0] * (N + 1)
        v = [0.0] * (N + 1)
        u[0] = u0
        v[0] = u0_prime

        for i in range(N):
            xi = x[i]
            k1_u = v[i]
            k1_v = 2 * xi * xi * v[i] - 4 * u[i] + 2 * xi + math.exp(-xi)

            x_mid = xi + h / 2
            k2_u = v[i] + h / 2 * k1_v
            k2_v = 2 * x_mid * x_mid * (v[i] + h / 2 * k1_v) - 4 * (u[i] + h / 2 * k1_u) + 2 * x_mid + math.exp(-x_mid)

            k3_u = v[i] + h / 2 * k2_v
            k3_v = 2 * x_mid * x_mid * (v[i] + h / 2 * k2_v) - 4 * (u[i] + h / 2 * k2_u) + 2 * x_mid + math.exp(-x_mid)

            x_next = xi + h
            k4_u = v[i] + h * k3_v
            k4_v = 2 * x_next * x_next * (v[i] + h * k3_v) - 4 * (u[i] + h * k3_u) + 2 * x_next + math.exp(-x_next)

            u[i + 1] = u[i] + h / 6 * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
            v[i + 1] = v[i] + h / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        return u, v

    def residual(u0):
        u, v = solve_ivp(u0, alpha)
        return v[N] - beta * u[N] - gamma

    def residual_deriv(u0):
        eps = 1e-7
        return (residual(u0 + eps) - residual(u0 - eps)) / (2 * eps)
    u0_initial = 0.0

    u0_solution, iterations = newton_method(residual, residual_deriv, u0_initial)
    u, v = solve_ivp(u0_solution, alpha)
    return x, u, v, u0_solution, iterations


print("2. Сравнение методов прогонки и Ньютона")

N = 20

x_prog, u_prog = solve_ode_progonka(N, alpha, beta, gamma)

x_newton, u_newton, v_newton, prime0, iterations = solve_ode(N, alpha, beta, gamma)

print("\nСравнение значений u(x):")
print(f"{'x':>8} | {'Метод прогонки':>16} | {'Метод Ньютона':>16} | {'Разница':>12}")
for i in range(0, N + 1, N // 10):
    diff = abs(u_prog[i] - u_newton[i])
    print(f"{x_prog[i]:8.4f} | {u_prog[i]:16.8f} | {u_newton[i]:16.8f} | {diff:12.2e}")

