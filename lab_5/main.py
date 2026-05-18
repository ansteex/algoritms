import math


def legendre_poly(n, x):
    if n == 0:
        return 1.0
    if n == 1:
        return x

    p_prev = 1.0
    p_curr = x

    for k in range(1, n):
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
        p_prev, p_curr = p_curr, p_next

    return p_curr


def legendre_poly_derivative(n, x):
    if n == 0:
        return 0.0
    if abs(x) == 1.0:
        return 0.5 * n * (n + 1) if x == 1.0 else 0.5 * n * (n + 1) * ((-1) ** (n + 1))

    p_n = legendre_poly(n, x)
    p_n_minus_1 = legendre_poly(n - 1, x)
    return n * (x * p_n - p_n_minus_1) / (x * x - 1.0)


def gauss_legendre_roots_weights(n):
    roots = []
    weights = []

    for i in range(1, n // 2 + 1):
        x0 = math.cos(math.pi * (4 * i - 1) / (4 * n + 2))

        for _ in range(20):
            p = legendre_poly(n, x0)
            p_prime = legendre_poly_derivative(n, x0)
            if abs(p_prime) < 1e-15:
                break
            dx = p / p_prime
            x0 -= dx
            if abs(dx) < 1e-15:
                break

        roots.append(x0)

    if n % 2 == 1:
        roots.append(0.0)
        roots.sort()
    else:
        roots.sort()

    full_roots = []
    for r in roots:
        full_roots.append(r)
    for r in reversed(roots):
        if r > 0:
            full_roots.insert(0, -r)

    for t in full_roots:
        p_prime = legendre_poly_derivative(n, t)
        weight = 2.0 / ((1 - t * t) * p_prime * p_prime)
        weights.append(weight)

    return full_roots, weights

def f_abs_pow(x, k):
    return abs(x) ** k


def integral_abs_pow_analytical(a, b, k):
    def F(x):
        return (abs(x) ** (k + 1)) / (k + 1) * (1 if x >= 0 else -1)

    return F(b) - F(a)


def trapezoid_3_nodes(f, a, b, k):
    h = (b - a) / 2.0
    x0 = a
    x1 = a + h
    x2 = b
    return (h / 2.0) * (f(x0, k) + 2.0 * f(x1, k) + f(x2, k))


def simpson_3_nodes(f, a, b, k):
    h = (b - a) / 2.0
    x0 = a
    x1 = a + h
    x2 = b
    return (h / 3.0) * (f(x0, k) + 4.0 * f(x1, k) + f(x2, k))


def gauss_3_nodes(f, a, b, k):
    roots, weights = gauss_legendre_roots_weights(3)

    result = 0.0
    for i in range(3):
        x = 0.5 * (b - a) * roots[i] + 0.5 * (b + a)
        result += weights[i] * f(x, k)

    return 0.5 * (b - a) * result


def task1():
    print("1. Сравнение методов численного интегрирования")

    a, b = -1.0, 1.0

    for k in [1, 2]:
        print(f"\nk = {k}")

        exact = integral_abs_pow_analytical(a, b, k)
        print(f"Аналитическое значение: {exact:.10f}")

        trap = trapezoid_3_nodes(f_abs_pow, a, b, k)
        simp = simpson_3_nodes(f_abs_pow, a, b, k)
        gauss = gauss_3_nodes(f_abs_pow, a, b, k)

        print(f"Трапеции (3 узла):     {trap:.10f}  | ошибка = {abs(trap - exact):.2e}")
        print(f"Симпсон (3 узла):      {simp:.10f}  | ошибка = {abs(simp - exact):.2e}")
        print(f"Гаусс (3 узла):        {gauss:.10f}  | ошибка = {abs(gauss - exact):.2e}")


def read_table_from_file(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    lines = content.strip().split('\n')

    first_line = lines[0].strip()
    parts = first_line.split()

    x_vals = []
    for p in parts:
        try:
            val = float(p)
            x_vals.append(val)
        except ValueError:
            continue

    y_vals = []
    table = {}

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            y = float(parts[0])
            y_vals.append(y)
            for i, x in enumerate(x_vals):
                if i + 1 < len(parts):
                    val = float(parts[i + 1])
                    table[(x, y)] = val
        except ValueError:
            continue

    return x_vals, y_vals, table

def bilinear_interpolation(table, x, y, x_vals, y_vals):
    ix = 0
    while ix + 1 < len(x_vals) and x > x_vals[ix + 1]:
        ix += 1
    if ix >= len(x_vals) - 1:
        ix = len(x_vals) - 2

    iy = 0
    while iy + 1 < len(y_vals) and y > y_vals[iy + 1]:
        iy += 1
    if iy >= len(y_vals) - 1:
        iy = len(y_vals) - 2

    x0, x1 = x_vals[ix], x_vals[ix + 1]
    y0, y1 = y_vals[iy], y_vals[iy + 1]

    f00 = table[(x0, y0)]
    f01 = table[(x0, y1)]
    f10 = table[(x1, y0)]
    f11 = table[(x1, y1)]

    if x1 != x0:
        fx0 = f00 + (f10 - f00) * (x - x0) / (x1 - x0)
        fx1 = f01 + (f11 - f01) * (x - x0) / (x1 - x0)
    else:
        fx0, fx1 = f00, f01

    if y1 != y0:
        fxy = fx0 + (fx1 - fx0) * (y - y0) / (y1 - y0)
    else:
        fxy = fx0

    return fxy


def simpson_fixed(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        coef = 4 if i % 2 == 1 else 2
        s += coef * f(a + i * h)
    return s * h / 3


def f_volume(x, y):
    return x * x + y * y


def double_integral_volume_simpson_outer_gauss_inner(a, b, n_simpson, n_gauss):
    t_gauss, A_gauss = gauss_legendre_roots_weights(n_gauss)

    def inner_integral_gauss(x):
        y_low = x * x
        y_high = 1.0
        if y_low >= y_high:
            return 0.0
        inner_sum = 0.0
        for j in range(n_gauss):
            y = 0.5 * (y_high - y_low) * t_gauss[j] + 0.5 * (y_high + y_low)
            inner_sum += A_gauss[j] * f_volume(x, y)
        return 0.5 * (y_high - y_low) * inner_sum

    return simpson_fixed(inner_integral_gauss, a, b, n_simpson)


#def task2():
    # alpha, beta = 1.0, 4.0
    # a, b = 0.0, 2.0
    # phi = lambda x: alpha * x ** 2
    # psi = lambda x: beta * x ** 2
    #
    # x_vals, y_vals, table = read_table_from_file('5.txt')
    #
    # print(f"{' Гаусс внеш/Симпсон внутр':>30} | {'Симпсон внеш/Гаусс внутр':>30} | {'Разница':>15}")
    #
    #
    # for n_nodes in [2, 3, 4, 5]:
    #     res1 = double_integral_gauss_outer_simpson_inner(table, x_vals, y_vals, a, b, phi, psi, n_nodes, n_nodes)
    #     res2 = double_integral_simpson_outer_gauss_inner(table, x_vals, y_vals, a, b, phi, psi, n_nodes, n_nodes                      )
    #     diff = abs(res1 - res2)
    #     print(f" {res1:30.10f} | {res2:30.10f} | {diff:15.2e}")
    # results = []
    #
    # for n_simpson in [4, 8, 16, 32, 64]:
    #     for n_gauss in [2, 3, 4, 5]:
    #         V = double_integral_simpson_outer_gauss_inner(a, b, n_simpson, n_gauss)
    #         results.append((n_simpson, n_gauss, V))
    #         print(f"{n_simpson:10d} | {n_gauss:8d} | {V:20.10f} | {'':20}")


#
# def demo_legendre():
#     print("Узлы и веса формулы Гаусса-Лежандра")
#     for n in [1, 2, 3, 4, 5]:
#         roots, weights = gauss_legendre_roots_weights(n)
#         print(f"\nn = {n}:")
#         print(f"  Узлы: {[round(r, 8) for r in roots]}")
#         print(f"  Веса: {[round(w, 8) for w in weights]}")
#         print(f"  Сумма весов = {sum(weights):.10f}")
def double_integral_volume_gauss_outer_simpson_inner(a, b, n_gauss, n_simpson):
    t_gauss, A_gauss = gauss_legendre_roots_weights(n_gauss)

    def inner_integral_simpson(x):
        y_low = x * x
        y_high = 1.0
        if y_low >= y_high:
            return 0.0

        def f_fixed_y(y):
            return f_volume(x, y)

        return simpson_fixed(f_fixed_y, y_low, y_high, n_simpson)

    result = 0.0
    for i in range(n_gauss):
        x = 0.5 * (b - a) * t_gauss[i] + 0.5 * (b + a)
        result += A_gauss[i] * inner_integral_simpson(x)
    return 0.5 * (b - a) * result
def task_volume():

    a, b = -1.0, 1.0
    exact = 88.0 / 105.0
    print(f"\nТочное значение: V = 88/105 = {exact:.15f}")
    print(f" {'Объём V':>20} | {'Ошибка':>15}")
    print("-" * 80)

    for n_simpson in [8, 16, 32, 64]:
            V = double_integral_volume_simpson_outer_gauss_inner(a, b, n_simpson, n_simpson)
            err = abs(V - exact)
            print(f" {V:20.15f} | {err:15.2e}")

if __name__ == "__main__":
    n = 100

    roots, weights = gauss_legendre_roots_weights(n)

    print(f"{'№':>4} | {'Узел t_i':>20} | {'Вес A_i':>20}")
    print("-" * 80)


    def f(x):
        return abs(x) ** 1
    for i in range(n):
        print(f"{i + 1:4d} | {roots[i]:20.15f} | {weights[i]:20.15f}")
    integral = 0.0
    for i in range(100):
        integral += weights[i] * f(roots[i])

    print(f"\nИнтеграл n=100: {integral:.15f}")
    task1()
    #task2()
    task_volume()
    # demo_legendre()