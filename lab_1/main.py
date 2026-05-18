import numpy as np
import matplotlib.pyplot as plt


def newton_interp_1d(x_points, y_points, x0, degree):
    n = len(x_points)
    if degree >= n:
        degree = n - 1
    if degree < 0:
        degree = 0
    m = degree + 1

    idx = np.searchsorted(x_points, x0)
    if idx == 0:
        idx = 1
    elif idx == n:
        idx = n - 1

    half = m // 2
    start = idx - half - 1
    if start < 0:
        start = 0
    if start + m > n:
        start = n - m

    x = x_points[start:start + m]
    y = y_points[start:start + m]

    diff = np.zeros((m, m))
    for i in range(m):
        diff[i, 0] = y[i]

    for j in range(1, m):
        for i in range(m - j):
            diff[i, j] = (diff[i+1, j-1] - diff[i, j-1]) / (x[i+j] - x[i])

    result = diff[0, 0]
    product = 1.0
    for j in range(1, m):
        product *= (x0 - x[j-1])
        result += diff[0, j] * product

    return result


def interp2d_sequential(T_vals, P_vals, Z, T0, P0, deg_T, deg_P):
    z_at_P = np.zeros(len(P_vals))
    for j in range(len(P_vals)):
        column = []
        for i in range(len(Z)):
            column.append(Z[i][j])
        z_at_P[j] = newton_interp_1d(T_vals, column, T0, deg_T)
    return newton_interp_1d(P_vals, z_at_P, P0, deg_P)


def _is_numeric_row(line, min_cols, exact_cols=False):
    parts = line.split()
    if exact_cols and len(parts) != min_cols:
        return False
    if not exact_cols and len(parts) < min_cols:
        return False
    try:
        float(parts[0])
        return True
    except ValueError:
        return False


def _is_pressure_row(line):
    parts = line.split()
    if len(parts) != 3:
        return False
    try:
        vals = [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError:
        return False
    return np.allclose(vals, [0.5, 1.5, 2.5], atol=1e-12)


def read_pressure_header(lines, cur_idx, table_name):
    while cur_idx < len(lines) and not _is_pressure_row(lines[cur_idx]):
        cur_idx += 1
    if cur_idx >= len(lines):
        raise ValueError(f"Не найдена строка давлений для таблицы {table_name}")
    pvals = np.array([float(v) for v in lines[cur_idx].split()], dtype=float)
    return pvals, cur_idx + 1


def read_fixed_rows(lines, cur_idx, n_rows, table_name):
    T_vals = []
    Z_vals = []
    while cur_idx < len(lines) and len(T_vals) < n_rows:
        if _is_numeric_row(lines[cur_idx], 4):
            parts = lines[cur_idx].split()
            T_vals.append(float(parts[0]))
            Z_vals.append([float(parts[1]), float(parts[2]), float(parts[3])])
        cur_idx += 1
    if len(T_vals) != n_rows:
        raise ValueError(f"Таблица {table_name} прочитана не полностью: {len(T_vals)} из {n_rows} строк")
    return np.array(T_vals), np.array(Z_vals), cur_idx


def read_input_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip() != '']

    data = {}
    idx = 0

    while idx < len(lines) and not _is_numeric_row(lines[idx], 3, exact_cols=True):
        idx += 1

    t_I = []
    I_vals = []
    while idx < len(lines) and _is_numeric_row(lines[idx], 3, exact_cols=True):
        parts = lines[idx].split()
        t_I.append(float(parts[1]))
        I_vals.append(float(parts[2]))
        idx += 1
    data['t_I'] = np.array(t_I)
    data['I'] = np.array(I_vals)

    data['T_Nh'], data['Nh'], idx = read_fixed_rows(lines, idx, 19, "Nh")

    P_vals, idx = read_pressure_header(lines, idx, "sigma")
    data['P_vals'] = P_vals
    data['T_sigma'], data['sigma'], idx = read_fixed_rows(lines, idx, 11, "sigma")

    _, idx = read_pressure_header(lines, idx, "C")
    data['T_C'], data['C'], idx = read_fixed_rows(lines, idx, 19, "C")

    _, idx = read_pressure_header(lines, idx, "q")
    data['T_q'], data['q'], idx = read_fixed_rows(lines, idx, 8, "q")

    return data


def solve_pressure(T, K_const, interp_nh, P_min=0.3, P_max=2.5, tol=1e-6, max_iter=100):
    f_low = interp_nh(T, P_min) - K_const
    f_high = interp_nh(T, P_max) - K_const

    for _ in range(max_iter):
        P_mid = (P_min + P_max) / 2.0
        f_mid = interp_nh(T, P_mid) - K_const
        if abs(f_mid) < tol:
            return P_mid
        if f_low * f_mid < 0:
            P_max = P_mid
            f_high = f_mid
        else:
            P_min = P_mid
            f_low = f_mid
    return P_mid


def solve_ode(t0, tk, tau, T0, px, Tx, R, l,
              I_interp, interp_nh, interp_sigma, interp_q, interp_C,
              deg_T=2, deg_P=1):
    K_const = 7.242e4 * px / Tx

    N = int(np.ceil((tk - t0) / tau))
    t = np.linspace(t0, t0 + N*tau, N+1)
    T = np.zeros(N+1)
    p = np.zeros(N+1)
    sigma = np.zeros(N+1)
    q_out = np.zeros(N+1)
    Rd = np.zeros(N+1)
    Fr = np.zeros(N+1)

    T[0] = T0
    p[0] = solve_pressure(T0, K_const, interp_nh)

    for n in range(N):
        tn = t[n]

        j_n = I_interp(tn) / (np.pi * R**2)
        sigma_n = interp_sigma(T[n], p[n])
        q_n = interp_q(T[n], p[n])
        C_n = interp_C(T[n], p[n])
        phi_n = (j_n**2 / sigma_n - q_n) / C_n

        T_half = T[n] + tau * phi_n

        p_half = solve_pressure(T_half, K_const, interp_nh)
        j_half = I_interp(tn + tau) / (np.pi * R**2)
        sigma_half = interp_sigma(T_half, p_half)
        q_half = interp_q(T_half, p_half)
        C_half = interp_C(T_half, p_half)
        phi_half = (j_half**2 / sigma_half - q_half) / C_half

        T[n+1] = T[n] + tau * phi_half
        p[n+1] = solve_pressure(T[n+1], K_const, interp_nh)

        sigma[n+1] = interp_sigma(T[n+1], p[n+1])
        q_out[n+1] = interp_q(T[n+1], p[n+1])
        Rd[n+1] = l / (np.pi * sigma[n+1] * R**2)
        Fr[n+1] = q_out[n+1] * R / 2.0

    sigma[0] = interp_sigma(T[0], p[0])
    q_out[0] = interp_q(T[0], p[0])
    Rd[0] = l / (np.pi * sigma[0] * R**2)
    Fr[0] = q_out[0] * R / 2.0

    return t, T, p, sigma, q_out, Rd, Fr


def make_interp2d_function(T_vals, P_vals, Z, deg_T, deg_P):
    def interp_func(T, P):
        T_clip = np.clip(T, T_vals[0], T_vals[-1])
        P_clip = np.clip(P, P_vals[0], P_vals[-1])
        return interp2d_sequential(T_vals, P_vals, Z, T_clip, P_clip, deg_T, deg_P)
    return interp_func


def make_interp1d_function(x, y, degree):
    def interp_func(x0):
        return newton_interp_1d(x, y, x0, degree)
    return interp_func


def is_monotonic(arr):
    increasing = all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
    decreasing = all(arr[i] >= arr[i+1] for i in range(len(arr)-1))
    return increasing or decreasing


def inverse_interpolation_local(x_points, y_points, y0, degree=3):
    n = len(x_points)
    closest_idx = np.argmin(np.abs(np.array(y_points) - y0))
    left = closest_idx
    right = closest_idx

    while left > 0:
        y_slice = y_points[left-1:right+1]
        if not is_monotonic(y_slice):
            break
        left -= 1

    while right < n - 1:
        y_slice = y_points[left:right+2]
        if not is_monotonic(y_slice):
            break
        right += 1

    x_local = x_points[left:right+1]
    y_local = y_points[left:right+1]
    return newton_interp_1d(y_local, x_local, y0, degree)


def main():
    R = 0.35
    l = 12.0
    t0 = 14e-6
    tk = 450e-6
    T0 = 5400.0
    px = 0.04
    Tx = 300.0
    tau = 1e-6

    deg_T = 2
    deg_P = 3
    deg_t = 5

    filename = '16-02-2026.txt'
    data = read_input_data(filename)

    I_interp = make_interp1d_function(data['t_I'], data['I'], deg_t)

    print("\n" + "="*60)
    print("ОБРАТНАЯ ИНТЕРПОЛЯЦИЯ ДЛЯ ТОКА")
    print("="*60)

    t_data = data['t_I']
    I_data = data['I']

    I_target = 500.0
    t_found = inverse_interpolation_local(t_data, I_data, I_target, degree=3)
    t_found_mks = t_found * 1e6

    max_idx = np.argmax(I_data)
    t_max = t_data[max_idx]

    print(f"Ток {I_target} А достигается при t = {t_found_mks:.2f} мкс")
    if t_found < t_max:
        print(f"(растущая ветка, максимум тока {max(I_data):.1f} А при {t_max*1e6:.1f} мкс)")
    else:
        print(f"(падающая ветка, максимум тока {max(I_data):.1f} А при {t_max*1e6:.1f} мкс)")

    P_vals = data['P_vals']

    nh_func = make_interp2d_function(data['T_Nh'], P_vals, data['Nh'], deg_T, deg_P)
    sigma_func = make_interp2d_function(data['T_sigma'], P_vals, data['sigma'], deg_T, deg_P)
    q_func = make_interp2d_function(data['T_q'], P_vals, data['q'], deg_T, deg_P)
    C_func = make_interp2d_function(data['T_C'], P_vals, data['C'], deg_T, deg_P)

    t, T, p, sigma, q_out, Rd, Fr = solve_ode(
        t0, tk, tau, T0, px, Tx, R, l,
        I_interp, nh_func, sigma_func, q_func, C_func,
        deg_T, deg_P
    )

    with open('results.txt', 'w') as f:
        f.write('t(s)\tT(K)\tp(MPa)\tsigma(1/(Ohm*cm))\tq(W/cm3)\tRd(Ohm)\tFr(W/cm2)\n')
        for i in range(len(t)):
            f.write(f'{t[i]:.3e}\t{T[i]:.3e}\t{p[i]:.3e}\t{sigma[i]:.3e}\t{q_out[i]:.3e}\t{Rd[i]:.3e}\t{Fr[i]:.3e}\n')

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(t*1e6, T, 'b-')
    plt.xlabel('t, мкс')
    plt.ylabel('T, K')
    plt.title('Температура')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(t*1e6, p, 'r-')
    plt.xlabel('t, мкс')
    plt.ylabel('p, МПа')
    plt.title('Давление')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(t*1e6, sigma, 'g-')
    plt.xlabel('t, мкс')
    plt.ylabel('σ, 1/(Ом·см)')
    plt.title('Электропроводность')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(t*1e6, q_out, 'm-')
    plt.xlabel('t, мкс')
    plt.ylabel('q, Вт/см³')
    plt.title('Мощность излучения')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(t*1e6, Rd, 'c-')
    plt.xlabel('t, мкс')
    plt.ylabel('R_d, Ом')
    plt.title('Сопротивление разряда')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(t*1e6, Fr, 'k-')
    plt.xlabel('t, мкс')
    plt.ylabel('F_r, Вт/см²')
    plt.title('Поток излучения на поверхности')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()