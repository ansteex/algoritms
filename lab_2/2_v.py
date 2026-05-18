import numpy as np
from read import read_3d_data
from math import *

def newton_interp_1d(x, y, x0, degree):
    n = len(x)
    if degree >= n:
        degree = n - 1
    m = degree + 1
    idx = np.searchsorted(x, x0)
    start = max(0, min(idx - m//2, n - m))
    
    x_sel = x[start:start + m]
    y_sel = y[start:start + m]
    
    coef = y_sel.copy()
    for j in range(1, m):
        for i in range(m-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x_sel[i] - x_sel[i-j])
    
    result = coef[-1]
    for i in range(m-2, -1, -1):
        result = result * (x0 - x_sel[i]) + coef[i]
    
    return result

def cubic_spline_coeffs(x, y, ypp0=0, yppn=0):
    n = len(x)
    h = np.diff(x)
    
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    F = np.zeros(n)
    
    for i in range(1, n-1):
        A[i] = h[i-1]
        B[i] = 2 * (h[i-1] + h[i])
        C[i] = h[i]
        F[i] = 3 * ((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    
    A[0] = 0
    B[0] = 1
    C[0] = 0
    F[0] = ypp0 / 2
    
    A[n-1] = 0
    B[n-1] = 1
    C[n-1] = 0
    F[n-1] = yppn / 2
    
    alpha = np.zeros(n)
    beta = np.zeros(n)
    
    alpha[0] = -C[0] / B[0]
    beta[0] = F[0] / B[0]
    
    for i in range(1, n):
        denom = B[i] + A[i] * alpha[i-1]
        alpha[i] = -C[i] / denom
        beta[i] = (F[i] - A[i] * beta[i-1]) / denom
    
    c = np.zeros(n)
    c[n-1] = beta[n-1]
    for i in range(n-2, -1, -1):
        c[i] = alpha[i] * c[i+1] + beta[i]
    
    a = y[:-1]
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    
    for i in range(n-1):
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2*c[i]) / 3
    
    return a, b, c[:-1], d, x

def spline_eval(coeffs, x0):
    a, b, c, d, x = coeffs
    
    if x0 <= x[0]:
        i = 0
    elif x0 >= x[-1]:
        i = len(a) - 1
    else:
        i = 0
        while i < len(a) and x0 > x[i+1]:
            i += 1
    
    dx = x0 - x[i]
    return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

def spline_interp_1d(x, y, x0,  yp0, ypn):
    coeffs = cubic_spline_coeffs(x, y, yp0, ypn)
    return spline_eval(coeffs, x0)

def interp3d(x_vals, y_vals, z_vals, data, 
             x0, y0, z0,
             method_x='newton', method_y='newton', method_z='newton',
             deg_x=2, deg_y=2, deg_z=2,yp0=0, ypn=0):
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    
    values_yz = np.zeros((nz, ny))
    
    for iz in range(nz):
        for iy in range(ny):
            u_vals = data[iz, iy, :]
            
            if method_x == 'newton':
                u_at_x = newton_interp_1d(x_vals, u_vals, x0, deg_x)
            else:
                u_at_x = spline_interp_1d(x_vals, u_vals, x0, yp0, ypn)
            
            values_yz[iz, iy] = u_at_x
    
    values_z = np.zeros(nz)
    
    for iz in range(nz):
        u_vals = values_yz[iz, :]
        
        if method_y == 'newton':
            u_at_y = newton_interp_1d(y_vals, u_vals, y0, deg_y)
        else:
            u_at_y = spline_interp_1d(y_vals, u_vals, y0, yp0, ypn)
        
        values_z[iz] = u_at_y
    
    if method_z == 'newton':
        return newton_interp_1d(z_vals, values_z, z0, deg_z)
    else:
        return spline_interp_1d(z_vals, values_z, z0, yp0, ypn)

def parse_function(expr):
    def func(x):
        return eval(expr)
    return func

def generate_grid(x_nodes, func):
    y = [func(xi) for xi in x_nodes]
    return y

def get_format_string(value):
    if abs(value) < 1e-10:
        return ".12f"
    elif abs(value) < 1e-8:
        return ".10f"
    elif abs(value) < 1e-6:
        return ".8f"
    elif abs(value) < 1e-4:
        return ".6f"
    else:
        return ".4f"

def second_derivative(f, x, h=1e-5):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h * h)

def main():
    # def f(x):
    #     return x**2
    #x, y

    # newton (3)
    # spline
    # exact
    
    func_str = input("\nВведите тестовую функцию: ")
    f = parse_function(func_str)
    
    x_nodes_str = input("x узлы: ")
    x_nodes = np.array([float(v) for v in x_nodes_str.split()])
    y_nodes = generate_grid(x_nodes, f)

    degree = int(input("Степень полинома Ньютона: "))
    print("\nВведите точку x0:")
    x0 = float(input("x0 = "))
    exact_val = f(x0)
    ypp0 = second_derivative(f, x_nodes[0])
    yppn = second_derivative(f, x_nodes[-1])
    newton_val = newton_interp_1d(x_nodes, y_nodes, x0, degree)
    spline_val = spline_interp_1d(x_nodes, y_nodes, x0, ypp0, yppn)
    
    error_newton = abs(newton_val - exact_val)
    error_spline = abs(spline_val - exact_val)
    
    fmt_newton = get_format_string(error_newton)
    fmt_spline = get_format_string(error_spline)
    
    print("\n" + "="*70)
    print(f"{'Метод':<15} {'Значение':>18} {'Погрешность':>18}")
    print("="*70)
    print(f"{'Ньютон':<15} {newton_val:>{18}{fmt_newton}} {error_newton:>18.2e}")
    print(f"{'Сплайн':<15} {spline_val:>{18}{fmt_spline}} {error_spline:>18.2e}")
    print(f"{'Точное':<15} {exact_val:>{18}{fmt_spline}} {'---':>18}")

if __name__ == "__main__":
    main()