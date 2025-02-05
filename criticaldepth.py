import math
def manning_function(yn, Q, n, S0, b, m):
    """Computes Manning's function f_n(yn) for normal depth."""
    A = (b + m * yn) * yn  # Cross-sectional area
    Pw = b + 2 * yn * math.sqrt(1 + m**2)  # Wetted perimeter
    return (math.sqrt(S0) / n) * (A ** (5/3)) / (Pw ** (2/3)) - Q

def manning_derivative(yn, n, S0, b, m):
    """Computes the derivative f'_n(yn) for Newton-Raphson iteration."""
    A = (b + m * yn) * yn
    Pw = b + 2 * yn * math.sqrt(1 + m**2)
    dA_dy = b + 2 * m * yn
    dPw_dy = 2 * math.sqrt(1 + m**2)
    return (math.sqrt(S0) / n) * ((5/3) * (A ** (2/3)) * (Pw ** (-2/3)) * dA_dy - (2/3) * (A ** (5/3)) * (Pw ** (-5/3)) * dPw_dy)

def critical_function(yc, Q, g, b, m):
    """Compute function f_c(yc) for critical depth."""
    A = (b + m * yc) * yc
    T = b + 2 * m * yc
    return (A ** (3/2)) / math.sqrt(T) - Q / math.sqrt(g)

def critical_derivative(yc, g, b, m):
    """Compute derivative f'_c(yc) for Newton-Raphson iteration."""
    A = (b + m * yc) * yc
    T = b + 2 * m * yc
    dA_dy = b + 2 * m * yc
    dT_dy = 2 * m
    return (3/2) * (A ** (1/2)) / math.sqrt(T) * dA_dy - (1/2) * (A ** (3/2)) / (T ** (3/2)) * dT_dy

def newton_raphson(func, dfunc, initial_guess, tol=1e-6, max_iter=100):
    """General Newton-Raphson solver."""
    y = initial_guess
    for _ in range(max_iter):
        f_val = func(y)
        df_val = dfunc(y)
        if abs(f_val) < tol:
            return y
        y -= f_val / df_val
    raise ValueError("Newton-Raphson did not converge")

def compute_depths(Q, S0, n, b, m, g=9.81):
    """Compute normal and critical depths using Newton-Raphson method."""
    yn = newton_raphson(lambda y: manning_function(y, Q, n, S0, b, m),
                        lambda y: manning_derivative(y, n, S0, b, m),
                        initial_guess=1.0)
    yc = newton_raphson(lambda y: critical_function(y, Q, g, b, m),
                        lambda y: critical_derivative(y, g, b, m),
                        initial_guess=1.0)
    return yn, yc

# Given problem parameters
Q = 26.5  # Flow rate in m^3/s
S0 = 0.001  # Bed slope
n = 0.025  # Manning’s coefficient
b_trap = 5  # Bottom width for trapezoidal
m_trap = 1  # Side slope for trapezoidal
b_rect = 5  # Bottom width for rectangular (m = 0)
m_rect = 0
b_tri = 0  # Bottom width for triangular (b = 0)
m_tri = 1

y_normal_trap, y_critical_trap = compute_depths(Q, S0, n, b_trap, m_trap)
y_normal_rect, y_critical_rect = compute_depths(Q, S0, n, b_rect, m_rect)
y_normal_tri, y_critical_tri = compute_depths(Q, S0, n, b_tri, m_tri)

print(f"Trapezoidal Channel - Normal Depth: {y_normal_trap:.4f} m, Critical Depth: {y_critical_trap:.4f} m")
print(f"Rectangular Channel - Normal Depth: {y_normal_rect:.4f} m, Critical Depth: {y_critical_rect:.4f} m")
print(f"Triangular Channel - Normal Depth: {y_normal_tri:.4f} m, Critical Depth: {y_critical_tri:.4f} m")
