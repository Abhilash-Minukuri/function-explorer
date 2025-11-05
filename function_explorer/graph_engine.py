"""Plot and math helpers (stub)."""

def sample_quadratic(a: float, b: float, c: float, x_min=-10, x_max=10, n=400):
    import numpy as np
    xs = np.linspace(x_min, x_max, n)
    ys = a*xs**2 + b*xs + c
    return xs, ys
