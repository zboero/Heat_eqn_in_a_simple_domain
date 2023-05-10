import numpy as np
#import numba as nb

def solve_tridiagonal_matrix(a, b, c, d):
    """
    Solves a tridiagonal matrix system using the Thomas algorithm (TDMA).

    Parameters:
        a (list): Lower diagonal elements of the matrix (length n-1).
        b (list): Main diagonal elements of the matrix (length n).
        c (list): Upper diagonal elements of the matrix (length n-1).
        d (list): Right-hand side vector (length n).

    Returns:
        list: The solution vector (length n).
    """
    n = len(b)
    c_dash = [0] * (n - 1)
    d_dash = [0] * n
    x = [0] * n

    # Forward elimination
    for i in range(1, n - 1):
        c_dash[i] = c[i] / (b[i] - a[i - 1] * c_dash[i - 1])

    for i in range(1, n):
        d_dash[i] = (d[i] - a[i - 1] * d_dash[i - 1]) / (b[i] - a[i - 1] * c_dash[i - 1])

    # Backward substitution
    x[n - 1] = d_dash[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = d_dash[i] - c_dash[i] * x[i + 1]

    return x
