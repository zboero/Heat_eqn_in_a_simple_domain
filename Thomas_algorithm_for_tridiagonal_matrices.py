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
    c_dash[0] = c[0] / b[0]
    d_dash[0] = d[0] / b[0]
    
    for i in range(1, n - 1):
        c_dash[i] = c[i] / (b[i] - a[i - 1] * c_dash[i - 1])

    for i in range(1, n):
        d_dash[i] = (d[i] - a[i - 1] * d_dash[i - 1]) / (b[i] - a[i - 1] * c_dash[i - 1])

    # Backward substitution
    x[n - 1] = d_dash[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = d_dash[i] - c_dash[i] * x[i + 1]

    return x


# Tridiagonal matrix
def create_tridiagonal_matrix(n):
    """
    Creates a tridiagonal matrix of size n x n with 1 on the main diagonal and -1 on the upper and lower diagonals.

    Parameters:
        n (int): Size of the matrix.

    Returns:
        ndarray: The tridiagonal matrix.
    """
    main_diag = np.ones(n)
    off_diag = -np.ones(n - 1)

    matrix = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    return matrix

# Create a tridiagonal matrix of size 100x100
matrix = create_tridiagonal_matrix(100)
print(matrix)
