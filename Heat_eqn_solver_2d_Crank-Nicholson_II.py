import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# Define parameters
Lx = 1.0       # Length of x-domain
Ly = 1.0       # Length of y-domain
nx = 101       # Number of grid points in x-direction
ny = 101       # Number of grid points in y-direction
dx = Lx / (nx - 1)  # Grid spacing in x-direction
dy = Ly / (ny - 1)  # Grid spacing in y-direction
x = np.linspace(0.0, Lx, nx)  # Grid points in x-direction
y = np.linspace(0.0, Ly, ny)  # Grid points in y-direction
dt = 0.0001    # Time step size
T = 0.1        # Final time

# Define the diffusion coefficient
D = 0.1

# Define the initial temperature distribution
T0 = np.zeros((ny, nx))
T0[ny//4:3*ny//4, nx//4:3*nx//4] = 1.0

# Set up the implicit matrix
@njit(parallel=True)
def setup_implicit_matrix(nx, ny, dx, dy, dt, D):
    A = np.zeros((nx*ny, nx*ny))
    for i in prange(nx):
        for j in prange(ny):
            k = i + j*nx
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                # Boundary point
                A[k, k] = 1.0
            else:
                A[k, k-nx] = -0.5 * D * dt / dy**2  # Top neighbor
                A[k, k-1] = -0.5 * D * dt / dx**2   # Left neighbor
                A[k, k] = 1.0 + D * dt / (dx**2 + dy**2)  # Diagonal
                A[k, k+1] = -0.5 * D * dt / dx**2   # Right neighbor
                A[k, k+nx] = -0.5 * D * dt / dy**2  # Bottom neighbor
    return A

A = setup_implicit_matrix(nx, ny, dx, dy, dt, D)

# Time-stepping loop
#@njit(parallel=True)                                 # Let us note that this won't work in Python 3.9 and so it has to be commented.
@njit
def time_stepping(A, Tn, T, dt):
    for n in prange(int(T/dt)):
        Tn = np.linalg.solve(A, Tn)
    return Tn

Tn = T0.flatten()
Tn = time_stepping(A, Tn, T, dt)

# Plot the final temperature distribution
Tn = Tn.reshape((ny, nx))
plt.imshow(Tn, cmap='hot', origin='lower', extent=[0, Lx, 0, Ly])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Temperature at t = {:.3f}'.format(T))
plt.show()

