import numpy as np
from numba import jit

@jit(nopython=True)
def solve_heat_equation(nx, ny, nt, Lx, Ly, kappa):
    dx = Lx / (nx-1)
    dy = Ly / (ny-1)
    dt = dx**2 * dy**2 / (2*kappa*(dx**2 + dy**2))
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    u = np.zeros((nx, ny, nt))
    u[:,:,0] = 100  # initial condition

    # Define coefficients
    ax_x = kappa*dt/(dx**2)
    ay_y = kappa*dt/(dy**2)
    bx_x = 1 - 2*ax_x
    by_y = 1 - 2*ay_y

    # Define the tridiagonal matrices for x and y
    Ax = np.zeros((nx-2, nx-2))
    Ay = np.zeros((ny-2, ny-2))
    np.fill_diagonal(Ax[1:], ax_x)
    np.fill_diagonal(Ax[:,1:], ax_x)
    np.fill_diagonal(Ax, bx_x)
    np.fill_diagonal(Ay[1:], ay_y)
    np.fill_diagonal(Ay[:,1:], ay_y)
    np.fill_diagonal(Ay, by_y)

    # Define the inverse of the matrices Ax and Ay
    Ix = np.identity(nx-2)
    Iy = np.identity(ny-2)
    invAx = np.linalg.inv(Ix - 0.5*Ax)
    invAy = np.linalg.inv(Iy - 0.5*Ay)

    # Time-stepping
    for k in range(nt-1):
        # Implicit in x
        for j in range(1, ny-1):
            u[1:nx-1,j,k+1] = invAx.dot(u[1:nx-1,j,k] + 0.5*ax_x*(u[2:nx,j,k] - 2*u[1:nx-1,j,k] + u[0:nx-2,j,k]))
        # Implicit in y
        for i in range(1, nx-1):
            u[i,1:ny-1,k+1] = invAy.dot(u[i,1:ny-1,k+1] + 0.5*ay_y*(u[i,2:ny,k] - 2*u[i,1:ny-1,k] + u[i,0:ny-2,k]))
        # Boundary conditions
        u[0,:,k+1] = 0
        u[nx-1,:,k+1] = 0
        u[:,0,k+1] = 0
        u[:,ny-1,k+1] = 0

    return u

# Define problem parameters
nx = 501
ny = 501
nt = 500
Lx = 1
Ly = 1
kappa = 0.01

u = solve_heat_equation(nx, ny, nt, Lx, Ly, kappa)

dx = Lx / (nx-1)
dy = Ly / (ny-1)
dt = dx**2 * dy**2 / (2*kappa*(dx**2 + dy**2))

# Plot solution
import matplotlib.pyplot as plt
plt.imshow(u[:,:,nt-1], cmap='coolwarm', extent=[0, Lx, 0, Ly], origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Temperature distribution after {nt*dt:.2f} seconds')
plt.show()
