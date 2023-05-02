import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 1 # Length of the rod
T = 1 # Total time
n = 50 # Number of grid points
m = 1000 # Number of time steps
alpha = 0.01 # Thermal diffusivity
dx = L/n # Grid spacing
dt = T/m # Time step

# Initialize temperature array
u = np.zeros((n+1, m+1))

# Set initial conditions
u[:,0] = np.sin(np.pi*np.linspace(0,L,n+1))

# Set boundary conditions
u[0,:] = u[n,:] = 0

# Define finite difference coefficients
a = alpha*dt/(dx*dx)
b = 1 - 2*a

# Implement finite difference method
for j in range(0,m):
    for i in range(1,n):
        u[i,j+1] = a*u[i-1,j] + b*u[i,j] + a*u[i+1,j]

# Plot results
x = np.linspace(0,L,n+1)
t = np.linspace(0,T,m+1)
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u.T)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.show()
