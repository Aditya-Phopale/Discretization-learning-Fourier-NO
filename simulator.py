import numpy as np
import matplotlib.pyplot as plt

domain = [-1,1]
dim = 2
n_dx_elements = 100
dx = (domain[1] - domain[0])/(n_dx_elements)
x = np.linspace(domain[0]-dx, domain[1]+dx, n_dx_elements+2)
y = np.linspace(domain[0]-dx, domain[1]+dx, n_dx_elements+2)
mesh_x, mesh_y = np.meshgrid(x[1:-1],y[1:-1])


total_time = 5
dt = 0.01
timesteps = int(total_time/dt)
t_lin = np.linspace(0,total_time,timesteps)

D = 0.01
v = [0.1,0.1]
print(dx**2/(2*D))
alpha_x = v[0] * dt / (2 * dx)    # Courant number for convection in x
alpha_y = v[1] * dt / (2 * dx)    # Courant number for convection in y
beta_x = D * dt / (dx ** 2)     # Diffusion number in x
beta_y = D * dt / (dx ** 2)     # Diffusion number in y

c = np.zeros((len(t_lin), len(x), len(y)))
c_init_r = 0.25

## Initialization loop
for i in range(1,len(x)-1):
    for j in range(1,len(y)-1):
        if x[i]**2 + y[j]**2 < c_init_r**2:
            c[0,i,j] = 1.0

fig, ax = plt.subplots(figsize=(5, 5))
contour1 = ax.contourf(mesh_x, mesh_y, c[0,1:-1,1:-1], levels=20, cmap='viridis')

colorbar = plt.colorbar(contour1, ax=ax)
plt.show()

## Simulation loop:
## dc/dt = -v.grad(c) + D*lap(c)
for t in range(len(t_lin)-1):
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            c[t + 1, i, j] = (c[t, i, j]
                              - alpha_x * (c[t, i + 1, j] - c[t, i - 1, j])
                              - alpha_y * (c[t, i, j + 1] - c[t, i, j - 1])
                              + beta_x * (c[t, i + 1, j] - 2 * c[t, i, j] + c[t, i - 1, j])
                              + beta_y * (c[t, i, j + 1] - 2 * c[t, i, j] + c[t, i, j - 1]))
            
    c[t+1, 0, :] = 0
    c[t+1, len(x)-1, :] = 0
    c[t+1, :, 0] = 0
    c[t+1, :, len(y)-1] = 0
    print("Timestep: ", t,"/",len(t_lin)-1)

fig, ax = plt.subplots(figsize=(5, 5))
contour1 = ax.contourf(mesh_x, mesh_y, c[timesteps-1,1:-1,1:-1], levels=20, cmap='viridis')

colorbar = plt.colorbar(contour1, ax=ax)
plt.show()

np.save('dataset/simulation_data.npy', c)

