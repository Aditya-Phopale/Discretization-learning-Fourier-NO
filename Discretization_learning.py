import torch 
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim

class Neural_Diffeq(nn.Module):
    def __init__(self, D, v, dx, time_chunks, dt):
        super().__init__()
        self.diffusion_conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.gradient_conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)


        self.D = D
        self.v = v
        self.dx = dx
        self.time_chunks = time_chunks
        self.dt = dt

        self.alpha = 0.1 * self.dt / (2 * self.dx)    # Courant number for convection in x
        self.beta = self.D * self.dt / (self.dx ** 2) 
        
    def init_kernels(self):
        laplacian_kernel = torch.tensor(
                            [[[[0.,  1.,  0.],
                                [1., -4.,  1.],
                                [0.,  1.,  0.]]]]
                            )
        gradient_kernel = torch.tensor(
                            [[[[0.,  -1.,  0.],
                                [-1., 0.,  1.],
                                [0.,  1.,  0.]]]]
                            )
        
        self.diffusion_conv.weight = nn.Parameter(laplacian_kernel)
        self.gradient_conv.weight = nn.Parameter(gradient_kernel)

    def forward(self, x):

        ## Replace 0.1 by v[0] for different velocities
        for _ in range(self.time_chunks):
            x = x + self.beta * self.diffusion_conv(x) - self.alpha*self.gradient_conv(x)
        
        return x

    def propagate(self, x, timesteps):
        for _ in range(timesteps):
            x = x + self.beta * self.diffusion_conv(x) - self.alpha*self.gradient_conv(x)

        return x

# def create_chunked_dataset(data, sequence_length):
#     """
#     Prepare dataset for RNN where the input is a sequence of `sequence_length` timesteps,
#     and the output is the next timestep.
    
#     Parameters:
#     - data (torch.Tensor): The input data of shape (timesteps, height, width).
#     - sequence_length (int): The number of timesteps to use for each sequence.

#     Returns:
#     - inputs (torch.Tensor): The input sequences for the RNN of shape (num_sequences, sequence_length, height, width).
#     - targets (torch.Tensor): The target timesteps of shape (num_sequences, height, width).
#     """
#     timesteps, height, width = data.shape
    
#     # Number of sequences we can create
#     num_sequences = timesteps - sequence_length
    
#     # Initialize lists to store input sequences and targets
#     inputs = []
#     targets = []
    
#     # Loop through the dataset to create input-target pairs
#     for i in range(num_sequences):
#         input_sequence = data[i:i+sequence_length]  # Take sequence of `sequence_length` timesteps
#         target = data[i+sequence_length]  # The next timestep
        
#         inputs.append(input_sequence)
#         targets.append(target)
    
#     # Convert lists to torch tensors
#     inputs = torch.stack(inputs)  # Shape: (num_sequences, sequence_length, height, width)
#     targets = torch.stack(targets)  # Shape: (num_sequences, height, width)
    
#     return inputs, targets

# # Example usage
# data = torch.randn(500, 100, 100)  # Example data with shape (timesteps, height, width)
# sequence_length = 5
# inputs, targets = create_chunked_dataset(data, sequence_length)

# print("Input shape:", inputs.shape)  # Should be (495, 5, 100, 100)
# print("Target shape:", targets.shape)  # Should be (495, 100, 100)



D = 0.01
v = [0.1,0.1]

domain = [-1,1]
n_dx_elements = 100
dx = (domain[1] - domain[0])/(n_dx_elements)
x = np.linspace(domain[0], domain[1], n_dx_elements)
y = np.linspace(domain[0], domain[1], n_dx_elements)
mesh_x, mesh_y = np.meshgrid(x,y)

c = torch.zeros(1,1,100,100)
c_init_r = 0.25

for i in range(1,len(x)-1):
    for j in range(1,len(y)-1):
        if x[i]**2 + y[j]**2 < c_init_r**2:
            c[0,0,i,j] = 1.0

fig, ax = plt.subplots(figsize=(5, 5))
contour1 = ax.contourf(mesh_x, mesh_y, torch.squeeze(c), levels=20, cmap='viridis')

colorbar = plt.colorbar(contour1, ax=ax)
plt.show()

t_end = 5
dt = 0.01
timesteps = int(t_end/dt)
t_lin = np.linspace(0,t_end,timesteps)
time_chunks = 1

Neural_solver = Neural_Diffeq(D, v, dx, time_chunks, dt)

## Uncomment if want the kernels to be exactly same as diffusion and gradeint kernels
# Neural_solver.init_kernels()

# # print(Neural_solver.diffusion_conv.weight)
# output = Neural_solver(c)

# output = torch.squeeze(output)

# fig, ax = plt.subplots(figsize=(5, 5))
# contour1 = ax.contourf(mesh_x, mesh_y, output.detach().numpy(), levels=20, cmap='viridis')

# colorbar = plt.colorbar(contour1, ax=ax)
# plt.show()

loaded_data = np.load('dataset/simulation_data.npy')

input_field = loaded_data[:-1, 1:-1, 1:-1]
output_field = loaded_data[1:, 1:-1, 1:-1]

input_field = torch.tensor(input_field[:,None,:,:], dtype=torch.float32)
output_field = torch.tensor(output_field[:,None,:,:], dtype=torch.float32)

epochs = 100
criterion = nn.MSELoss()
optimizer = optim.Adam(Neural_solver.parameters(), lr=0.01)

for epoch in range(epochs):

    optimizer.zero_grad()

    prediction = Neural_solver(input_field)

    loss = criterion(prediction, output_field)
    loss.backward()

    optimizer.step()
    print(epoch+1,"/", epochs, " loss: ",loss.item())

print("Laplacian kernel parameters: ", Neural_solver.diffusion_conv.weight)
print("Gradient kernal parameters: ", Neural_solver.gradient_conv.weight)


## Prediction:
domain = [-1,1]
dim = 2
n_dx_elements = 100
dx = (domain[1] - domain[0])/(n_dx_elements)
x = np.linspace(domain[0]-dx, domain[1]+dx, n_dx_elements+2)
y = np.linspace(domain[0]-dx, domain[1]+dx, n_dx_elements+2)
mesh_x, mesh_y = np.meshgrid(x[1:-1],y[1:-1])

c = np.zeros((len(x), len(y)))
c_init_r = 0.25
for i in range(1,len(x)-1):
    for j in range(1,len(y)-1):
        if (x[i])**2 + (y[j])**2 < c_init_r**2:
            c[i,j] = 1.0

fig, ax = plt.subplots(figsize=(5, 5))
contour1 = ax.contourf(mesh_x, mesh_y, c[1:-1,1:-1], levels=20, cmap='viridis')

colorbar = plt.colorbar(contour1, ax=ax)
plt.show()

Neural_solver.eval()
with torch.no_grad():
    eval_input_tensor = torch.tensor(c[None,:,:], dtype=torch.float32)
    eval_output_tensor = Neural_solver.propagate(eval_input_tensor, 100)

eval_output_tensor = torch.squeeze(eval_output_tensor)
fig, ax = plt.subplots(figsize=(5, 5))
contour1 = ax.contourf(mesh_x, mesh_y, eval_output_tensor[1:-1,1:-1].detach().numpy(), levels=20, cmap='viridis')

colorbar = plt.colorbar(contour1, ax=ax)
plt.show()