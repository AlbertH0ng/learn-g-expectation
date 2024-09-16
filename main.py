import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt
import os

# ========= Set Example Name =========
example_name = "OUMR model" 
generator_name = "g_2"  
# Choose the generator function g from the following list:
'''
$g_1 (t,y,z) = 2 |z|$, 
$g_2 (t,y,z) = y + |z|$, 
$g_3 (t,y,z) = 1/e \exp(y) + |z|$.
'''

if not os.path.exists(example_name):
    os.makedirs(example_name)

# ========= Define Domain =========
T = 10 # Terminal time
x_min = 0.0  # x ∈ [x_min, x_max]
x_max = 3.0  

geom = dde.geometry.Interval(x_min, x_max)
timedomain = dde.geometry.TimeDomain(0, T)  # s ∈ [0, T]
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ========= Define Known Functions=========
def mu_func(t, x):
    # Define mu, the drift function here
    # return 0.12 * x  # BSM model
    return 0.5*(1 - x)  # OUMR model

def sigma_func(t, x):
    # Define sigma, the diffusion function here
    # return 0.25 * x  # BSM model
    return t*tf.exp(-t)  # OUMR model

def g_func(t, y, z):
    # Define the generate function g here
    # return 2*tf.abs(z)  # for g_1
    return y + tf.abs(z) # for g_2
    # return 1/np.e * tf.exp(y) + tf.abs(z) # for g_3

# ========= Define PDE =================
def pde(x, v):
    s = x[:, 0:1]  # s = T - t
    x_ = x[:, 1:2]
    t = T - s  # Convert back to original time variable t

    # Compute derivatives
    v_s = dde.grad.jacobian(v, x, i=0, j=0)  # Partial derivative w.r.t s
    v_x = dde.grad.jacobian(v, x, i=0, j=1)  # Partial derivative w.r.t x
    v_xx = dde.grad.hessian(v, x, component=0, i=1, j=1)  # Second partial derivative w.r.t x

    # Compute the functions
    sigma_val = sigma_func(t, x_)
    mu_val = mu_func(t, x_)
    sigma_v_x = sigma_val * v_x
    g_val = g_func(t, v, sigma_v_x)

    # Return the PDE residual
    return v_s - (0.5 * sigma_val ** 2 * v_xx + mu_val * v_x + g_val)

# ========= Define Initial & Boundary Conditions =========
def ic_func(x):
    x_ = x[:, 1:2]
    return -x_  

ic = dde.IC(
    geomtime,
    ic_func,
    lambda x, on_initial: on_initial,
)

def boundary_condition(x, on_boundary):
    return on_boundary

def bc_func(x):
    x_ = x[:, 1:2]
    return -x_  

bc = dde.DirichletBC(geomtime, bc_func, boundary_condition)

# ========= Create Data =========
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc],
    num_domain=1000,
    num_boundary=100,
    num_initial=100,
)

# ========= Create Model =========
net = dde.maps.FNN([2] + [50] * 3 + [1], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# Train with Adam optimizer
losshistory, train_state = model.train(epochs=10000)

# Optionally, switch to L-BFGS-B optimizer for refinement
model.compile("L-BFGS-B")
losshistory, train_state = model.train()

# ========= Evalution & Plot =========
# Create a mesh grid for s and x
s = np.linspace(0, T, 100)
x = np.linspace(x_min, x_max, 100)
s_mesh, x_mesh = np.meshgrid(s, x)

# Flatten the mesh grids and stack them for prediction
X = np.vstack((s_mesh.flatten(), x_mesh.flatten())).T

# Predict the solution using the model
y_pred = model.predict(X)
y_pred = y_pred.reshape(s_mesh.shape)

# Convert s back to t for plotting
t_mesh = T - s_mesh

# Plot the solution in terms of t and x
plt.figure()
plt.contourf(t_mesh, x_mesh, y_pred, 100, cmap='rainbow')
plt.colorbar()

plt.xlabel('t')
plt.ylabel('x')
plt.title('v(t, x) values')

# Save the plot before showing it
plt.savefig(f"{example_name}/{example_name}_{generator_name}_Result.png")
plt.show()

# Plot and save the training loss history
dde.utils.plot_loss_history(losshistory)
plt.savefig(f"{example_name}/{example_name}_{generator_name}_loss_history.png")
plt.close()

# ======== Save the results ========
model.save(f"{example_name}/{example_name}_{generator_name}_model.ckpt")


