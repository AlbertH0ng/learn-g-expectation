import deepxde as dde
import numpy as np
import tensorflow as tf
# from deepxde.backend import tf
import matplotlib.pyplot as plt
import os

# tf.config.optimizer.set_jit(False)  # Disable XLA

# ========= Set Model Type and Generator Name =========
model_type = "constant"  # Options: "constant", "BSM", "OUMR"
# constant: dX_t = 0.12 dt + 0.2 dW_t
# BSM: dX_t = 0.12 X_t dt + 0.25 X_t dW_t
# OUMR: dX_t = 0.5 (1 - X_t) dt + t exp(-t) dW_t

generator_name = "g_0"      # Options: "g_0", "g_1", "g_2", "g_3"
# g_0: g(t, y, z) = 0
# g_1: g(t, y, z) = 2 * sqrt(z^2 + epsilon)
# g_2: g(t, y, z) = y + sqrt(z^2 + epsilon)
# g_3: g(t, y, z) = (1/e) * exp(y) + sqrt(z^2 + epsilon)

# Create a directory for the model if it doesn't exist
if not os.path.exists(model_type):
    os.makedirs(model_type)

# ========= Define Model Domain Based on Model Type =========
if model_type == "constant":
    T = 5
    x_min = 0.0
    x_max = 5.0
elif model_type == "BSM":
    T = 5
    x_min = 0.0
    x_max = 10.0
elif model_type == "OUMR":
    T = 10
    x_min = 0.0
    x_max = 5.0
else:
    raise ValueError("Invalid model_type. Choose from 'constant', 'BSM', or 'OUMR'.")

# ========= Define Domain =========
geom = dde.geometry.Interval(x_min, x_max)
timedomain = dde.geometry.TimeDomain(0, T)  # s ∈ [0, T]
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ========= Define Known Functions Based on Model Type =========
def mu_func(t, x):
    if model_type == "constant":
        return 0.12 * tf.ones_like(x)
    elif model_type == "BSM":
        return 0.12 * x
    elif model_type == "OUMR":
        k = 0.5
        theta = 1.0
        return k * (theta - x)
    else:
        raise ValueError("Invalid model_type.")

def sigma_func(t, x):
    if model_type == "constant":
        return 0.2 * tf.ones_like(x)
    elif model_type == "BSM":
        return 0.25 * x
    elif model_type == "OUMR":
        return t * tf.exp(-t) * tf.ones_like(x)
    else:
        raise ValueError("Invalid model_type.")

def g_func(t, y, z):
    # Define the generator function g based on generator_name
    epsilon = 1e-6  # For smooth approximation
    if generator_name == "g_0":
        return tf.zeros_like(y)
    elif generator_name == "g_1":
        return 2 * tf.sqrt(tf.square(z) + epsilon)
    elif generator_name == "g_2":
        return y + tf.sqrt(tf.square(z) + epsilon)
    elif generator_name == "g_3":
        return (1 / np.e) * tf.exp(y) + tf.sqrt(tf.square(z) + epsilon)
    else:
        raise ValueError("Invalid generator_name. Choose from 'g_0', 'g_1', 'g_2', 'g_3'.")

# ========= Define PDE =========
def pde(x, v):
    s = x[:, 0:1]  # s = T - t
    x_ = x[:, 1:2]
    t = T - s  # Convert back to t

    # Compute derivatives
    v_s = dde.grad.jacobian(v, x, i=0, j=0)  # ∂v/∂s
    v_x = dde.grad.jacobian(v, x, i=0, j=1)  # ∂v/∂x
    v_xx = dde.grad.hessian(v, x, component=0, i=1, j=1)  # ∂²v/∂x²

    # Compute the functions
    sigma_val = sigma_func(t, x_)
    mu_val = mu_func(t, x_)
    sigma_v_x = sigma_val * v_x
    g_val = g_func(t, v, sigma_v_x)

    # Initialize residual to avoid UnboundLocalError
    residual = None

    # Check for NaNs or Infs
    try:
        v_s = tf.debugging.check_numerics(v_s, "v_s contains NaN or Inf")
    except tf.errors.InvalidArgumentError as e:
        print("Error in v_s:", e)
        raise

    try:
        v_x = tf.debugging.check_numerics(v_x, "v_x contains NaN or Inf")
    except tf.errors.InvalidArgumentError as e:
        print("Error in v_x:", e)
        raise

    try:
        v_xx = tf.debugging.check_numerics(v_xx, "v_xx contains NaN or Inf")
    except tf.errors.InvalidArgumentError as e:
        print("Error in v_xx:", e)
        raise

    try:
        sigma_val = tf.debugging.check_numerics(sigma_val, "sigma_val contains NaN or Inf")
    except tf.errors.InvalidArgumentError as e:
        print("Error in sigma_val:", e)
        raise

    try:
        mu_val = tf.debugging.check_numerics(mu_val, "mu_val contains NaN or Inf")
    except tf.errors.InvalidArgumentError as e:
        print("Error in mu_val:", e)
        raise

    try:
        sigma_v_x = tf.debugging.check_numerics(sigma_v_x, "sigma_v_x contains NaN or Inf")
    except tf.errors.InvalidArgumentError as e:
        print("Error in sigma_v_x:", e)
        raise

    try:
        g_val = tf.debugging.check_numerics(g_val, "g_val contains NaN or Inf")
    except tf.errors.InvalidArgumentError as e:
        print("Error in g_val:", e)
        raise

    # Compute residual
    residual = v_s - (0.5 * tf.square(sigma_val) * v_xx + mu_val * v_x + g_val)

    try:
        residual = tf.debugging.check_numerics(residual, "Residual contains NaN or Inf")
    except tf.errors.InvalidArgumentError as e:
        print("Error in residual:", e)
        raise

    return residual

# ========= Define Initial & Boundary Conditions =========
def ic_func(x):
    x_ = x[:, 1:2]
    return -x_

ic = dde.IC(
    geomtime,
    ic_func,
    lambda x, on_initial: on_initial,
)

# Define the boundary condition for all boundary points
def boundary_condition(x, on_boundary):
    return on_boundary

# Set the points on boundary equal to the classical expectation
def bc_func(x):
    s = x[:, 0:1]  # s = T - t
    x_ = x[:, 1:2]
    t = T - s  # Convert back to t

    if model_type == "constant":
        mu = tf.constant(0.12, dtype=tf.float32)
        expected_X_T = x_ + mu * (T - t)
        return -expected_X_T
    elif model_type == "BSM":
        mu = tf.constant(0.12, dtype=tf.float32)
        exp_argument = mu * (T - t)
        exp_argument = tf.clip_by_value(exp_argument, clip_value_min=-100.0, clip_value_max=100.0)  # Prevent overflow
        expected_X_T = x_ * tf.exp(exp_argument)
        return -expected_X_T
    elif model_type == "OUMR":
        k = tf.constant(0.5, dtype=tf.float32)
        theta = tf.constant(1.0, dtype=tf.float32)
        exponent = tf.exp(-k * (T - t))
        expected_X_T = x_ * exponent + theta * (1 - exponent)
        return -expected_X_T
    else:
        raise ValueError("Invalid model_type.")

bc = dde.DirichletBC(geomtime, bc_func, boundary_condition)

# ========= Create Data =========
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc],  # Include initial and boundary conditions
    num_domain=4000,
    num_boundary=200,
    num_initial=200,
)

# ========= Create Model and Train =========
with tf.device('/GPU:0'):
    net = dde.maps.FNN([2] + [50] * 3 + [1], "tanh", "Glorot uniform")

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    # Train with Adam optimizer
    losshistory, train_state = model.train(epochs=10000)

    # Optionally, switch to L-BFGS-B optimizer for refinement
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()

# ========= Evaluation & Plot =========
# Create a mesh grid for s and x
t = np.linspace(0, T, 100)
x = np.linspace(x_min, x_max, 100)
t_mesh, x_mesh = np.meshgrid(t, x)

# Convert t to s for prediction
s_mesh = T - t_mesh

# Flatten the mesh grids and stack them for prediction
X = np.vstack((s_mesh.flatten(), x_mesh.flatten())).T

# Predict the solution using the model
y_pred = model.predict(X)
y_pred = y_pred.reshape(t_mesh.shape)

# Plot the solution in terms of t and x
plt.figure()
plt.contourf(t_mesh, x_mesh, y_pred, 100, cmap='rainbow')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title(r'$v$ values for ' + model_type + ' model with $' + generator_name + '$')

# Save the plot before showing it
plt.savefig(f"{model_type}/{model_type}_{generator_name}_Result.png")
plt.show()

# Plot and save the training loss history
dde.utils.plot_loss_history(losshistory)
plt.savefig(f"{model_type}/{model_type}_{generator_name}_loss_history.png")
plt.close()

# ======== Save the results ========
model.save(f"{model_type}/{model_type}_{generator_name}_model.ckpt")