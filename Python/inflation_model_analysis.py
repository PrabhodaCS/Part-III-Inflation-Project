import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.optimize import minimize

# Set reduced Planck mass to 1 for simplicity
Mp = 1.0

# Define parameters
params = {
    'alpha': 1.0,
    'beta': 1.0,
    'kappa': 1.0,
    'mu': 1.0,
    'lambda_': 1.0,
    'gamma': 0.0,
    'omega': 0.0,
    'chi': 0.0,
    'k': 0.0
}

# Define X(phi)
def X_phi(phi, params):
    alpha, beta, gamma, k = params['alpha'], params['beta'], params['gamma'], params['k']
    term = np.sqrt(4 * alpha * beta - gamma**2)
    arg = np.sqrt(2 / (12 + k)) * phi / Mp
    return (1 / (2 * beta)) * (term * np.tan(arg) - gamma)

# Define potential V(X)
def V_X(X, params):
    kappa, omega, mu, chi, lambda_ = params['kappa'], params['omega'], params['mu'], params['chi'], params['lambda_']
    alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
    numerator = kappa + 2 * omega * X + mu * X**2 + 2 * chi * X**3 + lambda_ * X**4
    denominator = (beta * X**2 + gamma * X + alpha)**2
    return (Mp**4 / 8) * numerator / denominator

# Derivative dX/dphi
def dX_dphi(phi, params):
    alpha, beta, gamma, k = params['alpha'], params['beta'], params['gamma'], params['k']
    term = np.sqrt(4 * alpha * beta - gamma**2)
    arg = np.sqrt(2 / (12 + k)) * phi / Mp
    return (1 / (2 * beta)) * term * (1 / np.cos(arg))**2 * np.sqrt(2 / (12 + k)) / Mp

# Derivative dV/dX
def dV_dX(X, params):
    kappa, omega, mu, chi, lambda_ = params['kappa'], params['omega'], params['mu'], params['chi'], params['lambda_']
    alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
    P = kappa + 2 * omega * X + mu * X**2 + 2 * chi * X**3 + lambda_ * X**4
    Q = beta * X**2 + gamma * X + alpha
    dP_dX = 2 * omega + 2 * mu * X + 6 * chi * X**2 + 4 * lambda_ * X**3
    dQ_dX = 2 * beta * X + gamma
    numerator = dP_dX * Q - 2 * P * dQ_dX
    denominator = Q**3
    return (Mp**4 / 8) * numerator / denominator

# Potential V(phi)
def V_phi(phi, params):
    return V_X(X_phi(phi, params), params)

# Derivative dV/dphi
def dV_dphi(phi, params):
    return dV_dX(X_phi(phi, params), params) * dX_dphi(phi, params)

# Second derivative d^2V/dphi^2 (numerical approximation)
def d2V_dphi2(phi, params, delta=1e-6):
    return (dV_dphi(phi + delta, params) - dV_dphi(phi - delta, params)) / (2 * delta)

# Slow-roll parameters
def epsilon(phi, params):
    V = V_phi(phi, params)
    dV = dV_dphi(phi, params)
    if V == 0:
        return np.inf
    return 0.5 * (Mp * dV / V)**2

def eta(phi, params):
    V = V_phi(phi, params)
    d2V = d2V_dphi2(phi, params)
    if V == 0:
        return np.inf
    return Mp**2 * d2V / V

# Find phi_end where epsilon = 1 or |eta| = 1
def find_phi_end(params, phi_start=3.0, phi_end=0.0, steps=1000):
    phi_vals = np.linspace(phi_start, phi_end, steps)
    for phi in phi_vals:
        eps = epsilon(phi, params)
        epsilon_value = 1e-6
        eta_val = eta(phi, params)
        if eps >= 1 or abs(eta_val) >= 1:
            return phi
    return phi_vals[-1]

# Number of e-folds
def integrand_N(phi, params):
    V = V_phi(phi, params)
    dV = dV_dphi(phi, params)
    if dV == 0:
        return 0
    return V / (Mp**2 * dV)

def N_efolds(phi, phi_end, params):
    result, _ = quad(integrand_N, phi, phi_end, args=(params,), limit=10000)
    return result

# Find phi_N for N=60
def find_phi_N(params, N_target=60):
    phi_end = find_phi_end(params)
    def objective(phi):
        return abs(N_efolds(phi, phi_end, params) - N_target)
    result = minimize(objective, x0=phi_end + 1.0, bounds=[(phi_end, 4.0)])
    return result.x[0]

# Compute n_s
def n_s(phi, params):
    eps = epsilon(phi, params)
    eta_val = eta(phi, params)
    return 1 - 6 * eps + 2 * eta_val

# Objective function for parameter tuning
def objective_n_s(param_values, param_keys, target_n_s=0.9626, N_target=60):
    temp_params = params.copy()
    for key, value in zip(param_keys, param_values):
        temp_params[key] = value
    phi_N = find_phi_N(temp_params, N_target)
    ns = n_s(phi_N, temp_params)
    return abs(ns - target_n_s)

# Tune parameters
param_keys = ['alpha', 'beta', 'kappa', 'mu', 'lambda_']
initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0]
bounds = [(0.1, 10.0)] * len(param_keys)
result = minimize(objective_n_s, initial_guess, args=(param_keys, 0.9626, 60), bounds=bounds)
optimized_params = params.copy()
for key, value in zip(param_keys, result.x):
    optimized_params[key] = value

# Phase space dynamics
def phase_space_dynamics(state, t, params):
    phi, phi_dot = state
    V_prime = dV_dphi(phi, params)
    H = np.sqrt((0.5 * phi_dot**2 + V_phi(phi, params)) / (3 * Mp**2))
    phi_ddot = -3 * H * phi_dot - V_prime
    return [phi_dot, phi_ddot]

# Generate phase space data for uncompactified plot
phi_range = np.linspace(-4, 4, 100)
phi_dot_range = np.linspace(-1, 1, 100)
Phi, Phi_dot = np.meshgrid(phi_range, phi_dot_range)
U = np.zeros_like(Phi)
V = np.zeros_like(Phi_dot)
for i in range(Phi.shape[0]):
    for j in range(Phi.shape[1]):
        state = [Phi[i, j], Phi_dot[i, j]]
        derivs = phase_space_dynamics(state, 0, optimized_params)
        U[i, j] = derivs[0]
        V[i, j] = derivs[1]
speed = np.sqrt(U**2 + V**2)

# Generate phase space data for compactified plot
epsilon = 1e-6
theta_vals = np.linspace(-np.pi/2 + epsilon, np.pi/2 - epsilon, 100)
psi_vals = np.linspace(-np.pi/2 + epsilon, np.pi/2 - epsilon, 100)
Theta, Psi = np.meshgrid(theta_vals, psi_vals)
Phi_comp = np.tan(Theta)
Phi_dot_comp = np.tan(Psi)
U_comp = np.zeros_like(Phi_comp)
V_comp = np.zeros_like(Phi_dot_comp)
for i in range(Phi_comp.shape[0]):
    for j in range(Phi_comp.shape[1]):
        phi = Phi_comp[i, j]
        phi_dot = Phi_dot_comp[i, j]
        state = [phi, phi_dot]
        derivs = phase_space_dynamics(state, 0, optimized_params)
        U_comp[i, j] = derivs[0] / (1 + phi**2)  # d theta/dt
        V_comp[i, j] = derivs[1] / (1 + phi_dot**2)  # d psi/dt
speed_comp = np.sqrt(U_comp**2 + V_comp**2)

# Plotting
plt.figure(figsize=(15, 10))

# Potential plot
phi_vals = np.linspace(-4, 4, 1000)
V_vals = [V_phi(phi, optimized_params) for phi in phi_vals]
plt.subplot(2, 2, 1)
plt.plot(phi_vals, V_vals, 'b-')
plt.title('Inflationary Potential')
plt.xlabel(r'$\varphi$')
plt.ylabel(r'$V(\varphi)$')
plt.grid(True)

# Uncompactified phase space
plt.subplot(2, 2, 2)
plt.streamplot(Phi, Phi_dot, U, V, color=speed, cmap='viridis')
plt.title('Uncompactified Phase Space')
plt.xlabel(r'$\varphi$')
plt.ylabel(r'$\dot{\varphi}$')
plt.colorbar(label='Velocity')
plt.grid(True)

# Compactified phase space
plt.subplot(2, 2, 3)
plt.streamplot(Theta, Psi, U_comp, V_comp, color=speed_comp, cmap='viridis')
plt.title('Compactified Phase Space')
plt.xlabel(r'$\theta = \arctan(\varphi)$')
plt.ylabel(r'$\psi = \arctan(\dot{\varphi})$')
plt.colorbar(label='Velocity')
plt.grid(True)

# Epsilon and eta vs N
phi_end = find_phi_end(optimized_params)
phi_N = find_phi_N(optimized_params)
phi_traj = np.linspace(phi_N, phi_end, 100)
N_vals = [N_efolds(phi, phi_end, optimized_params) for phi in phi_traj]
eps_vals = [epsilon(phi, optimized_params) for phi in phi_traj]
eta_vals = [eta(phi, optimized_params) for phi in phi_traj]
plt.subplot(2, 2, 4)
plt.plot(N_vals, eps_vals, 'r-', label=r'$\epsilon$')
plt.plot(N_vals, eta_vals, 'b-', label=r'$\eta$')
plt.title('Slow-Roll Parameters vs. e-folds')
plt.xlabel('N')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('inflation_plots.png')
plt.close()

# Save parameters
print("Optimized Parameters:", optimized_params)
print("n_s at N=60:", n_s(phi_N, optimized_params))