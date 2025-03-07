import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
M = 1  # Planck mass (set to 1 for simplicity)
k = 50  # Example value for k
gamma = 10  # Example value for gamma
alpha = 1  # Example value for alpha
N_target = 10  # Number of e-folds to evaluate

# Function definitions
def D_func(gamma, alpha, beta):
    return np.sqrt(gamma**2 - 4 * alpha * beta)

def f(X, gamma, D):
    return gamma * np.cosh(X) + D * np.sinh(X)

def epsilon_func(phi, beta, gamma, alpha):
    D = D_func(gamma, alpha, beta)
    K = np.sqrt(2 * beta / (k - 6 * beta))
    X = K * phi / (2 * M)
    sech_X2 = 1 / np.cosh(X)**2
    
    num = f(2*X, gamma, D)
    den = f(X, gamma, D) * (2 * gamma**2 - D**2 + 2 * gamma * f(2*X, gamma, D))
    
    epsilon = (M**2 * beta / (k - 6 * beta)) * D**4 * sech_X2 * (num / den)**2
    return epsilon

# Solve equation of motion for phi(N)
def equation_of_motion(N, phi, beta):
    return [phi[1], -3 * phi[1] - 0.5 * phi[1]**3 + (3 - 0.5 * phi[1]**2) * epsilon_func(phi[0], beta, gamma, alpha)]

# Range of beta values to evaluate
beta_values = np.linspace(0.1, 5, 100)
epsilon_values = []

for beta in beta_values:
    sol = solve_ivp(equation_of_motion, [0, N_target], [1.5, 0.1], args=(beta,), t_eval=[N_target])
    phi_N = sol.y[0][-1]  # Get phi at N_target
    epsilon_values.append(epsilon_func(phi_N, beta, gamma, alpha))

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(beta_values, epsilon_values, label=r'$\varepsilon(\beta)$')
plt.axhline(1, color='r', linestyle='dashed', label=r'$\varepsilon = 1$ (Inflation End)')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\varepsilon$')
plt.title(r'Slow-Roll Parameter $\varepsilon$ as Function of $\beta$')
plt.legend()
plt.grid()
plt.show()
