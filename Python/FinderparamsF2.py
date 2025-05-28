import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import optuna

# Define the potential V(phi)
def V(phi, C, lambda_, chi, mu, omega, kappa):
    theta = C * phi
    s = np.sin(theta)
    c = np.cos(theta)
    term1 = lambda_ * s**4
    term2 = 2 * chi * s**3 * c
    term3 = mu * s**2 * c**2
    term4 = 2 * omega * s * c**3
    term5 = kappa * c**4
    return (1/8) * (term1 + term2 + term3 + term4 + term5)

# Define the first derivative V'(phi)
def V_prime(phi, C, lambda_, chi, mu, omega, kappa):
    theta = C * phi
    term1 = (-chi + omega) * np.cos(4*theta)
    term2 = (-kappa + lambda_) * np.sin(2*theta)
    term3 = np.cos(2*theta) * (chi + omega - (kappa + lambda_ - mu) * np.sin(2*theta))
    return (C / 8) * (term1 + term2 + term3)

# Define the second derivative V''(phi)
def V_double_prime(phi, C, lambda_, chi, mu, omega, kappa):
    theta = C * phi
    term1 = (kappa - lambda_) * np.cos(2*theta)
    term2 = (kappa + lambda_ - mu) * np.cos(4*theta)
    term3 = (chi + omega) * np.sin(2*theta)
    term4 = 2 * (-chi + omega) * np.sin(4*theta)
    return - (C**2 / 4) * (term1 + term2 + term3 + term4)

# Objective function for Optuna
def objective(trial):
    C = trial.suggest_float('C', 0.1, 10)
    mu = trial.suggest_float('mu', -10, 10)
    omega = trial.suggest_float('omega', -10, 10)
    chi = trial.suggest_float('chi', -10, 10)
    lambda_ = trial.suggest_float('lambda', -10, 10)
    kappa = trial.suggest_float('kappa', -10, 10)
    params = (C, lambda_, chi, mu, omega, kappa)
    
    # Check if V(0) > 0
    if V(0, *params) <= 0:
        return 1e10
    
    # Find phi_end where epsilon = 1
    def g(phi):
        v = V(phi, *params)
        if v <= 0:
            return 1e10
        vp = V_prime(phi, *params)
        return (vp / v)**2 - 2
    
    try:
        phi_end = brentq(g, 0.001, np.pi / (2 * C) - 0.001)
    except ValueError:
        return 1e10
    
    # Define integrand for N(phi)
    def integrand(p):
        v = V(p, *params)
        vp = V_prime(p, *params)
        return - v / vp if vp != 0 else 0
    
    # Define N(phi)
    def N_func(phi):
        if phi >= phi_end:
            return 0
        N, _ = quad(integrand, phi, phi_end)
        return N
    
    # Find phi_60 where N(phi_60) = 60
    def h(phi):
        return N_func(phi) - 60
    
    # Check if N at some phi is >60
    phi_test = phi_end / 2
    N_test = N_func(phi_test)
    if N_test < 60:
        return 1e10
    
    try:
        phi_60 = brentq(h, 0.001, phi_end - 0.001)
    except ValueError:
        return 1e10
    
    # Compute epsilon and eta at phi_60
    v = V(phi_60, *params)
    vp = V_prime(phi_60, *params)
    vpp = V_double_prime(phi_60, *params)
    epsilon = 0.5 * (vp / v)**2
    eta = vpp / v
    n_s = 1 - 6 * epsilon + 2 * eta
    cost = (n_s - 0.9649)**2
    
    # Print progress
    print(f"Trial {trial.number}: cost={cost}, params={params}, n_s={n_s}")
    return cost

# Run Optuna optimization
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

# Get the best parameters
best_params = study.best_params
print("Optimal parameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

# Save to text file
with open('optimal_params.txt', 'w') as f:
    for key, value in best_params.items():
        f.write(f"{key}: {value}\n")

# Save to LaTeX file
with open('optimal_params.tex', 'w') as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{amsmath}\n")
    f.write("\\begin{document}\n")
    f.write("\\begin{align*}\n")
    for key, value in best_params.items():
        f.write(f"{key} &= {value} \\\\\n")
    f.write("\\end{align*}\n")
    f.write("\\end{document}\n")