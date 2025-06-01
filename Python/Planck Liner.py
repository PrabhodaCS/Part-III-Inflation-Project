import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Optimized parameters
optimized_params = {
    'C': 1.763517,
    'lam': 8.865184,
    'chi': 21.644677,
    'omega': 0.0,
    'mu': 48.957682,
    'kappa': 24.086857
}

# Create output directory
os.makedirs('inflation_plots', exist_ok=True)

# Potential and derivatives
def V(phi, params):
    theta = params['C'] * phi
    s = np.sin(theta)
    c = np.cos(theta)
    return (1/8) * (params['lam'] * s**4 + 2*params['chi'] * s**3 * c + 
                    params['mu'] * s**2 * c**2 + 2*params['omega'] * s * c**3 + 
                    params['kappa'] * c**4)

def dV_dphi(phi, params):
    theta = params['C'] * phi
    s2 = np.sin(2*theta)
    c2 = np.cos(2*theta)
    c4 = np.cos(4*theta)
    term1 = (-params['chi'] + params['omega']) * c4
    term2 = (-params['kappa'] + params['lam']) * s2
    term3 = c2 * (params['chi'] + params['omega'] - 
                  (params['kappa'] + params['lam'] - params['mu']) * s2)
    return (params['C'] / 8) * (term1 + term2 + term3)

def d2V_dphi2(phi, params):
    theta = params['C'] * phi
    s2 = np.sin(2*theta)
    c2 = np.cos(2*theta)
    s4 = np.sin(4*theta)
    c4 = np.cos(4*theta)
    term1 = (params['kappa'] - params['lam']) * c2
    term2 = (params['kappa'] + params['lam'] - params['mu']) * c4
    term3 = (params['chi'] + params['omega']) * s2
    term4 = 2 * (-params['chi'] + params['omega']) * s4
    return -(params['C']**2 / 4) * (term1 + term2 + term3 + term4)

# Slow-roll parameters
def epsilon_V(phi, params):
    v = V(phi, params)
    if v <= 0:
        return np.inf
    dv = dV_dphi(phi, params)
    return (1/2) * (dv / v)**2

def eta_V(phi, params):
    v = V(phi, params)
    if v <= 0:
        return np.inf
    d2v = d2V_dphi2(phi, params)
    return d2v / v

# Find phi_end where epsilon_V = 1
def find_phi_end(params):
    C = params['C']
    phi_min = -np.pi / (2 * C) + 1e-6
    phi_max = np.pi / (2 * C) - 1e-6
    def objective(phi):
        if phi <= phi_min or phi >= phi_max:
            return 1e10
        eps = epsilon_V(phi, params)
        if np.isinf(eps):
            return 1e10
        return (eps - 1)**2
    res = minimize_scalar(objective, bounds=(phi_min, phi_max), method='bounded')
    if res.fun < 1e-6:
        return res.x
    return None

# Integrand for N
def integrand(phi, params):
    v = V(phi, params)
    dv = dV_dphi(phi, params)
    if abs(dv) < 1e-30 or v <= 0:
        return 0
    return -v / dv

# Find phi_60 where N = 60
def find_phi_60(phi_end, params):
    C = params['C']
    phi_min = -np.pi / (2 * C) + 1e-6
    phi_max = np.pi / (2 * C) - 1e-6
    dv_end = dV_dphi(phi_end, params)
    if dv_end > 0:
        def g(phi):
            if phi <= phi_end or phi >= phi_max:
                return -1e10
            integral, _ = quad(integrand, phi, phi_end, args=(params,), limit=100)
            return integral - 60
        try:
            phi_60 = brentq(g, phi_end + 1e-6, phi_max)
            return phi_60
        except:
            return None
    elif dv_end < 0:
        def g(phi):
            if phi >= phi_end or phi <= phi_min:
                return -1e10
            integral, _ = quad(integrand, phi, phi_end, args=(params,), limit=100)
            return integral - 60
        try:
            phi_60 = brentq(g, phi_min, phi_end - 1e-6)
            return phi_60
        except:
            return None
    return None

# Compute n_s and r
def compute_ns_r(params):
    phi_end = find_phi_end(params)
    if phi_end is None:
        print(f"Failed to find phi_end for params: {params}")
        return None, None
    phi_60 = find_phi_60(phi_end, params)
    if phi_60 is None:
        print(f"Failed to find phi_60 for params: {params}")
        return None, None
    eps = epsilon_V(phi_60, params)
    eta = eta_V(phi_60, params)
    if np.isinf(eps) or np.isinf(eta):
        print(f"Infinite eps or eta at phi_60: {phi_60}")
        return None, None
    n_s = 1 - 6*eps + 2*eta
    r = 16*eps
    return n_s, r

# Compute for optimized parameters
n_s_opt, r_opt = compute_ns_r(optimized_params)
print(f"Optimized: n_s = {n_s_opt}, r = {r_opt}")

# Vary mu
kappa_fixed = optimized_params['kappa']
mu_min = max(optimized_params['mu'] * 0.99, 2*kappa_fixed + 0.01)
mu_max = optimized_params['mu'] * 1.01
mu_values = np.linspace(mu_min, mu_max, 10)
ns_mu = []
r_mu = []
for mu in mu_values:
    params = optimized_params.copy()
    params['mu'] = mu
    n_s, r = compute_ns_r(params)
    if n_s is not None and np.isfinite(n_s) and r is not None and np.isfinite(r):
        ns_mu.append(n_s)
        r_mu.append(r)
    else:
        print(f"Skipping mu = {mu}: n_s or r invalid")

# Vary kappa
mu_fixed = optimized_params['mu']
kappa_min = optimized_params['kappa'] * 0.99
kappa_max = min(optimized_params['kappa'] * 1.01, mu_fixed / 2 - 0.01)
kappa_values = np.linspace(kappa_min, kappa_max, 10)
ns_kappa = []
r_kappa = []
for kappa in kappa_values:
    params = optimized_params.copy()
    params['kappa'] = kappa
    n_s, r = compute_ns_r(params)
    if n_s is not None and np.isfinite(n_s) and r is not None and np.isfinite(r):
        ns_kappa.append(n_s)
        r_kappa.append(r)
    else:
        print(f"Skipping kappa = {kappa}: n_s or r invalid")

# Plotting
plt.style.use('seaborn')
plt.figure(figsize=(8, 6))
if ns_mu and r_mu:
    plt.plot(ns_mu, r_mu, 'b-', label=r'Varying $\mu$')
if ns_kappa and r_kappa:
    plt.plot(ns_kappa, r_kappa, 'g-', label=r'Varying $\kappa$')
if n_s_opt is not None and r_opt is not None and np.isfinite(n_s_opt) and np.isfinite(r_opt):
    plt.scatter([n_s_opt], [r_opt], color='red', marker='*', s=150, label='Optimized')
plt.xlabel(r'$n_s$')
plt.ylabel(r'$r$')
plt.title(r'Trajectories in $n_s$-$r$ Plane')
plt.xlim(0.95, 0.98)
plt.ylim(0, 0.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('inflation_plots/ns_r_trajectories.png', dpi=300)
plt.close()

# Save parameters
with open('inflation_plots/parameter_ranges.txt', 'w') as f:
    f.write("Optimized Parameters:\n")
    for key, value in optimized_params.items():
        f.write(f"{key} = {value}\n")
    f.write("\nMu variation range:\n")
    f.write(f"min = {mu_min}, max = {mu_max}\n")
    f.write("\nKappa variation range:\n")
    f.write(f"min = {kappa_min}, max = {kappa_max}\n")

print("Plot and parameter ranges saved in 'inflation_plots' directory.")