import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
import optuna
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Constants
MP = 1.0  # Planck mass
N_TARGET = 60.0  # Target number of e-folds
NS_TARGET = 0.9626  # Target spectral index

# Create output directory
output_dir = 'inflation_outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Potential and derivatives
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

def V_prime(phi, C, lambda_, chi, mu, omega, kappa):
    theta = C * phi
    term1 = (-chi + omega) * np.cos(4*theta)
    term2 = (-kappa + lambda_) * np.sin(2*theta)
    term3 = np.cos(2*theta) * (chi + omega - (kappa + lambda_ - mu) * np.sin(2*theta))
    return (C / 8) * (term1 + term2 + term3)

def V_double_prime(phi, C, lambda_, chi, mu, omega, kappa):
    theta = C * phi
    term1 = (kappa - lambda_) * np.cos(2*theta)
    term2 = (kappa + lambda_ - mu) * np.cos(4*theta)
    term3 = (chi + omega) * np.sin(2*theta)
    term4 = 2 * (-chi + omega) * np.sin(4*theta)
    return - (C**2 / 4) * (term1 + term2 + term3 + term4)

# Slow-roll parameters
def epsilon_V(phi, params):
    v = V(phi, *params)
    if abs(v) < 1e-30:
        return np.inf
    vp = V_prime(phi, *params)
    return 0.5 * (vp / v)**2

def eta_V(phi, params):
    v = V(phi, *params)
    if abs(v) < 1e-30:
        return np.inf
    vpp = V_double_prime(phi, *params)
    return vpp / v

# Optimizer objective function
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
    
    # Define field range
    phi_min = -np.pi / (2 * C) + 1e-6
    phi_max = np.pi / (2 * C) - 1e-6
    
    # Find phi_end where epsilon = 1
    def g(phi):
        v = V(phi, *params)
        if v <= 0:
            return 1e10
        vp = V_prime(phi, *params)
        return 0.5 * (vp / v)**2 - 1
    
    try:
        phi_end = brentq(g, 0.001, phi_max, rtol=1e-8)
        direction = 1
    except ValueError:
        try:
            phi_end = brentq(g, phi_min, -0.001, rtol=1e-8)
            direction = -1
        except ValueError:
            return 1e10
    
    # Determine rolling direction
    V_prime_phi_end = V_prime(phi_end, *params)
    if V_prime_phi_end > 0:
        a = phi_end + 1e-6
        b = phi_max
    else:
        a = phi_min
        b = phi_end - 1e-6
    
    # Define e-foldings
    def integrand(p):
        v = V(p, *params)
        vp = V_prime(p, *params)
        return -v / vp if vp != 0 else 0
    
    def N_func(phi):
        N, _ = quad(integrand, phi, phi_end)
        return N
    
    # Check if sufficient e-foldings
    N_b = N_func(b)
    if N_b < N_TARGET:
        return 1e10
    
    # Find phi_60
    def h(phi):
        return N_func(phi) - N_TARGET
    
    try:
        if V_prime_phi_end > 0:
            phi_60 = brentq(h, a, b, rtol=1e-8)
        else:
            phi_60 = brentq(h, b, a, rtol=1e-8)
    except ValueError:
        return 1e10
    
    # Compute n_s
    v = V(phi_60, *params)
    vp = V_prime(phi_60, *params)
    vpp = V_double_prime(phi_60, *params)
    epsilon = 0.5 * (vp / v)**2
    eta = vpp / v
    n_s = 1 - 6 * epsilon + 2 * eta
    cost = (n_s - NS_TARGET)**2
    
    # Store phi_60 for plotting
    trial.set_user_attr('phi_60', phi_60)
    
    # Print progress
    print(f"Trial {trial.number}: cost={cost:.2e}, params={params}, n_s={n_s:.4f}, phi_60={phi_60:.4f}")
    return cost

# Run optimization
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

# Get best parameters and phi_60
best_params = study.best_params
phi_60 = study.best_trial.user_attrs['phi_60']
params = (best_params['C'], best_params['lambda'], best_params['chi'], 
          best_params['mu'], best_params['omega'], best_params['kappa'])

# Save parameters
with open(os.path.join(output_dir, 'optimal_params.txt'), 'w') as f:
    for key, value in best_params.items():
        f.write(f"{key}: {value}\n")
    f.write(f"phi_60: {phi_60}\n")

with open(os.path.join(output_dir, 'optimal_params.tex'), 'w') as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{amsmath}\n")
    f.write("\\begin{document}\n")
    f.write("\\begin{align*}\n")
    for key, value in best_params.items():
        f.write(f"{key} &= {value} \\\\\n")
    f.write(f"\\phi_{{60}} &= {phi_60} \\\\\n")
    f.write("\\end{align*}\n")
    f.write("\\end{document}\n")

# Plotter setup
phi_min = -np.pi / (2 * best_params['C']) + 1e-6
phi_max = np.pi / (2 * best_params['C']) - 1e-6

# Find phi_end
def epsilon_minus_one(phi):
    return epsilon_V(phi, params) - 1.0

try:
    phi_end = brentq(epsilon_minus_one, 0, phi_max, rtol=1e-8)
except ValueError:
    try:
        phi_end = brentq(epsilon_minus_one, phi_min, 0, rtol=1e-8)
    except ValueError:
        phi_end = phi_max
print(f"End of slow-roll at phi_end = {phi_end}")

# Initial conditions
phi_initial = phi_60
V_val = V(phi_60, *params)
dV_val = V_prime(phi_60, *params)
dot_phi_initial = -dV_val / np.sqrt(3 * V_val) if V_val > 0 else 0
y0 = [phi_initial, dot_phi_initial]
print(f"Initial conditions: phi_60 = {phi_60}, dot_phi = {dot_phi_initial}")

# ODE for field evolution
def d2phi_dN(N, y):
    phi, phi_prime = y
    V_val = V(phi, *params)
    dV_val = V_prime(phi, *params)
    if abs(V_val) < 1e-30:
        return [phi_prime, 0]
    term1 = (1 / (2 * MP**2)) * phi_prime**3
    term2 = -3 * phi_prime
    term3 = - (dV_val * (3 * MP**2 - 0.5 * phi_prime**2)) / V_val
    return [phi_prime, term1 + term2 + term3]

# Events for ODE solver
def event_H2_negative(N, y):
    phi, dot_phi = y
    V_val = V(phi, *params)
    return (0.5 * dot_phi**2 + V_val) / (3 * MP**2)
event_H2_negative.terminal = True
event_H2_negative.direction = -1

def event_phi_min(N, y):
    return y[0] - phi_min
event_phi_min.terminal = True
event_phi_min.direction = -1

def event_phi_max(N, y):
    return y[0] - phi_max
event_phi_max.terminal = True
event_phi_max.direction = 1

# Solve ODE
sol = solve_ivp(d2phi_dN, [0, N_TARGET], y0, method='RK45', dense_output=True, 
                rtol=1e-8, atol=1e-10, max_step=0.1, 
                events=[event_H2_negative, event_phi_min, event_phi_max])
N_vals = np.linspace(0, min(sol.t[-1], N_TARGET), 1000)
phi_vals = sol.sol(N_vals)[0]
phi_prime_vals = sol.sol(N_vals)[1]

# Compute observables
eps_vals = np.array([epsilon_V(phi, params) for phi in phi_vals])
eta_vals = np.array([eta_V(phi, params) for phi in phi_vals])
n_s_vals = 1 - 6 * eps_vals + 2 * eta_vals
r_vals = 16 * eps_vals

# Parameter text box
param_text = (f"C = {best_params['C']:.2f}\n"
              f"λ = {best_params['lambda']:.2e}\n"
              f"χ = {best_params['chi']:.2f}\n"
              f"μ = {best_params['mu']:.2f}\n"
              f"ω = {best_params['omega']:.2f}\n"
              f"κ = {best_params['kappa']:.2f}\n"
              f"φ_60 = {phi_60:.2f}")
textbox_props = dict(boxstyle='round', facecolor='white', alpha=0.7)

# Plotting
plt.style.use('seaborn')

# 1. Canonical Potential
phi_plot = np.linspace(phi_min, phi_max, 1000)
V_plot = np.array([V(phi, *params) for phi in phi_plot])
plt.figure(figsize=(8, 6))
plt.plot(phi_plot, V_plot, 'b-')
plt.title('Canonical Potential')
plt.xlabel(r'$\tilde{\varphi}$')
plt.ylabel(r'$V(\tilde{\varphi})$')
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.savefig(os.path.join(output_dir, 'potential.png'), dpi=300)
plt.close()

# 2. phi vs N
plt.figure(figsize=(8, 6))
plt.plot(N_vals, phi_vals, 'b-')
plt.title('Field Evolution')
plt.xlabel('N (e-folds)')
plt.ylabel(r'$\tilde{\varphi}$')
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.savefig(os.path.join(output_dir, 'phi_vs_N.png'), dpi=300)
plt.close()

# 3. epsilon and eta vs phi
plt.figure(figsize=(8, 6))
finite_mask_eps = np.isfinite(eps_vals)
plt.plot(phi_vals[finite_mask_eps], eps_vals[finite_mask_eps], 'b-', label=r'$\epsilon_V$')
finite_mask_eta = np.isfinite(eta_vals)
plt.plot(phi_vals[finite_mask_eta], np.abs(eta_vals[finite_mask_eta]), 'r--', label=r'$|\eta_V|$')
plt.title('Slow-Roll Parameters vs Field')
plt.xlabel(r'$\tilde{\varphi}$')
plt.ylabel('Slow-Roll Parameters')
plt.legend()
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.savefig(os.path.join(output_dir, 'slow_roll_vs_phi.png'), dpi=300)
plt.close()

# 4. epsilon and eta vs N
plt.figure(figsize=(8, 6))
plt.plot(N_vals[finite_mask_eps], eps_vals[finite_mask_eps], 'b-', label=r'$\epsilon_V$')
plt.plot(N_vals[finite_mask_eta], np.abs(eta_vals[finite_mask_eta]), 'r--', label=r'$|\eta_V|$')
plt.title('Slow-Roll Parameters vs N')
plt.xlabel('N (e-folds)')
plt.ylabel('Slow-Roll Parameters')
plt.legend()
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.savefig(os.path.join(output_dir, 'slow_roll_vs_N.png'), dpi=300)
plt.close()

# 5. Spectral index vs N
plt.figure(figsize=(8, 6))
finite_mask_n_s = np.isfinite(n_s_vals)
plt.plot(N_vals[finite_mask_n_s], n_s_vals[finite_mask_n_s], 'g-')
plt.axhline(NS_TARGET, color='k', linestyle='--', label=f'Target $n_s$ = {NS_TARGET}')
plt.title('Spectral Index')
plt.xlabel('N (e-folds)')
plt.ylabel(r'$n_s$')
plt.legend()
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.savefig(os.path.join(output_dir, 'n_s.png'), dpi=300)
plt.close()

# 6. Tensor-to-scalar ratio vs N
plt.figure(figsize=(8, 6))
finite_mask_r = np.isfinite(r_vals)
plt.plot(N_vals[finite_mask_r], r_vals[finite_mask_r], 'b-')
plt.title('Tensor-to-Scalar Ratio')
plt.xlabel('N (e-folds)')
plt.ylabel(r'$r$')
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.savefig(os.path.join(output_dir, 'r.png'), dpi=300)
plt.close()

# 7. Phase-space portraits
def phase_rhs(t, y):
    phi, dot_phi = y
    V_val = V(phi, *params)
    H = np.sqrt((0.5 * dot_phi**2 + V_val) / (3 * MP**2)) if (0.5 * dot_phi**2 + V_val) >= 0 else 0
    ddot_phi = -3 * H * dot_phi - V_prime(phi, *params)
    return [dot_phi, ddot_phi]

# Uncompactified
phi_range = np.linspace(phi_min, phi_max, 100)
dot_phi_max = 2 * MP
dot_phi_range = np.linspace(-dot_phi_max, dot_phi_max, 100)
PHI, DOT_PHI = np.meshgrid(phi_range, dot_phi_range)
U = np.zeros_like(PHI)
V_phase = np.zeros_like(PHI)
for i in range(PHI.shape[0]):
    for j in range(PHI.shape[1]):
        phi = PHI[i,j]
        if phi < phi_min or phi > phi_max or V(phi, *params) <= 0:
            U[i,j] = np.nan
            V_phase[i,j] = np.nan
        else:
            derivs = phase_rhs(0, [phi, DOT_PHI[i,j]])
            U[i,j] = derivs[0]
            V_phase[i,j] = derivs[1]
speed = np.sqrt(U**2 + V_phase**2)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(PHI, DOT_PHI, np.log(speed + 1e-12), cmap='viridis', levels=50, alpha=0.8)
ax.streamplot(PHI, DOT_PHI, U, V_phase, color='black', density=1.5, linewidth=0.7, arrowsize=1)
ax.set_xlabel(r'$\tilde{\varphi}$')
ax.set_ylabel(r'$\dot{\tilde{\varphi}}$')
ax.set_title('Uncompactified Phase Portrait')
ax.grid(True)
plt.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(contour, cax=cax)
cbar.set_label(r"Log Flow Speed")
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'phase_portrait.png'), dpi=300)
plt.close()

# Compactified
def compact(x):
    return np.tanh(x)
def inv_compact(u):
    return np.arctanh(u)
uc = np.linspace(-0.99, 0.99, 40)
vc = np.linspace(-0.99, 0.99, 40)
UC, VC = np.meshgrid(uc, vc)
Uc = np.zeros_like(UC)
Vc = np.zeros_like(VC)
for i in range(UC.shape[0]):
    for j in range(UC.shape[1]):
        phi = inv_compact(UC[i,j])
        dot_phi = inv_compact(VC[i,j])
        if phi < phi_min or phi > phi_max or V(phi, *params) <= 0:
            Uc[i,j] = np.nan
            Vc[i,j] = np.nan
        else:
            derivs = phase_rhs(0, [phi, dot_phi])
            Uc[i,j] = (1 - UC[i,j]**2) * derivs[0]
            Vc[i,j] = (1 - VC[i,j]**2) * derivs[1]
speed_c = np.sqrt(Uc**2 + Vc**2)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(UC, VC, np.log(speed_c + 1e-12), cmap='viridis', levels=50, alpha=0.8)
ax.streamplot(UC, VC, Uc, Vc, color='black', density=1.5, linewidth=0.7, arrowsize=1)
ax.set_xlabel(r'$\tanh(\tilde{\varphi})$')
ax.set_ylabel(r'$\tanh(\dot{\tilde{\varphi}})$')
ax.set_title('Compactified Phase Portrait')
ax.grid(True)
plt.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(contour, cax=cax)
cbar.set_label(r"Log Flow Speed")
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'phase_portrait_compact.png'), dpi=300)
plt.close()

# 8. n_s vs r
plt.figure(figsize=(8, 6))
finite_mask_both = np.isfinite(n_s_vals) & np.isfinite(r_vals)
plt.plot(r_vals[finite_mask_both], n_s_vals[finite_mask_both], 'b-')
plt.scatter(r_vals[finite_mask_both][-1], n_s_vals[finite_mask_both][-1], color='r', label=f'N={N_TARGET}')
plt.title(r'$n_s$ vs $r$')
plt.xlabel(r'$r$')
plt.ylabel(r'$n_s$')
plt.legend()
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.savefig(os.path.join(output_dir, 'n_s_vs_r.png'), dpi=300)
plt.close()

print(f"All plots saved in '{output_dir}' directory.")