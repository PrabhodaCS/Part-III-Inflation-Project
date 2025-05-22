import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Constants
MP = 1.0  # Planck mass
N_TARGET = 60.0  # Target number of e-folds
NS_TARGET = 0.9626  # Target spectral index
"""
Best Parameters Found:
alpha = 1.982917
k = 0.000001
mu = -0.888148
omega = 3.313422
chi = 1.888643
kappa = 4.476117

At N = 60.49, n_s = 0.962600
C, mu, omega, chi_p, lambda_p, kappa_p
[ 0.04177624  1.56120225 -7.03409564  3.1513638   1.37424726  9.92633943], cost=1.354e-08
{'C': 0.2629019757834433, 'mu': -9.947646763664885, 'omega': -5.87937042061773, 'chi': 9.19648442654263, 'lambda': -5.889048002757206, 'kappa': -9.69828271467001, 'phi0': -5.010835193933723, 'phip0': 0.5971745790454654}
C, mu, omega, chi_p, lambda_p, kappa_p = params
Best params: {'C': 0.10560221559208352, 'mu': -4.037227348999432, 'omega': -8.776929269476803, 'chi': -7.1522903792017765, 
'lambda': 9.736384698890962, 'kappa': -0.2749459598978127, 'phi0': -12.15720646751496, 'phi_N0': -0.09259248087051818}


C = 0.19286793509396047
lambda = -3.7344900559415866
chi = -9.228047050350941
mu = -1.9040106446508474
omega = -2.581310612248299
kappa = 5.5269587291154645

"""# Parameters
params = {
    'C':  0.19286793509396047,
    'lam': -3.7344900559415866,
    'chi': -9.228047050350941  ,
    'mu': -1.9040106446508474,
    'omega': -2.581310612248299 ,
    'kappa': 5.5269587291154645
}

# Create output directory
if not os.path.exists('inflation_plots2'):
    os.makedirs('inflation_plots2')

# Potential and derivatives
def V(phi):
    theta = params['C'] * phi / MP
    s, c = np.sin(theta), np.cos(theta)
    return (MP**4 / 8) *  (params['lam'] * s**4 
        + 2 * params['chi'] * s**3 * c + 
        params['mu'] * s**2 * c**2 + 
        2 * params['omega'] * s * c**3 + 
        params['kappa'] * c**4
    )

def dV(phi):
    theta = params['C'] * phi / MP
    s, c = np.sin(theta), np.cos(theta)
    dtheta_dphi = params['C'] / MP
    return (MP**4 / 8) * dtheta_dphi * ( params['lam'] *4 * s**3 * c 
         + 2 * params['chi'] * (3 * s**2 * c**2 - s**4) + 
        params['mu'] * (2 * s * c**3 - 2 * s**3 * c) + 
        2 * params['omega'] * (c**4 - 3 * s**2 * c**2) - 
        4 * params['kappa'] * s * c**3
    )

def d2V(phi):
    theta = params['C'] * phi / MP
    s, c = np.sin(theta), np.cos(theta)
    dtheta_dphi = params['C'] / MP
    return (MP**4 / 8) * dtheta_dphi**2 * ( params['lam'] *4 * (3 * s**2 * c**2 - s**4) + 
        2 * params['chi'] * (6 * s * c**3 - 12 * s**3 * c) + 
        params['mu'] * (2 * c**4 - 12 * s**2 * c**2 + 2 * s**4) + 
        2 * params['omega'] * (-4 * c**3 * s + 12 * s**3 * c) - 
        params['kappa'] * (4 * c**4 - 12 * c**2 * s**2)
    )

# Slow-roll parameters
def epsilon_V(phi):
    v = V(phi)
    if abs(v) < 1e-30:  # Adjusted threshold
        return np.inf
    return (MP**2 / 2) * (dV(phi) / v)**2

def eta_V(phi):
    v = V(phi)
    if abs(v) < 1e-30:
        return np.inf
    return MP**2 * (d2V(phi) / v)

# Find phi_end where epsilon_V = 1
def epsilon_minus_one(phi):
    return epsilon_V(phi) - 1.0

phi_min = -np.pi * MP / (2 * params['C']) + 1e-6
phi_max = np.pi * MP / (2 * params['C']) - 1e-6
try:
    phi_end = brentq(epsilon_minus_one, 0, phi_max, rtol=1e-8)
except ValueError:
    try:
        phi_end = brentq(epsilon_minus_one, phi_min, 0, rtol=1e-8)
    except ValueError:
        phi_end = phi_max  # Fallback
print(f"End of slow-roll at phi_end = {phi_end}")

# Field evolution ODE
def d2phi_dN(N, y):
    phi, phi_prime = y
    V_val = V(phi)
    dV_val = dV(phi)
    if abs(V_val) < 1e-30:
        return [phi_prime, 0]
    term1 = (1 / (2 * MP**2)) * phi_prime**3
    term2 = -3 * phi_prime
    term3 = - (dV_val * (3 * MP**2 - 0.5 * phi_prime**2)) / V_val
    phi_double_prime = term1 + term2 + term3
    return [phi_prime, phi_double_prime]

# Initial conditions
phi_initial =   0.0
phidot_initial = 1
y0 = [phi_initial, phidot_initial]
N_span = [0, N_TARGET]

# Solve ODE
sol = solve_ivp(d2phi_dN, N_span, y0, method='RK45', dense_output=True, rtol=1e-8, atol=1e-10, max_step=0.1)
N_vals = np.linspace(0, N_TARGET, 1000)
phi_vals = sol.sol(N_vals)[0]
phi_prime_vals = sol.sol(N_vals)[1]

# Compute observables
eps_vals = np.array([epsilon_V(phi) for phi in phi_vals])
eta_vals = np.array([eta_V(phi) for phi in phi_vals])
n_s_vals = 1 - 6 * eps_vals + 2 * eta_vals
r_vals = 16 * eps_vals

# Parameter text box
param_text = (f"C = {params['C']:.2f}\n"
              f"λ = {params['lam']:.2e}\n"
              f"χ = {params['chi']:.2f}\n"
              f"μ = {params['mu']:.2f}\n"
              f"ω = {params['omega']:.2f}\n"
              f"κ = {params['kappa']:.2f}")
textbox_props = dict(boxstyle='round', facecolor='white', alpha=0.7)

# Plotting
plt.style.use('seaborn')

# 1. Canonical Potential
phi_plot = np.linspace(phi_min, phi_max, 1000)
V_plot = np.array([V(phi) for phi in phi_plot])
plt.figure(figsize=(8, 6))
plt.plot(phi_plot, V_plot, 'b-')
plt.title('Canonical Potential')
plt.xlabel(r'$\tilde{\varphi}$')
plt.ylabel(r'$V(\tilde{\varphi})$')
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.show()
plt.savefig('inflation_plots2/potential.png', dpi=300)
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
plt.show()
plt.savefig('inflation_plots2/phi_vs_N.png', dpi=300)
plt.close()

# 3. Slow-roll parameters vs N
plt.figure(figsize=(8, 6))
finite_mask_eps = np.isfinite(eps_vals)
plt.plot(N_vals[finite_mask_eps], eps_vals[finite_mask_eps], 'b-', label=r'$\epsilon_V$')
finite_mask_eta = np.isfinite(eta_vals)
plt.plot(N_vals[finite_mask_eta], np.abs(eta_vals[finite_mask_eta]), 'r--', label=r'$|\eta_V|$')
plt.title('Slow-Roll Parameters')
plt.xlabel('N (e-folds)')
plt.ylabel('Slow-Roll Parameters')
plt.legend()
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.show()
plt.savefig('inflation_plots2/slow_roll.png', dpi=300)
plt.close()

# 4. Spectral index vs N
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
plt.show()
plt.savefig('inflation_plots2/n_s.png', dpi=300)
plt.close()

# 5. Tensor-to-scalar ratio vs N
plt.figure(figsize=(8, 6))
finite_mask_r = np.isfinite(r_vals)
plt.plot(N_vals[finite_mask_r], r_vals[finite_mask_r], 'b-')
plt.title('Tensor-to-Scalar Ratio')
plt.xlabel('N (e-folds)')
plt.ylabel(r'$r$')
plt.grid(True)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8, 
         verticalalignment='top', ha='left', bbox=textbox_props)
plt.show()
plt.savefig('inflation_plots2/r.png', dpi=300)
plt.close()

# 6. Phase-space portraits
def phase_rhs(t, y):
    phi, dot_phi = y
    V_val = V(phi)
    H = np.sqrt((0.5 * dot_phi**2 + V_val) / (3 * MP**2))
    ddot_phi = -3 * H * dot_phi - dV(phi)
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
        derivs = phase_rhs(0, [PHI[i,j], DOT_PHI[i,j]])
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
plt.show()
plt.savefig('inflation_plots2/phase_portrait.png', dpi=300)
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
plt.show()
plt.savefig('inflation_plots2/phase_portrait_compact.png', dpi=300)
plt.close()

# 7. n_s vs r
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
plt.show()
plt.savefig('inflation_plots2/n_s_vs_r.png', dpi=300)
plt.close()

print("All plots saved in 'inflation_plots2' directory.")