"""
/**
 * @author Prabhoda CS
 * @email pcs52@cam.ac.uk
 * @create date 15-03-2025 20:28:34
 * @modify date 15-03-2025 20:28:34
 * @desc Optimized parameter finder and slow-roll plotting for bubble slow-roll inflation.
 */
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import math
from sympy import symbols, diff, ln, lambdify

# -----------------------------
# Global Constants and Targets
# -----------------------------
expns = 0.9626     # Target spectral tilt n_s
expN = 60          # Target e-fold at which to evaluate n_s
Nend = 70          # Integration endpoint for field evolution

# Fixed constants for evolution (second part)
m = 10
M = 20
Mp = math.sqrt(2)*M
sf = 1000000
c = 0

# -----------------------------
# Parameter Optimization Section
# -----------------------------
# Parameter vector: [γ, α, β, k, φ₀, λ, κ]
guess_params = [-100, 50, -10, 100, 30, -10, 10]

def slow_roll_params(tilde, V_arr):
    # Compute derivatives with respect to tilde (here used as our integration variable)
    dV = np.gradient(V_arr, tilde)
    ddV = np.gradient(dV, tilde)
    epsilon = 0.5 * (dV / V_arr)**2
    eta = np.abs(ddV / V_arr)
    return epsilon, eta

def compute_ns(params):
    # Unpack parameters for optimization:
    g, a, b, k, phi0, la, ka = params
    # Check discriminant if needed:
    if g**2 - 4*a*b < 0:
        return 1e6  # Penalty
    
    # Define functions for the potential used in the evolution ODE.
    # Here we use p as our field variable.
    def A(p):
        return a * phi0**2 + g * phi0 * p + b * p**2
    def V(p):
        return ((1 * phi0 * p)**2 + la * p**2 + ka * phi0**4) / (2 * A(p)**2)
    def K(p):
        return M**2 * (-36 * b / A(p) + (phi0**2*(g**2 - 4*b*a)*(k-6))/(2 * A(p)**2))
    
    # Solve the ODE for field evolution using the noncanonical dynamics:
    # The ODE for φ(N) is:
    #    dφ/dN = φ_dot,
    #    d(φ_dot)/dN = (1/K(φ)) * [ -3K(φ) φ_dot + 0.5*(K(φ)^2/Mp^2)*φ_dot^3 - 0.5*(dK/dφ)*φ_dot^2 + (3*Mp^2 - 0.5*K(φ)*φ_dot^2)*(d(ln V)/dφ) ]
    def dK_dp(p):
        # Use central finite differences over a small step:
        h = 1e-6
        return (K(p+h) - K(p-h))/(2*h)
    def dlnV_dp(p):
        h = 1e-6
        return (np.log(V(p+h)) - np.log(V(p-h)))/(2*h)
    def de(N, Y):
        phi, phi_dot = Y
        # Enforce K positive:
        K_val = K(phi)
        if K_val <= 0:
            return [phi_dot, 1e6]  # large penalty derivative
        Kd = dK_dp(phi)
        lv = dlnV_dp(phi)
        phi_ddot = (1/K_val)*(-3*K_val*phi_dot + 0.5*(K_val**2/Mp**2)*phi_dot**3 - 0.5*Kd*phi_dot**2 + (3*Mp**2 - 0.5*K_val*phi_dot**2)*lv)
        return [phi_dot, phi_ddot]
    
    # Solve from N=0 to Nend
    Nr_local = np.linspace(0, Nend, 500)  # Use a coarser grid for speed
    Y0 = [100, 1]
    sol = solve_ivp(de, (0, Nend), Y0, t_eval=Nr_local, method='RK45', rtol=1e-4, atol=1e-6)
    if not sol.success:
        return 1e6  # Penalty if ODE failed
    #hey
    phi_vals = sol.y[0]
    phi_dot_vals = sol.y[1]
    
    # Evaluate V and K along the solution:
    V_vals = np.array([V(p) for p in phi_vals])
    K_vals = np.array([K(p) for p in phi_vals])
    # If any K is nonpositive, penalize:
    if np.any(K_vals <= 1e-6):
        return 1e6
    
    denom = np.maximum(3*Mp**2 - 0.5*K_vals*phi_dot_vals**2, 1e-12)
    H_vals = np.sqrt(np.maximum(V_vals/denom, 0))
    dN = Nr_local[1] - Nr_local[0]
    dH_dN = np.gradient(H_vals, dN)
    epsilon = - dH_dN / H_vals
    dEps_dN = np.gradient(epsilon, dN)
    eta = epsilon - 0.5 * dEps_dN / epsilon
    
    # Find index closest to target_N:
    idx = np.argmin(np.abs(sol.t - expN))
    ns = 1 - 6*epsilon[idx] + 2*eta[idx]
    return (ns - expns)**2

# Optimize parameters (using Nelder-Mead without bounds)
result = minimize(compute_ns, guess_params, method='Nelder-Mead', options={'xatol': 1e-9})
optimal_params = result.x
print("Optimal Parameters:")
print(f"gamma = {optimal_params[0]:.4f}, alpha = {optimal_params[1]:.4f}, beta = {optimal_params[2]:.4f}, k = {optimal_params[3]:.4f}, phi0 = {optimal_params[4]:.4f}, lambda = {optimal_params[5]:.4f}, kappa = {optimal_params[6]:.4f}")

# Unpack optimal parameters for later use in the canonical mapping
gamma = optimal_params[0]
alpha = optimal_params[1]
beta = optimal_params[2]
k = optimal_params[3]
phi0 = optimal_params[4]
la = optimal_params[5]
ka = optimal_params[6]

# -----------------------------
# Canonical Field Mapping and ODE Evolution Section
# -----------------------------
# We'll use sympy definitions for the canonical mapping.
p_s = symbols('phi')

def A_func(phi):
    return beta * phi**2 + gamma * phi0 * phi + alpha * phi0**2

def K(phi):
    A_val = A_func(phi)
    return M**2 * ( -36*beta/A_val + (phi0**2*(gamma**2 - 4*beta*alpha)*(k-6))/(2*A_val**2) )

def V(phi):
    A_val = A_func(phi)
    return (m**2*phi0**2*phi**2 + la*phi**4 + ka*phi0**4)/(2*A_val**2)

# Symbolic definitions for derivative functions:
Kp_sym = diff(K(p_s), p_s)
dlnV_sym = diff(ln(V(p_s)), p_s)
Ki = lambdify(p_s, K(p_s), 'numpy')
dK = lambdify(p_s, Kp_sym, 'numpy')
dlnV = lambdify(p_s, dlnV_sym, 'numpy')

# ODE for canonical field mapping: d(tilde_phi)/d(phi) = sqrt(K(phi))
def dtilde_dphi(phi, tilde):
    return np.sqrt(np.maximum(K(phi), 1e-12))

phi_min = 0
phi_max = 2000
spacing = (phi_max - phi_min) * 30 + 1
phi_span = (phi_min, phi_max)
sol = solve_ivp(dtilde_dphi, phi_span, [c], t_eval=np.linspace(phi_min, phi_max, spacing), method='DOP853', rtol=1e-9, atol=1e-12)
phi_vals = sol.t
tilde_vals = sol.y[0]
print("Canonical field range: from", tilde_vals[0], "to", tilde_vals[-1])

phi_of_tilde = interp1d(tilde_vals, phi_vals, kind='cubic', fill_value="extrapolate")

tilde_start = -750
tilde_end = 1000
tilde_plot = np.linspace(tilde_start, tilde_end, 50000)
phi_plot = phi_of_tilde(tilde_plot)
V_plot = V(phi_plot)

dV_plot = np.gradient(V_plot, tilde_plot)
ddV_plot = np.gradient(dV_plot, tilde_plot)
epsilon = 0.5 * (dV_plot / V_plot)**2
eta = np.abs(ddV_plot / V_plot)

# -----------------------------
# Plotting Section
# -----------------------------
# Plot canonical mapping: φ vs. tilde_φ
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(phi_vals, tilde_vals, label=r"$\tilde{\varphi}(\varphi)$")
ax.set_xlabel(r"Field $\varphi$")
ax.set_ylabel(r"Canonical field $\tilde{\varphi}$")
ax.set_title(r"$\tilde{\varphi}$ vs. $\varphi$")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# Plot potential vs. canonical field:
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tilde_plot, V_plot, label=r"$V(\tilde{\varphi})$")
ax.set_xlabel(r"Canonical field $\tilde{\varphi}$")
ax.set_ylabel(r"$V(\tilde{\varphi})$")
ax.set_title("Potential vs. Canonical Field")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# Plot slow-roll parameters vs. canonical field:
ep_start = -50
ep_end = 50
ep_start = tilde_start
ep_end = tilde_end

mask = (epsilon < 1) & (eta < 1) & (tilde_plot > ep_start) & (tilde_plot < ep_end)
epsilon_masked = np.ma.masked_where(~mask, epsilon)
eta_masked = np.ma.masked_where(~mask, eta)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tilde_plot, epsilon_masked, label=r"$\epsilon$")
ax.plot(tilde_plot, eta_masked, label=r"$\eta$")
ax.set_xlabel(r"Canonical field $\tilde{\varphi}$")
ax.set_ylabel("Slow-roll parameters")
ax.set_title("Slow-roll Parameters vs. Canonical Field")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

"""
# Plot field evolution vs. N with parameters displayed:
Nr2 = np.linspace(0, expN, 1000)  # Use integration time from 0 to expN
sol2 = solve_ivp(de, (0, expN), V0, t_eval=Nr2, method='RK45', rtol=1e-6, atol=1e-8)
ps = np.array(sol2.y[0])
ps_dot = np.array(sol2.y[1])
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(1, 2, width_ratios=[15, 2], wspace=0.01)
ax = fig.add_subplot(gs[0])
ax.plot(Nr2, ps, label=r"$\phi(N)$")
ax.set_title("Field Evolution: $\phi$ vs. N")
ax.set_xlabel("N")
ax.set_ylabel(r"$\phi$")
ax.grid(True)
params_disp = [
    ("Initial field value", p0),
    ("Initial field velocity", p_0),
    (r"$\beta$", beta),
    ("c", c),
    (r"$\mu$", m),
    (r"$\varphi_0$", phi0),
    (r"$\gamma$", gamma),
    (r"$\alpha$", alpha),
    ("D", np.sqrt(g**2 - 4*a*b)),
    ("k", k),
    (r"$\lambda$", la),
    (r"$\kappa$", ka)
]
ax_text = fig.add_subplot(gs[1])
ax_text.axis('off')
text_str = "\n".join([f"{name} = {value:.2f}" for name, value in params_disp])
ax_text.text(0, 0.5, text_str, transform=ax_text.transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='left')
plt.tight_layout()
plt.show()
"""