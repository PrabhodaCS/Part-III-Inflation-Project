"""
/**
 * @author Prabhoda CS
 * @email pcs52@cam.ac.uk
 * @create date 15-03-2025 20:28:34
 * @modify date 15-03-2025 20:28:34
 * @desc [description]
 */
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from scipy.interpolate import interp1d
from sympy import *

expns = 0.9626
expN = 60

# Function to compute slow-roll parameters
def slow_roll_params(tilde_plot, V_plot):
    dV = np.gradient(V_plot, tilde_plot)
    ddV = np.gradient(dV, tilde_plot)
    
    epsilon = 0.5 * (dV / V_plot) ** 2
    eta = np.abs(ddV / V_plot)
    
    return epsilon, eta

# Function to compute n_s at N = 60
def compute_ns(params):
    g, a, b, k, phi0 = params
    
    # Derived parameters
    D = np.sqrt(g**2 - 4*a*b)
    m = 1
    sf = 1000000
    M = 20
    Mp = np.sqrt(2) * M
    
    # Initial conditions
    p0 = 100
    p_0 = 0.1
    V0 = [p0, p_0]
    Nend = 70
    Nr = np.linspace(0, Nend, 1000)
    
    # Define potential
    def potential(p):
        A = a * phi0**2 + g * phi0 * p + b * p**2
        return (m * phi0) ** 2 * (p / A) ** 2 / 2
    
    # Compute potential
    tilde_plot = np.linspace(-200, 200, 1000)
    V_plot = potential(tilde_plot)
    
    # Compute slow-roll parameters
    epsilon, eta = slow_roll_params(tilde_plot, V_plot)
    
    # Find index closest to N = 60
    idx_expN = np.argmin(np.abs(Nr - expN))
    
    # Compute spectral tilt
    ns = 1 + 2 * eta[idx_expN] - 6 * epsilon[idx_expN]
    return (ns - expns) ** 2  # Minimize squared difference from 0.97

# Initial guess for parameters
guess_params = [100, -50, 0.1, 1, 30]

# Optimize
result = minimize(compute_ns, guess_params, method='Nelder-Mead', options={'xatol': 1e-9})
optimal_params = result.x

print("Optimal Parameters:")
print(f"gamma = {optimal_params[0]:.4f}, alpha = {optimal_params[1]:.4f}, beta = {optimal_params[2]:.4f}, k = {optimal_params[3]:.4f}, phi0 = {optimal_params[4]:.4f}")

beta = optimal_params[2]
gamma = optimal_params[0]
alpha = optimal_params[1]
k = optimal_params[3]
phi0 = optimal_params[4]

# Fixed parameters
D = np.sqrt(gamma**2 - 4*alpha*beta)
m = 10
M = 20
Mp = np.sqrt(2) * M
c = 0

p_s = symbols('phi')

# Define A(phi) = beta*phi^2 + gamma*phi0*phi + alpha*phi0^2.
def A_func(phi):
    return beta * phi**2 + gamma * phi0 * phi + alpha * phi0**2

# Define K(phi) as given
def K(phi):
    A_val = A_func(phi)
    return M**2 * ( -36*beta/A_val + (phi0**2*(gamma**2 - 4*beta*alpha)*(k-6))/(2*A_val**2) )

# Define the ODE for the canonical field: dtilde_phi/dphi = sqrt(K(phi))
def dtilde_dphi(phi, tilde):
    return np.sqrt(K(phi))

# Choose a range for phi over which you want to integrate:
phi_min = 0
phi_max = 500

# Solve the ODE numerically to get tilde_phi as a function of phi
phi_span = (phi_min, phi_max)
# We'll solve the ODE using solve_ivp. We use phi as the independent variable.
sol = solve_ivp(dtilde_dphi, phi_span, [c], t_eval=np.linspace(phi_min, phi_max, 1001), method='DOP853', rtol=1e-9, atol=1e-12)

phi_vals = sol.t         # these are the phi values
tilde_vals = sol.y[0]     # corresponding canonical field values

print("beginning value of canonical phi : ",tilde_vals[0],"\n","End value  of canonical phi : ", tilde_vals[-1])

def V(phi):
    return  0.5 * (m*phi0)**2 * (phi**2)/A_func(phi)**2

# If you want to plot V as a function of the canonical field, you need to know phi as a function of tilde.
# We can build an interpolation function from tilde_vals vs. phi_vals.
phi_of_tilde = interp1d(tilde_vals, phi_vals, kind='cubic', fill_value="extrapolate")

# Specify the range for the canonical field you want to plot:
tilde_start = -500 # lower limit for canonical field
tilde_end   = 80  # upper limit for canonical field

# Generate canonical field values in that range:
tilde_plot = np.linspace(tilde_start, tilde_end, 50000)
phi_plot = phi_of_tilde(tilde_plot)
V_plot   = V(phi_plot)

dV = np.gradient(V_plot, tilde_plot)
ddV = np.gradient(dV, tilde_plot)

ratio = max(dV)/max(V_plot)
ratio2 = max(ddV)/max(V_plot)

epsilon = 0.5 * (dV / V_plot) ** 2
eta = np.abs(ddV / V_plot)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(phi_vals, tilde_vals)
ax.set_xlabel(r"Field $\varphi$")
ax.set_ylabel(r"Canonical field $\tilde{\varphi}$")
ax.set_title(r"$\tilde{\varphi}$ vs. $\varphi$")
ax.legend()
ax.grid(True)
plt.tight_layout()

# Parameters to display
parameters = [
    (r'$\beta$', beta),
    (r"$c$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', gamma),
    (r'$\alpha$', alpha),
    (r'$D$', D),
    (r'$k$', k)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

plt.show()


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tilde_plot, V_plot, label=r"$V(\tilde{\varphi})$")
"""
ax.plot(tilde_plot, dV/ratio, label=r"$dV/d\tilde{\varphi}$  (Scaled)")
ax.plot(tilde_plot, ddV/ratio2, label=r"$d^2V/d\tilde{\varphi}^2$  (Scaled)")"""
ax.set_xlabel(r"Canonical field $\tilde{\varphi}$")
ax.set_ylabel(r"$V(\tilde{\varphi})$")
ax.set_title("Potential vs. Canonical Field")
ax.grid(True)
plt.tight_layout()

# Parameters to display
parameters = [
    (r'$\beta$', beta),
    (r"$c$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', gamma),
    (r'$\alpha$', alpha),
    (r'$D$', D),
    (r'$k$', k)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')


plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\Potential with correct parameters.png")
plt.show()


ep_start = -50
ep_end = 50

mask = (epsilon < 1) & (eta < 1) & (tilde_plot > ep_start) & (tilde_plot < ep_end)
epsilon_masked = np.ma.masked_where(~mask, epsilon)
eta_masked = np.ma.masked_where(~mask, eta)

# --- Plot Slow-Roll Parameters vs Canonical Field ---
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tilde_plot, epsilon_masked, label=r"$\epsilon$")
ax.plot(tilde_plot, eta_masked, label=r"$\eta$")
ax.set_xlabel(r"Canonical field $\tilde{\varphi}$")
ax.set_ylabel("Slow-roll parameters")
ax.set_title("Slow-roll Parameters vs. Canonical Field")
ax.legend()
ax.grid(True)

# Parameters to display
parameters = [
    (r'$\beta$', beta),
    (r"$c$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', gamma),
    (r'$\alpha$', alpha),
    (r'$D$', D),
    (r'$k$', k)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\Full Slow Roll Parameters.png")
plt.show()