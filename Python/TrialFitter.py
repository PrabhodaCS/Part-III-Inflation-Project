import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Fixed constants ---
M_p = 1.0
phi0 = 10.0
alpha = 2.0
gamma = 0.0
chi = 0.0  # fixed

# --- Initial guess for tunable params ---
beta0, k0, kappa0, omega0, mu2_0, lambda0 = 6.0, 35.0, 0.7, 1.0, 3.0, 4.0

# --- Mapping constants ---
def prefac(k): return np.sqrt((12 + k) / 2) * M_p
def D(alpha, beta, gamma): return np.sqrt(4*alpha*beta - gamma**2)

def phi_from_tilde(phit, k, alpha, beta, gamma):
    return (phi0/(2*beta)) * (D(alpha,beta,gamma) * np.tan(phit/prefac(k)) - gamma)

# --- Potential in tilde-space ---
def V_tilde(phit, params):
    beta, k, kappa, omega, mu2, lam = params
    phi = phi_from_tilde(phit, k, beta, alpha, gamma)
    A = beta*phi**2 + gamma*phi0*phi + alpha*phi0**2
    num = (kappa*phi0**4 + 2*omega*phi0**3*phi + mu2*phi0**2*phi**2 + 2*chi*phi0*phi**3 + lam*phi**4)
    return (np.sqrt(2)*M_p)**4 * num / (2 * A**2)

# --- Derivatives via finite differences ---
def compute_derivs(tgrid, Vgrid):
    d1 = np.gradient(Vgrid, tgrid)
    d2 = np.gradient(d1, tgrid)
    return d1, d2

# --- Objective: combine metrics ---
def objective(params):
    # generate grid
    k = params[1]
    phit_max = prefac(k)*np.pi/2
    tgrid = np.linspace(-phit_max, phit_max, 5000)
    Vgrid = V_tilde(tgrid, params)
    d1, d2 = compute_derivs(tgrid, Vgrid)
    
    # locate peak (max V) and valley (min V)
    imax = np.nanargmax(Vgrid)
    imin = np.nanargmin(Vgrid)
    # metric1: want large negative second derivative at peak
    metric1 = -d2[imax]
    # metric2: want large positive second derivative at valley
    metric2 = d2[imin]
    # metric3: want small avg |d1| in edge regions (outer 20%)
    ne = len(tgrid)
    edge = int(0.5*ne)
    avg_edge = np.mean(np.abs(np.concatenate([d1[:edge], d1[-edge:]])))
    # combine: negative metric1, negative metric2, positive small edge derivative
    # We want to maximize metric1 and metric2, minimize avg_edge -> minimize cost
    cost = -metric1 - metric2 + 10*avg_edge
    return cost

# --- Run optimization ---
bounds = [(2,10),     # beta
          (10,50),    # k
          (0.2,1.2),  # kappa
          (0.1,5),    # omega
          (1,6),      # mu2
          (2,8)]      # lam

res = minimize(objective,
               x0=[beta0,k0,kappa0,omega0,mu2_0,lambda0],
               bounds=bounds,
               method='L-BFGS-B')

beta_opt, k_opt, kappa_opt, omega_opt, mu2_opt, lambda_opt = res.x
print("Optimized parameters:")
print(f"beta={beta_opt:.3f}, k={k_opt:.3f}, kappa={kappa_opt:.3f}, omega={omega_opt:.3f}, mu2={mu2_opt:.3f}, lambda={lambda_opt:.3f}")

# --- Plot results ---
phit_max = prefac(k_opt)*np.pi/2
tgrid = np.linspace(-phit_max, phit_max, 5000)
Vopt = V_tilde(tgrid, res.x)
d1opt, d2opt = compute_derivs(tgrid, Vopt)

fig, ax = plt.subplots(2,1,figsize=(8,6))
ax[0].plot(tgrid, Vopt); ax[0].set_title('Tuned V(tilde)')
ax[1].plot(tgrid, d2opt); ax[1].set_title('Second derivative')
plt.tight_layout()
plt.show()
