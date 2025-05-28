from token import TILDE
import numpy as np
import matplotlib.pyplot as plt

# --- Model parameters (example values; replace with yours) ---
M_p   = 1.0
M     = np.sqrt(2) * M_p
alpha = 2
beta  = 6.113
gamma = 0
phi0  = 10
k     = 34.806

kappa = 1.2
omega = 5.0
mu2    = 2.768
chi   = 0
lam   = 3.947

"""alpha = 2
beta  = 6.113
gamma = 0
phi0  = 10
k     = 34.806

kappa = 1.2
omega = 5.0
mu2    = 2.768
chi   = 0
lam   = 3.947
"""
"""
alpha = 2.0
gamma = 0.0
chi = 0
beta=9.968, k=10.000, kappa=1.200, omega=5.000, mu2=1.000, lambda=2.539



alpha = 5.37904158 
beta  = 4.84572706 
gamma = -1.2378985
phi0  = 10
k     = 46.47655135  

kappa = 42.92091333
omega = -7.74347196
mu2    = 7.12605145 
chi   = 9.67061868
lam   = 0.31666247 """
"""
[ 5.37904158 -1.2378985   4.84572706 46.47655135  7.12605145 -7.74347196
  9.67061868  0.31666247 42.92091333]
[alpha, gamma, beta, k, mu, omega, chi, lambda_val, kappa]
"""
# --- Define K(phi) and V(phi) ---------------------------------------
def K(phi):
    num = M**2 * (12 + k) * (4*alpha*beta - gamma**2) * phi0**2
    den = 2 * (beta*phi**2 + gamma*phi0*phi + alpha*phi0**2)
    return num / den

def V(phi):
    A = beta*phi**2 + gamma*phi + alpha
    num = (kappa
         + 2*omega*phi
         + mu2*phi**2
         + 2*chi*phi**3
         + lam*phi**4)
    return M**4 * num / (2 * A**2)

# --- Canonical mapping and its inverse -----------------------------
# ˜φ(φ): analytic arctan form
prefac = np.sqrt((12 + k)/2) * M_p
D = np.sqrt(4*alpha*beta - gamma**2)
"""
def phi_tilde(phi):
    return prefac * np.arctan((2*beta*phi + gamma*phi0)/(phi0*D))
"""
# inverse: φ(˜φ)
def phi_from_tilde(phit):
    return (1/(2*beta)) * (D * np.tan(phit / prefac) - gamma)

print("Size of graph: ",prefac*np.pi/2)

# --- Gridding -------------------------------------------------------
"""phi_vals   = np.linspace(-10000, 10000, 100000)
tilde_vals = phi_tilde(phi_vals)
"""

# for canonical potential
phit_max = prefac*np.pi/2
extra = phit_max/11
tilde_grid = np.linspace(-phit_max - extra, phit_max  + extra, 5000)

#tilde_grid = np.linspace(-15, 15, 100000)
phi_inv    = phi_from_tilde(tilde_grid)

def d1(phi):
    phi_inv    = phi_from_tilde(phi)
    return np.gradient(V(phi_inv), phi)
epsilon = (M_p**2/2) * (d1(tilde_grid)/V(phi_inv))**2
"""
# --- Plotting -------------------------------------------------------
plt.figure(figsize=(12, 9))

# 1) Kinetic coefficient K(phi)
ax1 = plt.subplot(2, 2, 1)
ax1.plot(phi_vals, K(phi_vals), 'C0', lw=2)
ax1.set_title(r'$K(\varphi)$')
ax1.set_xlabel(r'$\varphi$')
ax1.set_ylabel(r'$K$')
ax1.grid(True)

# 2) Noncanonical potential V(phi)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(phi_vals, V(phi_vals), 'C1', lw=2)
ax2.set_title(r'$V(\varphi)$')
ax2.set_xlabel(r'$\varphi$')
ax2.set_ylabel(r'$V$')
ax2.grid(True)

# 3) Canonical mapping φ → ˜φ
ax3 = plt.subplot(2, 2, 3)
ax3.plot(phi_vals, tilde_vals, 'C2', lw=2)
ax3.set_title(r'Canonical map $\tilde\varphi(\varphi)$')
ax3.set_xlabel(r'$\varphi$')
ax3.set_ylabel(r'$\tilde\varphi$')
ax3.grid(True)
"""

# 4) Canonical potential Ṽ(˜φ) = V(φ(˜φ))
#ax4 = plt.subplot(2, 2, 4)
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(tilde_grid, V(phi_inv))
plt.title(r'Canonical potential $V(\varphi(\tilde\varphi))$')
plt.xlabel(r'$\tilde\varphi$')
plt.ylabel(r'$V$')
plt.grid(True)

plt.tight_layout()
plt.show()

plt.plot(tilde_grid, d1(tilde_grid))
plt.title(r'$\epsilon$')    
plt.xlabel(r'$\tilde\varphi$')
plt.ylabel(r'$\epsilon$')
plt.grid(True)

plt.tight_layout()
plt.show()



plt.plot(tilde_grid, epsilon)
plt.title(r'$\epsilon$')    
plt.xlabel(r'$\tilde\varphi$')
plt.ylabel(r'$\epsilon$')
plt.grid(True)

plt.tight_layout()
plt.show()

"""M_p   = 1.0
M     = np.sqrt(2) * M_p
alpha = 5.0
beta  = 10
gamma = 14.1
phi0  = 10
k     = 1

kappa = -1
omega = -1
mu2    = -10
chi   = -2
lam   = 15
"""
