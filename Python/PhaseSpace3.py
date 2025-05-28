
import numpy as np
import matplotlib.pyplot as plt
theta = np.tanh
# ---------------------------- PARAMETERS ----------------------------
M_p  = 1.0    # Planck mass
m    = 1.0    # inflaton mass
phi0 = 1.5    # non‑propagating field
a, b, d = 50.0, 5.0, 1.5  # model parameters
gamma = 10.0  # choose a representative value
li, ka, la, om = 1.0, 0.01, 1.0, 1.0  # potential coefficients
k    = 1

# ---------------------------- GRID ----------------------------
phi_min, phi_max = -3.0, 3.0
y_min, y_max     = -5.0, 5.0
ngrid = 500
phi_vals = np.linspace(phi_min, phi_max, ngrid)
y_vals   = np.linspace(y_min, y_max, ngrid)
PHI, Y   = np.meshgrid(phi_vals, y_vals)

# ------------------------ MODEL FUNCTIONS ------------------------
def X_of_phi(phi, gamma):
    D = np.sqrt(4*a * b - (gamma**2) )
    return (phi0 / (2 * b)) * (-gamma + D * np.tan(phi / (np.sqrt(k) * M_p)))

def V_phi(phi, gamma):
    Xp = X_of_phi(phi, gamma)
    denom = b * Xp**2 + gamma * phi0 * Xp + a * phi0**2
    num = ka * phi0**4 + li * phi0**3 * Xp + m**2 * Xp**2 + om * phi0 * Xp**3 + la * Xp**4
    return -0.5 * M_p**4 * (num / denom**2)

def dV_dphi(phi, gamma):
    V = V_phi(phi, gamma)
    return np.gradient(V, phi_vals)

# ------------------------ VECTOR FIELD ---------------------------
dV = dV_dphi(phi_vals, gamma)
# interpolate dV/dphi on mesh
interp_dV = np.interp(PHI.flatten(), phi_vals, dV).reshape(PHI.shape)
# Hubble parameter
H = np.sqrt((Y**2 / 2 + V_phi(PHI, gamma)) / 3.0)
# derivatives
dphi_dN = Y
ddy_dN = -3 * H * Y - interp_dV
# normalize speed for coloring
speed = np.sqrt(dphi_dN**2 + ddy_dN**2)
speed_n = theta(speed)

# ---------------------------- PLOT ------------------------------
fig, ax = plt.subplots(figsize=(6,6))
# colored background by speed
cf = ax.contourf(PHI, Y, speed_n, levels=50, cmap='viridis', vmin=0, vmax=1, alpha=0.85)
plt.colorbar(cf, ax=ax, label='tanh(|(φ\u2032,φ\u2033)|)')
# streamplot
ax.streamplot(phi_vals, y_vals, dphi_dN, ddy_dN, density=1.2, color='k', linewidth=0.6, arrowsize=1)
# labels
ax.set_xlim(phi_min, phi_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$d\phi/dN$')
ax.set_title('Phase Portrait for Canonical Inflation')
ax.grid(True)
plt.tight_layout()
plt.show()
