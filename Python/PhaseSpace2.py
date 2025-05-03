#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------------------
# 1) PHYSICAL PARAMETERS
# ----------------------------
M_p   = 1.0     # reduced Planck mass (set to 1 for simplicity)
M     = np.sqrt(2)*M_p    # your M
# example parameter values – replace with your “optimal” ones:


"""
Optimal parameters:
alpha = -1.9272, beta = -4.7452, gamma = 0.0893
k = 0.1244, phi0 = 0.1304, mu = 0.1079

Optimal irrparams:, la = -0.0004, ka = 0.1056, q = -1.2946,li = 0.1169

Final ns at N=60: 0.96260
"""

alpha, beta, gamma = -52.40480, -0.4628, -185.15550
phi0, k = -529.70760, 27.49760
mu, lam, kap, q , li  = 10, -0.2, 0,0,-0.1
"""
alpha, beta, gamma  = -0.2821 , -81.8965,  -36.1827
k ,phi0 , mu  = 59.2854, -361.5663,  0.0755
lam, kap ,  q, li = 0.8120, 0.0501, -1.1516, 0.1553
"""
# ----------------------------
# 2) DEFINE K(phi) and V(phi)
# ----------------------------
def K(phi):
    A = beta*phi**2 + gamma*phi0*phi + alpha*phi0**2
    return (
        -36*beta*M**2      / A
        + (M**2*phi0**2*(gamma**2 - 4*beta*alpha)*(k-6))
          / (2*A**2)
    )

def V(phi):
    A = beta*phi**2 + gamma*phi0*phi + alpha*phi0**2
    num = (2*li*phi0**3*phi + mu**2*phi0**2*phi**2 + 2*q*phi0*phi**3 + lam*phi**4 + kap*phi0**4)
    return M**4 * num / (2*A**2)


def central_diff(f, x, eps=1e-6):
    x = np.asarray(x)
    return (f(x + eps) - f(x - eps)) / (2 * eps)

dK_dphi = np.vectorize(lambda φ: central_diff(K, φ))
dlnV_dphi = np.vectorize(lambda φ: central_diff(lambda x: np.log(np.abs(V(x)) + 1e-20), φ))

# ----------------------------
# 3) ODE SYSTEM
# ----------------------------
def phase_derivs(Y, N):
    φ, y = Y
    Kφ = K(φ)
    Kp = dK_dphi(φ)
    lv = dlnV_dphi(φ)
    # second derivative from your eq. (21):
    num = (
        3*Kφ*y
        - (Kφ**2)/(2*M_p**2)*y**3
        + (Kp/2)*y**2
        + (3*M_p**2 - 0.5*Kφ*y**2)*lv
    )
    ddy = - num / Kφ
    return [y, ddy]

# ----------------------------
# 4) BUILD VECTOR FIELD ON GRID
# ----------------------------
phi_min, phi_max = -1000, 1000
y_min,   y_max   = -100, +100
ngrid = 30000

phi_vals = np.linspace(phi_min, phi_max, ngrid)
y_vals   = np.linspace(y_min,   y_max,   ngrid)
Φ, Y      = np.meshgrid(phi_vals, y_vals)

U = Y   # dφ/dN = y
W = np.zeros_like(U)
for i in range(ngrid):
    for j in range(ngrid):
        _, ddy = phase_derivs([Φ[i,j], Y[i,j]], 0.0)
        W[i,j] = ddy

speed = np.sqrt(U**2 + W**2)
speed_n = np.tanh(speed)   # clamp to (-1,1)

# ----------------------------
# 5) PLOT PHASE PORTRAIT
# ----------------------------
fig, ax = plt.subplots(figsize=(7,6))
divider = make_axes_locatable(ax)
cax     = divider.append_axes("right", size="5%", pad=0.1)

# background contour of speed
cf = ax.contourf(Φ, Y, speed_n, levels=50,
                 cmap='viridis', vmin=0, vmax=1, alpha=0.85)
fig.colorbar(cf, cax=cax, label=r'$\tanh(\sqrt{U^2+V^2})$')

# streamlines
ax.streamplot(phi_vals, y_vals, U, W,
              density=1.2, color='k', linewidth=0.6, arrowsize=1)

seed_x = np.linspace(5, 100, 5)
seed_y = np.linspace(-10, 10, 5)
SX, SY = np.meshgrid(seed_x, seed_y)
SEED_POINTS = np.vstack((SX.ravel(), SY.ravel())).T 

# overlay a handful of trajectories from different seeds
#seeds = SEED_POINTS"
"""
seeds = [(-4,0.5), (-4,1.0), (4,-1.0), (0,1.5), (2,0.0), (-2,-0.5)]
for φ0, y0 in seeds:
    Ns = np.linspace(0, 100, 2000)
    sol = odeint(phase_derivs, [φ0, y0], Ns)
    ax.plot(sol[:,0], sol[:,1], lw=1.5)"""


ax.set_xlim(phi_min, phi_max)
ax.set_ylim(y_min,   y_max)
ax.set_xlabel(r'$\varphi$')
ax.set_ylabel(r'$\dfrac{d\varphi}{dN}$')
ax.set_title("Phase Portrait of Non‐Canonical Inflaton")
ax.legend(loc='upper right', fontsize=8)
ax.grid(True)
plt.tight_layout()
plt.show()
