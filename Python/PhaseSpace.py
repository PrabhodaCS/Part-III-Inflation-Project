import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------------------
# 1.  Model Definitions
# ----------------------------------------
# Physical constants (choose units where M_p = 1 for simplicity)
M_p = 1.0
M = np.sqrt(2)*M_p

# Example model parameters (you can swap in your fitted values)
alpha, beta, gamma, k_param = -52.4048, -0.4628, -185.1555, 27.3976
phi0 = -529.7076
mu, la, ka = 10, 0.0, 0.0  # set λ=κ=0 for plateau

def A(phi):
    """Denominator function A(φ) = βφ² + γφ₀φ + αφ₀²."""
    return beta*phi**2 + gamma*phi0*phi + alpha*phi0**2  # (38)

def K(phi):
    """
    Non-canonical kinetic coefficient K(φ) from Eq. (37):
      = M²[−36β/A + φ₀²(γ²−4βα)(k−6)/(2A²)]
    """
    A_val = A(phi)
    return M**2 * (
        -36*beta/A_val
        + (phi0**2*(gamma**2 - 4*beta*alpha)*(k_param - 6)) / (2*A_val**2)
    )

def V(phi):
    """
    Potential V(φ) from Eq. (38):
      = M⁴ [μ²φ₀²φ² + λφ⁴ + κφ₀⁴] / (2A²)
    """
    A_val = A(phi)
    return M**4 * (mu**2 * phi0**2 * phi**2 + la*phi**4 + ka*phi0**4) / (2*A_val**2)

# Derivatives of K and V for the ODE
def dK_dphi(phi):
    h = 1e-6
    return (K(phi + h) - K(phi - h)) / (2*h)

def dV_dphi(phi):
    h = 1e-6
    return (V(phi + h) - V(phi - h)) / (2*h)


# ----------------------------------------
# 2.  Dynamical System: φ' = π, π' = ...
# ----------------------------------------
def rhs(N, Y):
    """
    Right-hand side of the system:
      Y = [φ, π],  φ' = π,
      π' = -(1/K)[ ½K'π² + 3H K π + V' ]
    where H² = [Kπ²/2 + V]/(3M_p²).
    """
    phi, pi = Y
    K_val = K(phi)
    if K_val <= 0:
        return [pi, 0]  # avoid singularities

    # Hubble
    energy = 0.5*K_val*pi**2 + V(phi)
    H = np.sqrt(np.maximum(energy/(3*M_p**2), 0))

    # derivatives
    term1 = 0.5 * dK_dphi(phi) * pi**2
    term2 = 3 * H * K_val * pi
    term3 = dV_dphi(phi)
    pi_dot = - (term1 + term2 + term3) / K_val

    return [pi, pi_dot]


# ----------------------------------------
# 3.  Phase‑Portrait Grid of Initial Conditions
# ----------------------------------------
# Define grid of (φ₀, π₀) to sample
phi_vals0 = np.linspace(-600, 600, 100)
pi_vals0  = np.linspace(-2, 2, 10)

# ODE integration settings
N_span = (0, 100)                # "time" range in e‑folds
sol_kwargs = dict(method='RK45', rtol=1e-6, atol=1e-8)

# Prepare figure
fig, ax = plt.subplots(figsize=(8,6))

# Plot vector field with streamplot (smooth arrows)
# Sample a coarser mesh for arrows
Phi, Pi = np.meshgrid(np.linspace(-600,600,1000), np.linspace(-3,3,1000))
U = np.zeros_like(Phi)
Vv = np.zeros_like(Pi)
for i in range(Phi.shape[0]):
    for j in range(Phi.shape[1]):
        dphi, dpi = rhs(0, [Phi[i,j], Pi[i,j]])
        U[i,j], Vv[i,j] = dphi, dpi
ax.streamplot(Phi, Pi, U, Vv, density=1.0, color='gray', linewidth=0.5)  # :contentReference[oaicite:1]{index=1}

# Overlay solution curves for each initial condition
for phi0_ic in phi_vals0:
    for pi0_ic in pi_vals0:
        sol = solve_ivp(rhs, N_span, [phi0_ic, pi0_ic], t_eval=np.linspace(0,100,500), **sol_kwargs)
        ax.plot(sol.y[0], sol.y[1], color='C0', alpha=0.6, lw=1)

# Formatting
ax.set_xlabel(r'$\varphi$')
ax.set_ylabel(r'$\pi = d\varphi/dN$')
ax.set_title('Phase Portrait: non‑canonical inflationary dynamics')
ax.set_xlim(-600,600)
ax.set_ylim(-3,3)
ax.grid(True)
plt.tight_layout()
plt.show()
