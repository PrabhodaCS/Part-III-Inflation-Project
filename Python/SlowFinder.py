#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d
from matplotlib.transforms import Bbox
import matplotlib.patches as mpatches

# -----------------------------------------------------------------------------
# 1) GLOBAL AND TARGET PARAMETERS
# -----------------------------------------------------------------------------
M_p        = 1.0
M          = np.sqrt(2)*M_p
N_target   = 60
n_s_target = 0.9626
# weight on attractor penalty
lambda_attr = 1e2
"""
alpha = -0.2821, beta = -81.8965, gamma = -36.1827
k = 59.2854, phi0 = -361.5663, mu = 0.0755

Optimal irrparams:, la = 0.8120, ka = 0.0501, q = -1.1516,li = 0.1553

"""
# initial guess
initial_guess = [
     -0.2821,   # α
    -81.8965,   # β
     -36.1827,   # γ
     59.2854,   # k
    -361.5663,   # φ₀
     0.0755,   # μ
     0.8120,   # λ
     0.0501,   # κ
     0.1553,   # ω
     -1.1516    # χ
]

# -----------------------------------------------------------------------------
# 2) MODEL FUNCTIONS
# -----------------------------------------------------------------------------
def K_func(phi, α,β,γ,k,φ0):
    A = β*phi**2 + γ*φ0*phi + α*φ0**2
    return (
        -36*β*M**2 / A
        + (M**2*φ0**2*(γ**2 - 4*β*α)*(k-6)) / (2*A**2)
    )

def V_func(phi, α,β,γ,k,φ0,μ,lam,kap,ω,χ):
    A = β*phi**2 + γ*φ0*phi + α*φ0**2
    num = (kap*φ0**4
           + 2*ω*φ0**3*phi
           + μ**2*φ0**2*phi**2
           + 2*χ*φ0*phi**3
           + lam*phi**4)
    return M**4 * num / (2*A**2)

# finite-difference helpers
def dlnV_dphi(phi, params):
    h = 1e-6
    Vp = V_func(phi+h, *params)
    Vm = V_func(phi-h, *params)
    return (np.log(abs(Vp)+1e-20) - np.log(abs(Vm)+1e-20))/(2*h)

def dK_dphi(phi, params):
    h = 1e-6
    Kp = K_func(phi+h, *params[:5])
    Km = K_func(phi-h, *params[:5])
    return (Kp - Km)/(2*h)

# the non‑canonical dynamics: φ'' = f(φ,φ')
def dyn(N, Y, params):
    φ, φp = Y
    α,β,γ,k,φ0,μ,lam,kap,ω,χ = params
    K  = K_func(φ,α,β,γ,k,φ0)
    dK = dK_dphi(φ, params)
    dlnV = dlnV_dphi(φ, params)
    num = (  3*K*φp
           - (K**2)/(2*M_p**2)*φp**3
           + 0.5*dK*φp**2
           + (3*M_p**2 - 0.5*K*φp**2)*dlnV )
    return [φp, -num/K]

# -----------------------------------------------------------------------------
# 3) ATTRACTOR TEST
#    Given params, launch two trajectories with Δφ₀ small
#    and measure their separation at N=N_target.
# -----------------------------------------------------------------------------
def attractor_penalty(params):
    base_IC    = [100.0, 0.1]
    perturbed  = [100.0+1e-2, 0.1]
    Ns = np.linspace(0, N_target, 200)
    sol1 = solve_ivp(lambda N,Y: dyn(N,Y,params), (0,N_target), base_IC, t_eval=Ns, max_step=0.5)
    sol2 = solve_ivp(lambda N,Y: dyn(N,Y,params), (0,N_target), perturbed, t_eval=Ns, max_step=0.5)
    φ1, φp1 = sol1.y
    φ2, φp2 = sol2.y
    # final separation in phase space
    sep = np.sqrt((φ1[-1]-φ2[-1])**2 + (φp1[-1]-φp2[-1])**2)
    # penalty = larger sep ⇒ more penalized; we want sep→0
    return sep

# -----------------------------------------------------------------------------
# 4) SPECTRAL INDEX COST
# -----------------------------------------------------------------------------
def cost(params):
    # enforce K>0 over the integration
    α,β,γ,k,φ0,*rest = params
    Ns = np.linspace(0,70,500)
    sol = solve_ivp(lambda N,Y: dyn(N,Y,params), (0,70), [100.0,0.1],
                    t_eval=Ns, max_step=0.5)
    φ, φp = sol.y
    if np.any(K_func(φ,α,β,γ,k,φ0)<=0):
        return 1e6
    # compute n_s(N)
    Varr = V_func(φ, *params)
    dlnV = np.gradient(np.log(Varr), φ)
    ε = 0.5*(M_p**2)*(dlnV**2)/K_func(φ,α,β,γ,k,φ0)
    η = (M_p**2)*np.gradient(dlnV, φ)/K_func(φ,α,β,γ,k,φ0)
    n_s = 1 - 6*ε + 2*η
    idx   = np.argmin(np.abs(Ns-N_target))
    δ_ns = (n_s[idx]-n_s_target)**2
    # attractor penalty
    p_attr = attractor_penalty(params)
    return δ_ns + lambda_attr*p_attr

# run the minimization
result = minimize(cost, initial_guess, method='Nelder-Mead',
                  options={'xatol':1e-5,'maxiter':200})
opt = result.x
print("Optimal params:", np.round(opt,4))

# -----------------------------------------------------------------------------
# 5) PLOT PHASE PORTRAIT
# -----------------------------------------------------------------------------
α,β,γ,k,φ0,μ,lam,kap,ω,χ = opt
# grid
phi_vals = np.linspace(-200,200,300)
y_vals   = np.linspace(-20,20,300)
Φ,Y = np.meshgrid(phi_vals,y_vals)
U   = Y.copy()
W   = np.zeros_like(U)
for i in range(Φ.shape[0]):
    for j in range(Φ.shape[1]):
        W[i,j] = dyn(0,[Φ[i,j],Y[i,j]],opt)[1]
speed = np.tanh(np.sqrt(U**2+W**2))

fig,ax = plt.subplots(figsize=(7,6))
cf = ax.contourf(Φ,Y,speed,50,cmap='viridis',alpha=0.8)
fig.colorbar(cf,ax=ax,label=r'$\tanh|\dot X|$')
ax.streamplot(phi_vals,y_vals,U,W,density=1.2,color='k',linewidth=0.6)
ax.set_xlabel(r'$\varphi$'); ax.set_ylabel(r'$d\varphi/dN$')
ax.set_title("Phase Portrait with Attractor Optimization")

# anchored textbox
textstr = "\n".join([
    rf"$\alpha={opt[0]:.4f}$, $\beta={opt[1]:.4f}$",
    rf"$\gamma={opt[2]:.4f}$, $k={opt[3]:.4f}$",
    rf"$\phi_0={opt[4]:.4f}$, $\mu={opt[5]:.4f}$",
    rf"$\lambda={opt[6]:.4f}$, $\kappa={opt[7]:.4f}$",
    rf"$\omega={opt[8]:.4f}$, $\chi={opt[9]:.4f}$"
])
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.02,0.98, textstr, transform=ax.transAxes, fontsize=9,
        va='top', ha='left', bbox=props)  # :contentReference[oaicite:0]{index=0}

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 6) POTENTIAL COMPARISON
# -----------------------------------------------------------------------------
phi_plot = np.linspace(-200,200,1000)
V_nc = V_func(phi_plot, *opt)

# canonical mapping
def canonical(phi_array, params):
    sol = solve_ivp(lambda x,y: np.sqrt(max(K_func(y,*params[:5]),1e-12)),
                    (phi_array[0],phi_array[-1]), [0.0],
                    t_eval=phi_array, method='DOP853')
    return sol.y[0]

φc = canonical(phi_plot,opt)
V_c = V_nc  # same V but vs φc

fig,axes = plt.subplots(1,2,figsize=(12,5))
axes[0].plot(phi_plot, V_nc, 'C0')
axes[0].set_title("Non‐Canonical Potential"); axes[0].grid()
axes[0].set_xlabel(r'$\varphi$'); axes[0].set_ylabel(r'$V$')

axes[1].plot(φc, V_c, 'C1')
axes[1].set_title("Canonical Potential"); axes[1].grid()
axes[1].set_xlabel(r'$\varphi_{\rm can}$')

plt.tight_layout()
plt.show()
