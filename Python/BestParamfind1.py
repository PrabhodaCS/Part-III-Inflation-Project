#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import optuna
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import quad, solve_ivp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1) Model definitions --------------------------------------------------

MP = 1.0
NS_TARGET = 0.9649
N_TARGET  = 60.0

def V(phi, C, lam, chi, mu, omega, kappa):
    θ = C*phi/MP
    s, c = np.sin(θ), np.cos(θ)
    return (MP**4/8) * (lam*s**4 + 2*chi*s**3*c + mu*s**2*c**2 + 2*omega*s*c**3 + kappa*c**4)

def dV(phi, C, lam, chi, mu, omega, kappa):
    θ = C*phi/MP
    s, c = np.sin(θ), np.cos(θ)
    dθ = C/MP
    # derivative expanded from your analytic expression:
    return (MP**4/8)*dθ * (
       lam*4*s**3*c
     + 2*chi*(3*s**2*c**2 - s**4)
     + mu*(2*s*c**3 - 2*s**3*c)
     + 2*omega*(c**4 - 3*s**2*c**2)
     - 4*kappa*s*c**3
    )

def d2V(phi, C, lam, chi, mu, omega, kappa):
    θ = C*phi/MP
    s, c = np.sin(θ), np.cos(θ)
    dθ = C/MP
    # second derivative expanded:
    return (MP**4/8)*dθ**2 * (
       lam*4*(3*s**2*c**2 - s**4)
     + 2*chi*(6*s*c**3 - 12*s**3*c)
     + mu*(2*c**4 - 12*s**2*c**2 + 2*s**4)
     + 2*omega*(-4*c**3*s + 12*s**3*c)
     - kappa*(4*c**4 - 12*c**2*s**2)
    )

def eps_SR(phi, params):
    C, lam, chi, mu, omega, kappa = params
    v = V(phi, *params)
    if v<=0: return np.inf
    dv = dV(phi, *params)
    return 0.5*(MP*dv/v)**2

def eta_SR(phi, params):
    C, lam, chi, mu, omega, kappa = params
    v = V(phi, *params)
    if v<=0: return np.inf
    d2 = d2V(phi, *params)
    return MP**2 * d2 / v

# --- 2) Find φ_end & φ_* -----------------------------------------------------

def find_phi_end(params):
    C, *_ = params
    φ_min = -np.pi*MP/(2*C)+1e-6
    φ_max =  np.pi*MP/(2*C)-1e-6
    def obj(φ):
        return (eps_SR(φ, params) - 1)**2
    res = minimize_scalar(obj, bounds=(φ_min, φ_max), method='bounded')
    return res.x if res.fun<1e-6 else None

def N_integral(φ, φ_end, params):
    """ ∫_φ^φ_end (V/dV) dφ """
    return quad(lambda x: -V(x,*params)/dV(x,*params),
                φ, φ_end, limit=200)[0]

def find_phi_star(params, N_target=N_TARGET):
    φ_end = find_phi_end(params)
    if φ_end is None: return None, None
    C = params[0]
    # depending on slope sign, bracket correctly:
    sign = np.sign(dV(φ_end,*params))
    a,b = (φ_end+1e-6, np.pi*MP/(2*C)-1e-6) if sign>0 else (-np.pi*MP/(2*C)+1e-6, φ_end-1e-6)
    try:
        φ_star = brentq(lambda φ: N_integral(φ,φ_end,params) - N_target, a, b, rtol=1e-8)
        return φ_star, φ_end
    except ValueError:
        return None, None

# --- 3) Optuna objective ----------------------------------------------------

def is_potential_positive(params):
    C = params[0]
    φ_min = -np.pi * MP / (2 * C) + 1e-6
    φ_max = np.pi * MP / (2 * C) - 1e-6
    φ_values = np.linspace(φ_min, φ_max, 100)  # Sample 100 points in the range
    for φ in φ_values:
        if V(φ, *params) <= 0:
            return False
    return True

def objective(trial):
    # Sample shape parameters
    C = trial.suggest_float('C', 0.01, 1.0)
    mu = trial.suggest_float('mu', -10.0, 10.0)
    omega = trial.suggest_float('omega', -10.0, 10.0)
    chi = trial.suggest_float('chi', -10.0, 10.0)
    lam = trial.suggest_float('lam', -10.0, 10.0)
    kappa = trial.suggest_float('kappa', -10.0, 10.0)
    params = [C, lam, chi, mu, omega, kappa]

    # Check if the potential is positive across the range
    if not is_potential_positive(params):
        return 1e6  # Penalize non-positive potential

    # Find φ* and φ_end
    φ_star, φ_end = find_phi_star(params)
    if φ_star is None or φ_end is None:
        return 1e6  # Penalize invalid parameter sets

    # Compute n_s at φ*
    eps = eps_SR(φ_star, params)
    et = eta_SR(φ_star, params)
    n_s = 1 - 6 * eps + 2 * et

    # Require inflation ends
    if eps_SR(φ_end, params) <= 1:
        return 1e5  # Penalize if inflation doesn't end

    # Compute the cost
    cost = (n_s - NS_TARGET)**2
    trial.set_user_attr('n_s', n_s)
    return cost

# --- 4) Run the optimization -----------------------------------------------

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2000)

best = study.best_params
params_opt = [
    best['C'],
    best['lam'],
    best['chi'],
    best['mu'],
    best['omega'],
    best['kappa']
]
print("Optimal shape params:", params_opt, "→ n_s=", study.best_trial.user_attrs['n_s'])

# save params
os.makedirs('results', exist_ok=True)
with open('results/optimal_params.txt','w') as f:
    for k,v in best.items(): f.write(f"{k} = {v}\n")
with open('results/optimal_params.tex','w') as f:
    f.write("\\begin{tabular}{ll}\n")
    for k,v in best.items(): f.write(f"{k} & {v:.6f}\\\\\n")
    f.write("\\end{tabular}\n")

# --- 5) Find best initial condition at N_init=70 ---------------------------

φ_end = find_phi_end(params_opt)
# shoot back to N_init = 70 e-folds before end
from functools import partial
def find_phi_init(params, N_init=70.0):
    φ_end = find_phi_end(params)
    if φ_end is None: return None
    return find_phi_star(params, N_target=N_init)[0]
φ_init = find_phi_init(params_opt, N_init=70.0)

# Hubble & phi-dot initial from slow-roll at φ_init
H0 = np.sqrt(V(φ_init,*params_opt)/(3*MP**2))
φdot0 = -dV(φ_init,*params_opt)/(3*H0)
y0 = [φ_init, φdot0, 0.0]      # [φ, φ̇,  N(t)]

# --- 6) Full ODE integration in cosmic time -------------------------------

def background(t,y):
    φ, v, N = y
    VV = V(φ,*params_opt)
    H  = np.sqrt((0.5*v**2+VV)/(3*MP**2))
    dv = -3*H*v - dV(φ,*params_opt)
    return [v, dv, H]

# stop when ε → 1
def end_inflation(t,y):
    φ, v, N = y
    H  = np.sqrt((0.5*v**2+V(φ,*params_opt))/(3*MP**2))
    return 0.5*(v**2)/(H**2) - 1.0
end_inflation.terminal = True

sol = solve_ivp(background, [0,1e5], y0,
                events=end_inflation,
                rtol=1e-8, atol=1e-10, dense_output=True)

t  = sol.t
φ  = sol.y[0]
v  = sol.y[1]
Nt = sol.y[2]

# compute N-space trajectories by interpolation
from scipy.interpolate import interp1d
t_of_N = interp1d(Nt, t, fill_value="extrapolate")
N_vals   = np.linspace(0, Nt[-1], 1000)
sol_N    = solve_ivp(lambda N, Y: [ Y[1]/(d0:=Nt[-1]), 0 ], [0, Nt[-1]], [φ_init, φdot0])

# recompute on uniform N grid by dense_output
φ_of_t = sol.sol
φ_vals = sol.sol(t)[0]
eps_vals = 0.5*(v**2)/( (0.5*v**2+V(φ,*params_opt))/(3*MP**2) )
eta_vals = [] 
for φi, vi in zip(φ, v):
    eps_i = eps_SR(φi, params_opt)
    eta_vals.append(eta_SR(φi, params_opt))

n_s_vals = 1 - 6*eps_vals + 2*np.array(eta_vals)
r_vals   = 16*eps_vals

# --- 7) Plotting -----------------------------------------------------------

os.makedirs('results', exist_ok=True)
txt = "\n".join(f"{k}={v:.3g}" for k,v in best.items())
bbox = dict(boxstyle='round', facecolor='white', alpha=0.7)

# 1) Potential
phis = np.linspace(-np.pi/(2*best['C']), np.pi/(2*best['C']), 500)
Vs   = [V(p,*params_opt) for p in phis]
plt.figure(); plt.plot(phis, Vs); plt.title('Potential'); plt.xlabel('φ'); plt.ylabel('V'); 

plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
        fontsize=7, verticalalignment='top', bbox=bbox)

plt.savefig('results/potential.png',dpi=200); plt.close()

# 2) φ vs N
plt.figure(); plt.plot(Nt, φ); plt.title('φ vs N'); plt.xlabel('N'); plt.ylabel('φ'); 

plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
        fontsize=7, verticalalignment='top', bbox=bbox)
plt.savefig('results/phi_vs_N.png',dpi=200); plt.close()

# 3) ε,η vs φ
mask = (np.abs(eta_vals)<10)&(eps_vals<10)
plt.figure(); plt.plot(φ[mask], eps_vals[mask],label='ε'); plt.plot(φ[mask], np.abs(eta_vals)[mask],label='|η|')
plt.legend(); plt.title('ε,η vs φ'); plt.xlabel('φ'); plt.ylabel(''); 
plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
        fontsize=7, verticalalignment='top', bbox=bbox)
plt.savefig('results/eps_eta_vs_phi.png',dpi=200); plt.close()

# 4) ε,η vs N
plt.figure(); plt.plot(Nt, eps_vals,label='ε'); plt.plot(Nt, np.abs(eta_vals),label='|η|')
plt.legend(); plt.title('ε,η vs N'); plt.xlabel('N'); plt.ylabel(''); 
plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
        fontsize=7, verticalalignment='top', bbox=bbox)
plt.savefig('results/eps_eta_vs_N.png',dpi=200); plt.close()

# 5) n_s vs N
plt.figure(); plt.plot(Nt, n_s_vals); plt.axhline(NS_TARGET,ls='--',color='k')
plt.title('n_s vs N'); plt.xlabel('N'); plt.ylabel('n_s'); 
plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
        fontsize=7, verticalalignment='top', bbox=bbox)
plt.savefig('results/ns_vs_N.png',dpi=200); plt.close()

# 6) r vs N
plt.figure(); plt.plot(Nt, r_vals); plt.title('r vs N'); plt.xlabel('N'); plt.ylabel('r');
plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
        fontsize=7, verticalalignment='top', bbox=bbox)
plt.savefig('results/r_vs_N.png',dpi=200); plt.close()

def phase_rhs(t, y):
    φ0, v0 = y
    Hc = np.sqrt((0.5*v0**2 + V(φ0,*params_opt)) / (3*MP**2))
    return [v0, -3*Hc*v0 - dV(φ0,*params_opt)]

phi_range = np.linspace(-np.pi/(2*best['C']), np.pi/(2*best['C']), 100)
dot_phi_max = 2.0
dot_phi_range = np.linspace(-dot_phi_max, dot_phi_max, 100)
PHI, DOT_PHI = np.meshgrid(phi_range, dot_phi_range)

U = np.zeros_like(PHI)
V_phase = np.zeros_like(PHI)
for i in range(PHI.shape[0]):
    for j in range(PHI.shape[1]):
        U[j,i], V_phase[j,i] = phase_rhs(0, [PHI[j,i], DOT_PHI[j,i]])

speed = np.sqrt(U**2 + V_phase**2)

fig, ax = plt.subplots(figsize=(6,5))
cont = ax.contourf(PHI, DOT_PHI, np.log(speed + 1e-12),
                   levels=50, cmap='viridis', alpha=0.8)
strm = ax.streamplot(PHI, DOT_PHI, U, V_phase,
                     color='k', density=1.2, linewidth=0.6, arrowsize=1)
ax.set_xlabel(r'$\tilde{\varphi}$')
ax.set_ylabel(r'$\dot{\tilde{\varphi}}$')
ax.set_title('Uncompactified Phase Portrait')
ax.grid(True)
ax.text(0.02, 0.95, txt, transform=ax.transAxes,
        fontsize=7, verticalalignment='top', bbox=bbox)
# colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(cont, cax=cax, label='Log Flow Speed')
plt.savefig('results/phase_uncompact.png', dpi=200)
plt.close()

# Compactified
def inv_tanh(u): return np.arctanh(u)
uc = np.linspace(-0.99, 0.99, 100)
vc = np.linspace(-0.99, 0.99, 100)
UC, VC = np.meshgrid(uc, vc)

Uc = np.zeros_like(UC)
Vc = np.zeros_like(VC)
for i in range(UC.shape[0]):
    for j in range(UC.shape[1]):
        phi0 = inv_tanh(UC[j,i])
        v0   = inv_tanh(VC[j,i])
        derivs = phase_rhs(0, [phi0, v0])
        Uc[j,i] = (1 - UC[j,i]**2)*derivs[0]
        Vc[j,i] = (1 - VC[j,i]**2)*derivs[1]

speed_c = np.sqrt(Uc**2 + Vc**2)

fig, ax = plt.subplots(figsize=(6,5))
cont_c = ax.contourf(UC, VC, np.log(speed_c + 1e-12),
                     levels=50, cmap='viridis', alpha=0.8)
ax.streamplot(UC, VC, Uc, Vc,
              color='k', density=1.2, linewidth=0.6, arrowsize=1)
ax.set_xlabel(r'$\tanh(\tilde{\varphi})$')
ax.set_ylabel(r'$\tanh(\dot{\tilde{\varphi}})$')
ax.set_title('Compactified Phase Portrait')
ax.grid(True)
ax.text(0.02, 0.95, txt, transform=ax.transAxes,
        fontsize=7, verticalalignment='top', bbox=bbox)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(cont_c, cax=cax, label='Log Flow Speed')
plt.savefig('results/phase_compact.png', dpi=200)
plt.close()

# 8) n_s vs r
plt.figure(); plt.plot(r_vals,n_s_vals); plt.xlabel('r'); plt.ylabel('n_s')
plt.title('n_s vs r');
plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
        fontsize=7, verticalalignment='top', bbox=bbox)
plt.savefig('results/ns_vs_r.png',dpi=200); plt.close()

print("➡️ All results, plots & param files are in ./results/")

