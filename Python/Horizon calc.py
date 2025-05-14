import os
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import numdifftools as nd

# suppress that old-deprecated warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Constants and Target Values ---
N_TARGET = 60.0
NS_TARGET = 0.9626
MP = 1.0
"""
Best Parameters Found:
alpha = 1.982917
k = 0.000001
mu = -0.888148
omega = 3.313422
chi = 1.888643
kappa = 4.476117

At N = 60.49, n_s = 0.962600
"""
# --- Model Parameters ---
params = {
    'alpha': 1.982917,
    'beta' : 1,
    'gamma': 0.0,
    'k'    : 0.000001,
    'mu'   : -0.888148,
    'omega': 3.313422,
    'chi'  : 1.888643,
    'lambda': 10,
    'kappa':4.476117
}

# make sure output directory exists
outdir = os.path.join(os.pardir, "inflation_plots")
os.makedirs(outdir, exist_ok=True)

# --- Derived constants ---
C_k = np.sqrt(2.0/(12.0 + params['k']))/MP
D   = np.sqrt(4*params['alpha']*params['beta'] - params['gamma']**2)

# --- Core functions ---
def X_of_phi_tilde(phi):
    return (D*np.tan(C_k*phi) - params['gamma'])/(2*params['beta'])

def V(phi):
    X = X_of_phi_tilde(phi)
    A = params['beta']*X**2 + params['gamma']*X + params['alpha']
    N = (params['kappa'] + 2*params['omega']*X + params['mu']*X**2 +
         2*params['chi']*X**3 + params['lambda']*X**4)
    return 0.125 * MP**4 * N / A**2

# use numdifftools for derivatives
dV   = nd.Derivative(V)
d2V  = nd.Derivative(V, n=2)

def epsilon_V(phi):
    return 0.5*(MP**2)*(dV(phi)/V(phi))**2

def eta_V(phi):
    return MP**2 * d2V(phi)/V(phi)

# find phi_end where epsilon_V=1
def find_phi_end():
    f     = lambda ph: epsilon_V(ph) - 1.0
    limit = 0.9*(0.5*np.pi/C_k)
    try:
        return optimize.brentq(f, -limit, -limit*0.1)
    except ValueError:
        return optimize.brentq(f, limit*0.1, limit)

phi_end = find_phi_end()
X_end   = X_of_phi_tilde(phi_end)
print(f"End of slow-roll at φ̃={phi_end:.4f}, X={X_end:.4f}")

# integrate φ(N)
def d2phi_dN(N, y):
    φ, dφ = y
    Vp = dV(φ)
    H  = np.sqrt(max(V(φ)/(3*MP**2) + dφ**2/(6*MP**2), 0))
    term = (3*MP**2 - 0.5*dφ**2)
    ddφ = -3*dφ + (term/(2*MP**2))*(Vp/V(φ)) + 0.5*(dφ**3)/(MP**2)
    return [dφ, ddφ]

y0 = [phi_end*0.2, 0.0]
solN = integrate.solve_ivp(d2phi_dN, [0, N_TARGET], y0, dense_output=True, max_step=0.1)
Nvals, φvals = solN.t, solN.y[0]

# phase portrait RHS
def phase_rhs(_, y):
    φ, dφ = y
    Vp = dV(φ)
    H  = np.sqrt(max((dφ**2/2 + V(φ))/(3*MP**2), 0))
    return [dφ, -3*H*dφ - Vp]

# grid for phase
phi_stream_min, phi_stream_max = -abs(phi_end)*1.5, abs(phi_end)*1.5
phi_dot_range = 1.0
phi_vals     = np.linspace(phi_stream_min, phi_stream_max, 40)
phi_dot_vals = np.linspace(-phi_dot_range, phi_dot_range, 40)
PHI, DPHI    = np.meshgrid(phi_vals, phi_dot_vals)
U = DPHI.copy()
Vv= np.zeros_like(U)
for i in range(PHI.shape[0]):
    for j in range(PHI.shape[1]):
        Vv[i,j] = phase_rhs(0, [PHI[i,j], DPHI[i,j]])[1]
speed = np.sqrt(U**2 + Vv**2)

# slow-roll & observables vs N
epsN = np.array([epsilon_V(phi) for phi in φvals])
etaN = np.array([eta_V(phi)     for phi in φvals])
n_sN  = 1 - 6*epsN + 2*etaN
rN    = 16*epsN

# a little text box with params
param_text = "\n".join(f"{k}={v:.3g}" for k,v in params.items())
textbox = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)

# === 1) Canonical potential ===
try:
    φgrid = np.linspace(-0.9*0.5*np.pi/C_k, 0.9*0.5*np.pi/C_k, 400)
    plt.figure()
    plt.plot(φgrid, [V(phi)/MP**4 for phi in φgrid], 'k-', label='V(φ̃)')
    plt.axvline(phi_end, color='r', ls='--', label='End of slow-roll (ε=1)')
    plt.xlabel(r'$\tilde\varphi$')
    plt.ylabel(r'$V/M_p^4$')
    plt.title('Canonical Potential')
    plt.legend(loc='upper right')
    plt.text(0.02, 0.95, param_text, transform=plt.gca().transAxes,
             fontsize=8, va="top", ha="left", bbox=textbox)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "potential.png"), dpi=300)
    plt.show()
except Exception as e:
    print("Error plotting potential:", e)

# === 2a) ε_V vs N ===
try:
    plt.figure()
    plt.plot(Nvals, epsN, 'r-')
    plt.xlabel('N')
    plt.ylabel(r'$\epsilon_V$')
    plt.title(r'$\epsilon_V(N)$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "epsilon_vs_N.png"), dpi=300)
    plt.show()
except Exception as e:
    print("Error plotting ε vs N:", e)

# === 2b) η_V vs N ===
try:
    plt.figure()
    plt.plot(Nvals, etaN, 'b-')
    plt.xlabel('N')
    plt.ylabel(r'$\eta_V$')
    plt.title(r'$\eta_V(N)$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "eta_vs_N.png"), dpi=300)
    plt.show()
except Exception as e:
    print("Error plotting η vs N:", e)

# === 2c) n_s vs N ===
try:
    plt.figure()
    plt.plot(Nvals, n_sN, 'g-')
    plt.xlabel('N')
    plt.ylabel(r'$n_s$')
    plt.title(r'$n_s(N)$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ns_vs_N.png"), dpi=300)
    plt.show()
except Exception as e:
    print("Error plotting n_s vs N:", e)

# === 3) φ(N) ===
try:
    plt.figure()
    plt.plot(Nvals, φvals, 'm-')
    plt.xlabel('N')
    plt.ylabel(r'$\tilde\varphi(N)$')
    plt.title('Field evolution φ(N)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phi_vs_N.png"), dpi=300)
    plt.show()
except Exception as e:
    print("Error plotting φ(N):", e)

# === 4) Phase portrait (uncompactified) ===
try:
    x = np.sort(PHI[0,:]); y = np.sort(DPHI[:,0])
    plt.figure()
    plt.contourf(PHI, DPHI, np.log10(speed+1e-12), levels=30, cmap='viridis')
    plt.streamplot(x, y, U, Vv, color='k', density=1.2, linewidth=0.5)
    # annotate start/end
    plt.scatter([φvals[0]], [0], c='white', edgecolors='k', s=80, label='φ_start (N=0)')
    plt.scatter([φvals[-1]], [0], c='cyan', edgecolors='k', s=80, label='φ_end (N=60)')
    plt.xlabel(r'$\tilde\varphi$')
    plt.ylabel(r'$\dot{\tilde\varphi}$')
    plt.title('Phase Portrait (Uncompactified)')
    plt.colorbar(label=r'flow speed')
    plt.legend(loc='upper right')
    plt.text(0.02, 0.95, param_text, transform=plt.gca().transAxes,
             fontsize=8, va="top", ha="left", bbox=textbox)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase_portrait.png"), dpi=300)
    plt.show()
except Exception as e:
    print("Error plotting phase portrait:", e)

# === 6) Compactified phase portrait ===
try:
    def compact(x): return np.tanh(x)
    def inv_compact(u): return np.arctanh(u)
    uc = np.linspace(-0.99, 0.99, 40)
    vc = np.linspace(-0.99, 0.99, 40)
    UC, VC = np.meshgrid(uc, vc)
    Uc = np.zeros_like(UC)
    Vc = np.zeros_like(VC)
    for i in range(UC.shape[0]):
        for j in range(UC.shape[1]):
            φ = inv_compact(UC[i,j])
            dφ = inv_compact(VC[i,j])
            derivs = phase_rhs(0, [φ, dφ])
            Uc[i,j] = (1-UC[i,j]**2)*derivs[0]
            Vc[i,j] = (1-VC[i,j]**2)*derivs[1]
    speed_c = np.sqrt(Uc**2 + Vc**2)

    plt.figure(figsize=(8,6))
    plt.contourf(UC, VC, np.log10(speed_c+1e-12), levels=30, cmap='viridis', alpha=0.8)
    plt.streamplot(UC, VC, Uc, Vc, color='k', density=1.2, linewidth=0.5)
    plt.xlabel(r'$\tanh(\tilde\varphi)$')
    plt.ylabel(r'$\tanh(\dot{\tilde\varphi})$')
    plt.title('Phase Portrait (Compactified)')
    plt.colorbar(label=r'$\log_{10}(\mathrm{flow\ speed})$')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase_portrait_compact.png"), dpi=300)
    plt.show()
except Exception as e:
    print("Error plotting compactified phase portrait:", e)

# === 7) n_s vs r ===
try:
    plt.figure()
    plt.plot(rN, n_sN, 'o-')
    plt.xlabel('r'); plt.ylabel('n_s')
    plt.title('n_s vs r')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ns_vs_r.png"), dpi=300)
    plt.show()
except Exception as e:
    print("Error plotting n_s vs r:", e)
