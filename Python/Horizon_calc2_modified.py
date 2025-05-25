# inflation_optimizer.py
# --------------------
# This script finds optimal parameters for a six-parameter inflaton potential such that:
#  1. The spectral index n_s = 0.9649 at N = 60
#  2. Slow-roll violation occurs somewhere in the integration range
# It uses Optuna for efficient hyperparameter search and saves outputs (parameters, plots, LaTeX, text) into an output folder.

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import optuna
from functools import partial

# Set Planck mass to unity for simplicity
M_p = 1.0
# Target spectral index at N=60
TARGET_NS = 0.9649
# Integration range for e-fold N
N_START, N_END = 0.0, 70.0
# Slow-roll threshold
EPSILON_THRESHOLD = 1.0

# Define the potential V(phi) and its derivatives in pure Python

def V(theta, params):
    C, mu, omega, chi, lam, kappa = params
    # theta = C * phi / M_p
    s = np.sin(theta)
    c = np.cos(theta)
    return 0.125 * M_p**4 * (
        lam * s**4 + 2*chi * s**3 * c + mu * s**2 * c**2 + 2*omega * s * c**3 + kappa * c**4
    )

def dV_dphi(phi, params):
    C, mu, omega, chi, lam, kappa = params
    theta = C * phi / M_p
    s2 = np.sin(2*theta)
    c2 = np.cos(2*theta)
    c4 = np.cos(4*theta)
    term = ((-chi + omega) * c4 + (-kappa + lam) * s2 + c2 *
            (chi + omega - (kappa + lam - mu) * np.sin(2*theta)))
    return 0.125 * C * M_p**3 * term

def d2V_dphi2(phi, params):
    C, mu, omega, chi, lam, kappa = params
    theta = C * phi / M_p
    c2 = np.cos(2*theta)
    c4 = np.cos(4*theta)
    s2 = np.sin(2*theta)
    s4 = np.sin(4*theta)
    term = (
        (kappa - lam) * c2 + (kappa + lam - mu) * c4 + (chi + omega) * s2 + 2*( -chi + omega) * s4
    )
    return -0.25 * C**2 * M_p**2 * term

# Define background EOM in terms of phi(N)
# d2phi/dN2 + 3 dphi/dN - 0.5 (dphi/dN)^3 / M_p^2
# + (3 M_p^2 - 0.5 (dphi/dN)^2) * d ln V / dphi = 0

def eom_N(N, y, params):
    phi, phip = y
    Vval = V(params[0]*phi/M_p, params)
    dV = dV_dphi(phi, params)
    denom = 3 * M_p**2 - 0.5 * phip**2
    # second derivative
    phipp = -3*phip + 0.5 * phip**3 / M_p**2 - denom * (dV / Vval)
    return [phip, phipp]

# Compute slow-roll parameters and spectral index as functions of N

def compute_observables(phi_series, phip_series, params):
    # H^2 = (phip^2/2 + V)/ (3 M_p^2)
    Vseries = V(params[0]*phi_series/M_p, params)
    H2 = (0.5 * phip_series**2 + Vseries) / (3 * M_p**2)
    epsilon = 0.5 * (phip_series**2) / (M_p**2)
    # second slow-roll eta: 
    dV2 = d2V_dphi2(phi_series, params)
    eta = dV2 / Vseries
    # spectral index
    ns = 1 - 6*epsilon + 2*eta
    return epsilon, eta, ns

# Objective for Optuna

def objective(trial):
    # Sample parameters within physically allowed ranges
    C = trial.suggest_uniform('C', 0.1, 2.0)
    mu = trial.suggest_uniform('mu', -10.0, 10.0)
    omega = trial.suggest_uniform('omega', -10.0, 10.0)
    chi = trial.suggest_uniform('chi', -10.0, 10.0)
    lam = trial.suggest_uniform('lambda', -10.0, 10.0)
    kappa = trial.suggest_uniform('kappa', -10.0, 10.0)
    params = (C, mu, omega, chi, lam, kappa)
    # initial conditions: phi(N=0), phip(N=0)
    phi0 = trial.suggest_uniform('phi0', -np.pi/(2*C)*0.9, np.pi/(2*C)*0.9)
    phip0 = trial.suggest_uniform('phip0', -1, 1)

    # integrate
    sol = solve_ivp(lambda N, y: eom_N(N, y, params), 
                    [N_START, N_END], [phi0, phip0], dense_output=True, max_step=0.1)
    Nvals = np.linspace(N_START, N_END, 1000)
    y = sol.sol(Nvals)
    phi_series, phip_series = y

    # compute observables
    epsilon, eta, ns = compute_observables(phi_series, phip_series, params)
    # find N=60 index
    idx60 = np.argmin(np.abs(Nvals - 60))
    ns60 = ns[idx60]
    # check slow-roll violation
    violated = np.any(epsilon > EPSILON_THRESHOLD)

    # objective: squared deviation of ns60 and heavy penalty if no violation
    loss = (ns60 - TARGET_NS)**2
    if not violated:
        loss += 1e3  # large penalty
    trial.report(loss, step=0)
    return loss


def main():
    # create outputs directory
    outdir = 'output'
    os.makedirs(outdir, exist_ok=True)

    # run Optuna study with progress
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=2000, show_progress_bar=True)

    # collect best params
    best = study.best_params
    print("Best parameters:", best)

    # save params
    with open(os.path.join(outdir, 'best_params.txt'), 'w') as f:
        for k,v in best.items(): f.write(f"{k} = {v}\n")
    # also save as LaTeX
    with open(os.path.join(outdir, 'best_params.tex'), 'w') as f:
        f.write("\\begin{align*}\n")
        for k,v in best.items(): f.write(f"{k} &= {v} \\\n")
        f.write("\\end{align*}\n")

    # generate plots for best params
    params = (best['C'], best['mu'], best['omega'], best['chi'], best['lambda'], best['kappa'])
    sol = solve_ivp(lambda N, y: eom_N(N, y, params),
                    [N_START, N_END], [best['phi0'], best['phip0']], dense_output=True, max_step=0.1)
    Nvals = np.linspace(N_START, N_END, 1000)
    phi, phip = sol.sol(Nvals)
    eps, eta, ns = compute_observables(phi, phip, params)
    # 1. Potential vs phi
    plt.figure(); plt.plot(phi, V(params[0]*phi/M_p, params)); plt.xlabel('phi'); plt.ylabel('V'); plt.savefig(os.path.join(outdir,'potential.png'))
    # 2. phi vs N
    plt.figure(); plt.plot(Nvals, phi); plt.xlabel('N'); plt.ylabel('phi'); plt.savefig(os.path.join(outdir,'phi_vs_N.png'))
    # 3. epsilon and eta vs phi
    plt.figure(); plt.plot(phi, eps, label='epsilon'); plt.plot(phi, eta, label='eta'); plt.legend(); plt.xlabel('phi'); plt.savefig(os.path.join(outdir,'slowroll_vs_phi.png'))
    # 4. epsilon and eta vs N
    plt.figure(); plt.plot(Nvals, eps, label='epsilon'); plt.plot(Nvals, eta, label='eta'); plt.legend(); plt.xlabel('N'); plt.savefig(os.path.join(outdir,'slowroll_vs_N.png'))
    # 5. n_s vs N
    plt.figure(); plt.plot(Nvals, ns); plt.axvline(60, linestyle='--'); plt.xlabel('N'); plt.ylabel('n_s'); plt.savefig(os.path.join(outdir,'ns_vs_N.png'))
    # 6. r vs N (r ~ 16 epsilon)
    plt.figure(); plt.plot(Nvals, 16*eps); plt.xlabel('N'); plt.ylabel('r'); plt.savefig(os.path.join(outdir,'r_vs_N.png'))
    # 7. phase space portraits
    # uncompactified
    plt.figure(); plt.scatter(phi, phip, s=1); plt.xlabel('phi'); plt.ylabel('phip'); plt.savefig(os.path.join(outdir,'phase_uncompact.png'))
    # compactified (e.g. tanh)
    plt.figure(); plt.scatter(np.tanh(phi), np.tanh(phip), s=1); plt.xlabel('tanh(phi)'); plt.ylabel('tanh(phip)'); plt.savefig(os.path.join(outdir,'phase_compact.png'))
    # 8. n_s vs r
    plt.figure(); plt.scatter(16*eps, ns, s=2); plt.xlabel('r'); plt.ylabel('n_s'); plt.savefig(os.path.join(outdir,'ns_vs_r.png'))

    print(f"All outputs saved in {outdir}/")

if __name__ == '__main__':
    main()
