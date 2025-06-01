#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_inflation_de.py

This script uses Differential Evolution (DE) to find the best‐fit 5 parameters
{C, λ, χ, μ, κ} (with ω=0 and μ−2κ>0) of the canonical inflaton potential:

    V(ϕ) = (M_p^4 / 8) [ λ sin^4(θ) + 2χ sin^3(θ)cos(θ)
                        + μ sin^2(θ)cos^2(θ) + κ cos^4(θ) ],
    where θ = C·ϕ / M_p.

We enforce the following “hard” constraints via a fine φ‐grid scan:
  1)  δ = μ − 2κ > 0  ⇒  V''(0)>0  (ϕ=0 is a local minimum).
  2)  V(ϕ) > 0  ∀ ϕ ∈ (−π/(2C), +π/(2C)).
  3)  ∃ ϕ in that domain with ε(ϕ) > 1  (slow‐roll violation ⇒ inflation ends).
  4)  ∃ ϕ in that domain with |η(ϕ)| > 1  (second slow‐roll violation).
  5)  At N=60 e‐folds, the spectral index n_s = 0.9649.

Output:
  • Prints optimized (C, λ, χ, μ, κ) and resulting n_s.
  • Saves plots under ./results_de/.
"""

import os
import numpy as np
from scipy.optimize import differential_evolution, brentq
from scipy.integrate import quad, solve_ivp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === 0) Global Constants ===
MP = 1.0             # Reduced Planck mass (set to 1)
NS_TARGET = 0.9649   # Planck 2018 best‐fit scalar spectral index at N=60
N_TARGET = 60.0      # Required number of e‐folds

# Output directory
OUT_DIR = "results_de"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1) Inflationary Model: V(ϕ), dV/dϕ, and finite‐difference d²V/dϕ²
# =============================================================================

def V(phi, C, lam, chi, mu, kappa):
    """
    Canonical inflaton potential:
    V(ϕ) = (MP^4/8) [ λ sin^4(θ) + 2χ sin^3(θ)cos(θ)
                     + μ sin^2(θ)cos^2(θ) + κ cos^4(θ) ],
    where θ = C·ϕ / MP  and ω=0 so the sinθ cos^3θ term is absent.
    """
    θ = C * phi / MP
    s = np.sin(θ)
    c = np.cos(θ)
    return (MP**4 / 8.0) * (
        lam * s**4
        + 2.0 * chi * s**3 * c
        + mu * s**2 * c**2
        + kappa * c**4
    )

def dV(phi, C, lam, chi, mu, kappa):
    """
    Analytic first derivative: dV/dϕ = (C/MP)⋅(dV/dθ), where
      dV/dθ = (MP^4/8)[ 4λ s^3 c
                       + 2χ(3s^2 c^2 − s^4)
                       + μ(2s c^3 − 2s^3 c)
                       − 4κ s c^3 ].
    """
    θ = C * phi / MP
    s = np.sin(θ)
    c = np.cos(θ)
    dθ_dϕ = C / MP
    dV_dθ = (MP**4 / 8.0) * (
        4.0 * lam * s**3 * c
      + 2.0 * chi * (3.0 * s**2 * c**2 - s**4)
      + mu * (2.0 * s * c**3 - 2.0 * s**3 * c)
      - 4.0 * kappa * s * c**3
    )
    return dV_dθ * dθ_dϕ

def d2V_finite_diff(phi, C, lam, chi, mu, kappa):
    """
    Second derivative via central finite‐difference (h=1e−6):
      d²V/dϕ² ≈ [dV(ϕ+h) - 2dV(ϕ) + dV(ϕ-h)] / h².
    Avoids unwieldy symbolic expression.
    """
    h = 1e-6
    return (
        dV(phi + h, C, lam, chi, mu, kappa)
        - 2.0 * dV(phi,     C, lam, chi, mu, kappa)
        +   dV(phi - h, C, lam, chi, mu, kappa)
    ) / (h**2)

# =============================================================================
# 2) Slow‐Roll Parameters: ε(ϕ), η(ϕ), and grid‐based “violation” checks
# =============================================================================

def eps_SR(phi, params):
    """
    ε(ϕ) = 0.5 * [MP * V'(ϕ)/V(ϕ)]^2.
    If V(ϕ) ≤ 0, return +∞ to mark infeasibility.
    """
    C, lam, chi, mu, kappa = params
    Vval = V(phi, C, lam, chi, mu, kappa)
    if Vval <= 0.0:
        return np.inf
    dVval = dV(phi, C, lam, chi, mu, kappa)
    return 0.5 * (MP * dVval / Vval)**2

def eta_SR(phi, params):
    """
    η(ϕ) = MP^2 * [ d²V(ϕ)/dϕ² ] / V(ϕ) via finite difference for d²V.
    If V(ϕ) ≤ 0, return +∞.
    """
    C, lam, chi, mu, kappa = params
    Vval = V(phi, C, lam, chi, mu, kappa)
    if Vval <= 0.0:
        return np.inf
    d2Vval = d2V_finite_diff(phi, C, lam, chi, mu, kappa)
    return MP**2 * (d2Vval / Vval)

def grid_slow_roll_violation(params, Ngrid=50000):
    """
    Check on a uniform φ‐grid in (−π/(2C), +π/(2C)) whether BOTH
      • max ε(ϕ) > 1  (first slow‐roll violation), AND 
      • max |η(ϕ)| > 1 (second slow‐roll violation).
    We sample Ngrid points.  If either condition fails, return False.
    """
    C, lam, chi, mu, kappa = params
    if C <= 0.0:
        return False
    φ_min = -np.pi/(2.0*C) + 1e-6
    φ_max = +np.pi/(2.0*C) - 1e-6
    φ_vals = np.linspace(φ_min, φ_max, Ngrid)
    eps_vals = np.array([eps_SR(ϕ, params) for ϕ in φ_vals])
    eta_vals = np.array([eta_SR(ϕ, params) for ϕ in φ_vals])
    # Require at least one ε>1 AND at least one |η|>1
    return (np.nanmax(eps_vals) > 1.0) and (np.nanmax(np.abs(eta_vals)) > 1.0)

# =============================================================================
# 3) e‐fold Computing and φ* Search (N=60)
# =============================================================================

def compute_efolds(phi, phi_end, params):
    """
    N(ϕ) = ∫_{ϕ}^{ϕ_end} [ −V(ϕ) / V'(ϕ) ] dϕ.
    Integrate from (ϕ + ε_cut) to (ϕ_end − ε_cut) to avoid divergence at V'→0.
    If the integral fails or diverges, return +∞.
    """
    eps_cut = 1e-8
    if (phi_end - phi) < eps_cut:
        return 0.0
    a = phi + eps_cut
    b = phi_end - eps_cut
    if a >= b:
        return 0.0
    try:
        val, _ = quad(
            lambda x: -V(x, *params) / dV(x, *params),
            a, b, epsabs=1e-8, epsrel=1e-8, limit=20000
        )
        return val
    except Exception:
        return np.inf

def find_phi_end(params):
    """
    Bracket‐scan for φ_end such that ε(φ_end)=1:
      1) Sample ε(ϕ)−1 on a uniform grid of 500 points in (−π/(2C), +π/(2C)).
      2) Locate first adjacent sign change in [ε−1].
      3) Refine with brentq. Returns φ_end or None if no root.
    """
    C, lam, chi, mu, kappa = params
    if C <= 0.0:
        return None
    φ_min = -np.pi/(2.0*C) + 1e-6
    φ_max = +np.pi/(2.0*C) - 1e-6
    Ngrid = 500000
    φ_grid = np.linspace(φ_min, φ_max, Ngrid)
    f_vals = np.array([eps_SR(ϕ, params) - 1.0 for ϕ in φ_grid])
    idx = np.where(f_vals[:-1] * f_vals[1:] < 0.0)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    a, b = φ_grid[i], φ_grid[i+1]
    φ_end = brentq(lambda ϕ: eps_SR(ϕ, params) - 1.0, a, b, rtol=1e-2)
    return φ_end

def find_phi_star(params, N_target=N_TARGET):
    """
    Given params = [C, λ, χ, μ, κ], find (φ_star, φ_end) such that:
      1) φ_end solves ε(φ_end)=1
      2) φ_star solves compute_efolds(φ_star, φ_end)=N_target
    Bracketing logic:
      If V'(φ_end)>0 ⇒ φ_star<φ_end  (bracket on [φ_min, φ_end−ε])
      Else            ⇒ φ_star>φ_end  (bracket on [φ_end+ε, φ_max]).
    Returns (φ_star, φ_end) or (None,None) if failure.
    """
    φ_end = find_phi_end(params)
    if φ_end is None:
        print("Debug: Failed to find φ_end.")
        return None, None

    C = params[0]
    sign_slope = np.sign(dV(φ_end, *params))
    φ_min = -np.pi/(2.0*C) + 1e-6
    φ_max = +np.pi/(2.0*C) - 1e-6

    if sign_slope > 0:
        a, b = φ_min, φ_end - 1e-6
    else:
        a, b = φ_end + 1e-6, φ_max

    # Validate bracketing
    f_a = compute_efolds(a, φ_end, params) - N_target
    f_b = compute_efolds(b, φ_end, params) - N_target
    print(f"Debug: f(a) = {f_a}, f(b) = {f_b}")  # Debugging output

    if f_a * f_b > 0:
        print("Debug: f(a) and f(b) have the same sign. Bracketing failed.")
        return None, None

    try:
        φ_star = brentq(
            lambda ϕ: compute_efolds(ϕ, φ_end, params) - N_target,
            a, b, rtol=1e-4, maxiter=20000
        )
        return φ_star, φ_end
    except ValueError as e:
        print(f"Debug: Brentq failed with error: {e}")
        return None, None

# =============================================================================
# 4) Constraint‐Violation Penalty (CV)
# =============================================================================

def constraint_violation(x):
    """
    Compute a “constraint‐violation” penalty CV(x) for x=[C,λ,χ,μ,κ]:
      1) δ = μ − 2κ.  If δ ≤ 0 ⇒ CV += |δ| + 1e−6.
      2) C must be > 0.  If C ≤ 0 ⇒ CV += 1e6.
      3) V(ϕ) > 0 on a 100‐point φ‐grid: for each V(ϕ) ≤ 0 ⇒ CV += |V(ϕ)|.
      4) Slow‐roll violation: require ∃ϕ with ε(ϕ)>1 AND |η(ϕ)|>1 
         on a 500‐point grid; if not, CV += 1e6.
    Return CV (≥0).  Infeasible sets get large penalty → excluded from DE.
    """
    C, lam, chi, mu, kappa = x
    CV = 0.0

    # 1) δ = μ − 2κ > 0
    delta = mu - 2.0*kappa
    if delta <= 0.0:
        CV += abs(delta) + 1e-6


    # 3) V(ϕ) positivity on 100‐point grid
    if C > 0.0:
        φ_min = -np.pi/(2.0*C) + 1e-6
        φ_max = +np.pi/(2.0*C) - 1e-6
        φ_vals = np.linspace(φ_min, φ_max, 100)
        for ϕ in φ_vals:
            Vval = V(ϕ, C, lam, chi, mu, kappa)
            if Vval <= 0.0:
                CV += abs(Vval)

    # 4) Slow‐roll violation: require both ε>1 and |η|>1 somewhere
    if not grid_slow_roll_violation([C, lam, chi, mu, kappa], Ngrid=500):
        CV += 1e100

    return CV

# =============================================================================
# 5) DE Objective Function
# =============================================================================

def objective_DE(x):
    """
    Differential Evolution objective:
      1) Compute CV = constraint_violation(x).  If CV > 0 ⇒ return CV.
      2) Else, let params=[C, λ, χ, μ, κ].  Find (φ_star, φ_end).  If None ⇒ return 1e100.
      3) Compute ε*, η* at φ_star.  If not finite ⇒ return 1e100.
      4) Compute n_s = 1 − 6ε* + 2η*, and return (n_s − NS_TARGET)^2.
    """
    C, lam, chi, mu, kappa = x

    # (1) Feasibility check
    CV = constraint_violation(x)
    if CV > 0.0:
        return CV

    params = [C, lam, chi, mu, kappa]
    # (2) Find φ_star, φ_end
    φ_star, φ_end = find_phi_star(params)
    if (φ_star is None) or (φ_end is None):
        # Penalize heavily for bracketing failure
        return 1e100

    # (3) Compute ε*(φ_star)
    eps_val = eps_SR(φ_star, params)
    if not np.max(eps_val) > 1:
        return 1e100

    # (4) Compute η*(φ_star)
    d2V_val = d2V_finite_diff(φ_star, C, lam, chi, mu, kappa)
    Vval_star = V(φ_star, C, lam, chi, mu, kappa)
    if Vval_star <= 0.0:
        return 1e100
    eta_val = MP**2 * (d2V_val / Vval_star)
    if not np.max(eta_val) > 1:
        return 1e100

    # (5) Spectral index
    n_s = 1.0 - 6.0*eps_val + 2.0*eta_val
    return (n_s - NS_TARGET)**2

# =============================================================================
# 6) Run Differential Evolution
# =============================================================================

def run_DE_optimization():
    """
    Run SciPy’s differential_evolution with recommended settings:
      – strategy='best1bin'
      – maxiter=500, popsize=10 (NP=50)
      – tol=1e-3, mutation=(0.5,0.9), recombination=0.9, seed=42, disp=True, polish=True
    Returns the OptimizeResult.
    """
    bounds = [
        (0.001,   2.0),   # C > 0
        (-100.0, 100.0),  # λ
        (-100.0, 100.0),  # χ
        (-100.0, 100.0),  # μ (can be negative)
        (-100.0, 100.0)   # κ (can be negative)
    ]
    de_args = {
        "strategy":      "best1bin",
        "maxiter":       50,
        "popsize":       10,
        "tol":           1e-5,
        "mutation":      (0.5, 0.9),
        "recombination": 0.9,
        "seed":          42,
        "disp":          True,
        "polish":        False
    }
    result = differential_evolution(objective_DE, bounds, **de_args)
    return result

# =============================================================================
# 7) Main: Optimize & Then Plot All Quantities
# =============================================================================

if __name__ == "__main__":
    # 7.1) Optimize
    print("🔍 Running Differential Evolution optimization... (may take a few minutes)")
    res = run_DE_optimization()

    # 7.2) Extract best‐fit parameters
    C_opt, lam_opt, chi_opt, mu_opt, kappa_opt = res.x
    delta_opt = mu_opt - 2.0*kappa_opt
    best_ns = np.sqrt(res.fun) + NS_TARGET if (res.fun >= 0.0) else NS_TARGET

    print("\n=== Optimization Result ===")
    print(f"C    = {C_opt:.6f}")
    print(f"λ    = {lam_opt:.6f}")
    print(f"χ    = {chi_opt:.6f}")
    print(f"μ    = {mu_opt:.6f}")
    print(f"κ    = {kappa_opt:.6f}")
    print(f"δ=μ-2κ = {delta_opt:.6f}")
    print(f"Achieved n_s ≈ {best_ns:.6f}\n")

    # Save LaTeX table
    with open(os.path.join(OUT_DIR, "optimal_params.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{ll}\n")
        f.write(f"C    & {C_opt:.6f} \\\\\n")
        f.write(f"λ    & {lam_opt:.6f} \\\\\n")
        f.write(f"χ    & {chi_opt:.6f} \\\\\n")
        f.write(f"μ    & {mu_opt:.6f} \\\\\n")
        f.write(f"κ    & {kappa_opt:.6f} \\\\\n")
        f.write(f"δ=μ-2κ& {delta_opt:.6f} \\\\\n")
        f.write("\\end{tabular}\n")

    # 7.3) Find φ_end, φ_star (should succeed by construction)
    params_opt = [C_opt, lam_opt, chi_opt, mu_opt, kappa_opt]
    φ_star_opt, φ_end_opt = find_phi_star(params_opt)
    if (φ_star_opt is None) or (φ_end_opt is None):
        raise RuntimeError("Error: failed to find valid φ_star or φ_end despite explicit constraints")
    print(f"φ_end (ε=1) ≈ {φ_end_opt:.6f},  φ_star (N=60) ≈ {φ_star_opt:.6f}")

    # 7.4) Integrate full background ODE:
    #      dϕ/dt = π,  dπ/dt = −3H π − dV/dϕ,  dN/dt = H
    def background_equations(t, y):
        ϕ_val, π_val, N_val = y
        V_val = V(ϕ_val, C_opt, lam_opt, chi_opt, mu_opt, kappa_opt)
        H = np.sqrt(max((0.5*π_val**2 + V_val)/(3.0*MP**2), 0.0))
        dπ = -3.0*H*π_val - dV(ϕ_val, C_opt, lam_opt, chi_opt, mu_opt, kappa_opt)
        dϕ = π_val
        dN = H
        return [dϕ, dπ, dN]

    # Initial conditions at φ_star (approximate π_star from slow‐roll)
    H_star = np.sqrt(V(φ_star_opt, C_opt, lam_opt, chi_opt, mu_opt, kappa_opt)/(3.0*MP**2))
    π_star = -dV(φ_star_opt, C_opt, lam_opt, chi_opt, mu_opt, kappa_opt)/(3.0*H_star)
    y0 = [φ_star_opt, π_star, 0.0]  # N=0 at φ_star

    # Event: stop when ε(ϕ(t)) = 1 again (inflation ends)
    def end_inflation_event(t, y):
        ϕ_val, π_val, N_val = y
        V_val = V(ϕ_val, C_opt, lam_opt, chi_opt, mu_opt, kappa_opt)
        if V_val <= 0.0:
            return 1.0
        H_val = np.sqrt((0.5*π_val**2 + V_val)/(3.0*MP**2))
        eps_val = 0.5*(π_val**2)/(H_val**2*MP**2)
        return eps_val - 1.0
    end_inflation_event.terminal = True
    end_inflation_event.direction = 1

    sol_bg = solve_ivp(
        background_equations,
        (0.0, 1e5),
        y0,
        events=end_inflation_event,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True
    )

    t_vals = sol_bg.t
    ϕ_vals = sol_bg.y[0]
    π_vals = sol_bg.y[1]
    N_vals = sol_bg.y[2]

    # 7.5) Compute ε(N), η(N), n_s(N), r(N)
    eps_vals = np.array([eps_SR(ϕ, params_opt) for ϕ in ϕ_vals])
    eta_vals = np.array([eta_SR(ϕ, params_opt) for ϕ in ϕ_vals])
    n_s_vals = 1.0 - 6.0*eps_vals + 2.0*eta_vals
    r_vals = 16.0*eps_vals

    # =============================================================================
    # 8) Plotting: Potential, ϕ(N), ε(N), η(N), n_s(N), r(N), Phase Portraits
    # =============================================================================

    label_text = (
        f"C={C_opt:.3g}, λ={lam_opt:.3g}, χ={chi_opt:.3g},\n"
        f"μ={mu_opt:.3g}, κ={kappa_opt:.3g}, δ={delta_opt:.3g}\n"
        f"n_s={best_ns:.4f}"
    )
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)

    # 8.1) Potential V(ϕ)
    φ_plot = np.linspace(-0.9*np.pi/(2.0*C_opt), +0.9*np.pi/(2.0*C_opt), 1000)
    V_plot = np.array([V(ϕ, C_opt, lam_opt, chi_opt, mu_opt, kappa_opt) for ϕ in φ_plot])
    plt.figure(figsize=(6,4))
    plt.plot(φ_plot, V_plot, "k-", label="V(ϕ)")
    plt.axvline(φ_end_opt, color="r", linestyle="--", label="ϕ_end (ε=1)")
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$V(\varphi)$")
    plt.title("Canonical Inflaton Potential")
    plt.text(0.02, 0.95, label_text, transform=plt.gca().transAxes,
             fontsize=8, va="top", ha="left", bbox=bbox_props)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "potential.png"), dpi=200)
    plt.close()

    # 8.2) ϕ vs N
    plt.figure(figsize=(6,4))
    plt.plot(N_vals, ϕ_vals, "b-")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\varphi$")
    plt.title(r"$\varphi(N)$")
    plt.text(0.02, 0.95, label_text, transform=plt.gca().transAxes,
             fontsize=8, va="top", ha="left", bbox=bbox_props)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phi_vs_N.png"), dpi=200)
    plt.close()

    # 8.3) ε(N) and |η(N)|
    plt.figure(figsize=(6,4))
    plt.plot(N_vals, eps_vals, "r-", label=r"$\epsilon$")
    plt.plot(N_vals, np.abs(eta_vals), "b--", label=r"$|\eta|$")
    plt.xlabel(r"$N$")
    plt.ylabel("Slow‐roll parameters")
    plt.title(r"$\epsilon(N)$ and $|\eta(N)|$")
    plt.legend()
    plt.text(0.02, 0.95, label_text, transform=plt.gca().transAxes,
             fontsize=8, va="top", ha="left", bbox=bbox_props)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "slowroll_vs_N.png"), dpi=200)
    plt.close()

    # 8.4) n_s vs N
    plt.figure(figsize=(6,4))
    plt.plot(N_vals, n_s_vals, "g-")
    plt.axhline(NS_TARGET, color="k", linestyle="--", label=r"$n_s^{\rm target}$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$n_s$")
    plt.title(r"$n_s(N)$")
    plt.legend()
    plt.text(0.02, 0.95, label_text, transform=plt.gca().transAxes,
             fontsize=8, va="top", ha="left", bbox=bbox_props)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ns_vs_N.png"), dpi=200)
    plt.close()

    # 8.5) r vs N
    plt.figure(figsize=(6,4))
    plt.plot(N_vals, r_vals, "m-")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$r$")
    plt.title(r"$r(N)$")
    plt.text(0.02, 0.95, label_text, transform=plt.gca().transAxes,
             fontsize=8, va="top", ha="left", bbox=bbox_props)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "r_vs_N.png"), dpi=200)
    plt.close()

    # 8.6) n_s vs r (parametric)
    plt.figure(figsize=(6,4))
    plt.plot(r_vals, n_s_vals, "ko-", label="Trajectory")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$n_s$")
    plt.title(r"$n_s \;{\rm vs.}\; r$")
    plt.axhline(NS_TARGET, color="gray", linestyle="--")
    plt.text(0.02, 0.95, label_text, transform=plt.gca().transAxes,
             fontsize=8, va="top", ha="left", bbox=bbox_props)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ns_vs_r.png"), dpi=200)
    plt.close()

    # 8.7) Phase Portraits (Uncompactified & Compactified)
    def phase_rhs(phi_val, pi_val):
        V_val = V(phi_val, C_opt, lam_opt, chi_opt, mu_opt, kappa_opt)
        Hc = np.sqrt(max((0.5*pi_val**2 + V_val)/(3.0*MP**2), 0.0))
        return pi_val, -3.0*Hc*pi_val - dV(phi_val, C_opt, lam_opt, chi_opt, mu_opt, kappa_opt)

    # Uncompactified grid
    φ_range = np.linspace(-0.9*np.pi/(2.0*C_opt), +0.9*np.pi/(2.0*C_opt), 100)
    π_max = max(np.abs(π_vals)) * 1.5
    π_range = np.linspace(-π_max, +π_max, 100)
    PHI, PI = np.meshgrid(φ_range, π_range)
    U = np.zeros_like(PHI)
    V_field = np.zeros_like(PHI)
    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            u_raw, v_raw = phase_rhs(PHI[i, j], PI[i, j])
            U[i, j] = u_raw
            V_field[i, j] = v_raw
    speed = np.sqrt(U**2 + V_field**2)
    plt.figure(figsize=(6,5))
    plt.contourf(PHI, PI, np.log10(speed + 1e-12), levels=50, cmap="viridis", alpha=0.8)
    plt.streamplot(φ_range, π_range, U, V_field, color="k", density=1.2, linewidth=0.5, arrowsize=0.4)
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$\dot\varphi$")
    plt.title("Phase Portrait (Uncompactified)")
    plt.text(0.02, 0.95, label_text, transform=plt.gca().transAxes,
             fontsize=7, va="top", ha="left", bbox=bbox_props)
    plt.colorbar(label="Log Flow Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phase_uncompactified.png"), dpi=200)
    plt.close()

    # Compactified: u = tanh(ϕ), v = tanh(π)
    def inv_tanh(u):
        return np.arctanh(np.clip(u, -0.999999, 0.999999))

    u_vals = np.linspace(-0.99, 0.99, 100)
    v_vals = np.linspace(-0.99, 0.99, 100)
    UC, VC = np.meshgrid(u_vals, v_vals)
    Uc = np.zeros_like(UC)
    Vc = np.zeros_like(VC)
    for i in range(UC.shape[0]):
        for j in range(UC.shape[1]):
            phi0 = inv_tanh(UC[i, j])
            pi0 = inv_tanh(VC[i, j])
            u_raw, v_raw = phase_rhs(phi0, pi0)
            Uc[i, j] = (1.0 - UC[i, j]**2)*u_raw
            Vc[i, j] = (1.0 - VC[i, j]**2)*v_raw
    speed_c = np.sqrt(Uc**2 + Vc**2)
    plt.figure(figsize=(6,5))
    plt.contourf(UC, VC, np.log10(speed_c + 1e-12), levels=50, cmap="viridis", alpha=0.8)
    plt.streamplot(UC, VC, Uc, Vc, color="k", density=1.2, linewidth=0.5, arrowsize=0.4)
    plt.xlabel(r"$\tanh(\varphi)$")
    plt.ylabel(r"$\tanh(\dot\varphi)$")
    plt.title("Phase Portrait (Compactified)")
    plt.text(0.02, 0.95, label_text, transform=plt.gca().transAxes,
             fontsize=7, va="top", ha="left", bbox=bbox_props)
    plt.colorbar(label="Log Flow Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phase_compactified.png"), dpi=200)
    plt.close()

    print(f"\n✅ All plots saved under '{OUT_DIR}/'")

