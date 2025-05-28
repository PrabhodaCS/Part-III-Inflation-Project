import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # For better colorbar placement
import optuna
import os
import datetime
from numba import jit

# Global constants
M_p = 1.0  # Planck mass
THETA_PLOT_EPSILON = 1e-5 # How close theta can be to +/- pi/2 for plotting evaluations

# --- Potential and its derivatives (JIT compiled for speed) ---
@jit(nopython=True)
def potential_V(phi, C, mu, omega, chi, lam, kap):
    """
    Calculates the inflaton potential V(phi).
    phi is tilde_phi.
    Returns np.nan if theta is outside (-pi/2 + eps, pi/2 - eps) for plotting.
    """
    if C <= 1e-9: 
        return np.nan # Undefined if C is too small for a valid theta range

    theta = C * phi / M_p
    
    if theta <= -np.pi/2 + THETA_PLOT_EPSILON or theta >= np.pi/2 - THETA_PLOT_EPSILON:
        return np.nan # For plotting, treat points outside strict domain as NaN

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Guard against cos_theta being zero if it's in a denominator implicitly
    # For this potential, all cos_theta are in numerators or squared.
    # If cos_theta is extremely small (theta near pi/2), terms might become large or small.

    term1 = lam * sin_theta**4
    term2 = 2 * chi * sin_theta**3 * cos_theta
    term3 = mu * sin_theta**2 * cos_theta**2
    term4 = 2 * omega * sin_theta * cos_theta**3
    term5 = kap * cos_theta**4
    
    val = (M_p**4 / 8.0) * (term1 + term2 + term3 + term4 + term5)
    
    if not np.isfinite(val) or val <= 1e-30: # Potential should be positive and finite
        return np.nan # Or a very small positive number if that's more appropriate for logs later
    return val

@jit(nopython=True)
def potential_V_phi(phi, C, mu, omega, chi, lam, kap):
    """
    Calculates dV/dphi.
    phi is tilde_phi.
    Returns np.nan if theta is outside (-pi/2 + eps, pi/2 - eps) for plotting.
    """
    if C <= 1e-9:
        return np.nan 

    theta = C * phi / M_p
    if theta <= -np.pi/2 + THETA_PLOT_EPSILON or theta >= np.pi/2 - THETA_PLOT_EPSILON:
        return np.nan
    
    s = np.sin(theta)
    c = np.cos(theta)
    
    dVdtheta_term1 = lam * 4 * s**3 * c
    dVdtheta_term2 = 2 * chi * (3 * s**2 * c**2 - s**4) 
    dVdtheta_term3 = mu * (2 * s * c**3 - 2 * s**3 * c) 
    dVdtheta_term4 = 2 * omega * (c**4 - 3 * s**2 * c**2) 
    dVdtheta_term5 = kap * 4 * c**3 * (-s)
    
    dVdtheta = (M_p**4 / 8.0) * (dVdtheta_term1 + dVdtheta_term2 + dVdtheta_term3 + dVdtheta_term4 + dVdtheta_term5)
    val = dVdtheta * (C / M_p)

    if not np.isfinite(val):
        return np.nan
    return val


@jit(nopython=True)
def potential_V_phi_phi(phi, C, mu, omega, chi, lam, kap):
    """
    Calculates d^2V/dphi^2.
    phi is tilde_phi.
    Returns np.nan if theta is outside (-pi/2 + eps, pi/2 - eps) for plotting.
    """
    if C <= 1e-9:
        return np.nan

    theta = C * phi / M_p
    if theta <= -np.pi/2 + THETA_PLOT_EPSILON or theta >= np.pi/2 - THETA_PLOT_EPSILON:
        return np.nan 
    
    s = np.sin(theta)
    c = np.cos(theta)

    d2Vdtheta2_term1 = lam * (12 * s**2 * c**2 - 4 * s**4)
    d2Vdtheta2_term2 = 2 * chi * (6*s*c**3 - 10*s**3*c) 
    d2Vdtheta2_term3 = mu * (2*c**4 - 12*s**2*c**2 + 2*s**4)
    d2Vdtheta2_term4 = 2 * omega * (-10*c**3*s + 6*s**3*c)
    d2Vdtheta2_term5 = kap * (12*c**2*s**2 - 4*c**4)

    d2Vdtheta2 = (M_p**4 / 8.0) * (d2Vdtheta2_term1 + d2Vdtheta2_term2 + d2Vdtheta2_term3 + d2Vdtheta2_term4 + d2Vdtheta2_term5)
    val = d2Vdtheta2 * (C / M_p)**2

    if not np.isfinite(val):
        return np.nan
    return val

# --- ODE system for phi(N) and dphi/dN (JIT compiled) ---
# This system uses the original potential functions which return large values at boundaries
# for the solver's benefit, guided by event detection.
@jit(nopython=True)
def potential_V_solver(phi, C, mu, omega, chi, lam, kap):
    if C <= 1e-9: return 1e30 
    theta = C * phi / M_p
    epsilon_boundary = 1e-7 # Solver uses tighter boundary check before returning large V
    if theta <= -np.pi/2 + epsilon_boundary or theta >= np.pi/2 - epsilon_boundary: return 1e30 
    sin_theta = np.sin(theta); cos_theta = np.cos(theta)
    term1 = lam * sin_theta**4; term2 = 2 * chi * sin_theta**3 * cos_theta
    term3 = mu * sin_theta**2 * cos_theta**2; term4 = 2 * omega * sin_theta * cos_theta**3
    term5 = kap * cos_theta**4
    val = (M_p**4 / 8.0) * (term1 + term2 + term3 + term4 + term5)
    if not np.isfinite(val) or val <= 1e-30: return 1e30 
    return val

@jit(nopython=True)
def potential_V_phi_solver(phi, C, mu, omega, chi, lam, kap):
    if C <= 1e-9: return 0.0 
    theta = C * phi / M_p
    epsilon_boundary = 1e-7
    if theta <= -np.pi/2 + epsilon_boundary or theta >= np.pi/2 - epsilon_boundary: return 1e30 * np.sign(theta) 
    s = np.sin(theta); c = np.cos(theta)
    dVdtheta_term1 = lam * 4 * s**3 * c; dVdtheta_term2 = 2 * chi * (3 * s**2 * c**2 - s**4)
    dVdtheta_term3 = mu * (2 * s * c**3 - 2 * s**3 * c); dVdtheta_term4 = 2 * omega * (c**4 - 3 * s**2 * c**2)
    dVdtheta_term5 = kap * 4 * c**3 * (-s)
    dVdtheta = (M_p**4 / 8.0) * (dVdtheta_term1 + dVdtheta_term2 + dVdtheta_term3 + dVdtheta_term4 + dVdtheta_term5)
    val = dVdtheta * (C / M_p)
    if not np.isfinite(val): return 1e30 * np.sign(val) if val !=0 and np.isfinite(np.sign(val)) else 0.0
    return val

@jit(nopython=True)
def ode_system(N, y, C, mu, omega, chi, lam, kap):
    phi, phi_prime = y 
    V = potential_V_solver(phi, C, mu, omega, chi, lam, kap)       # Use _solver version
    V_p = potential_V_phi_solver(phi, C, mu, omega, chi, lam, kap) # Use _solver version

    if V >= 1e29: 
        return np.array([0.0, 0.0]) 
    if V <= 1e-30: 
        return np.array([0.0, 0.0]) 
    H_sq_factor_inv_V = (3 * M_p**2 - 0.5 * phi_prime**2)
    if H_sq_factor_inv_V <= 1e-9: 
        return np.array([phi_prime, 1e5]) 
    if np.abs(V_p) >= 1e29: 
         return np.array([phi_prime, -np.sign(V_p) * 1e5]) 
    phi_double_prime = -3 * phi_prime + \
                       (1.0 / (2.0 * M_p**2)) * phi_prime**3 - \
                       H_sq_factor_inv_V * (V_p / V)
    if not np.isfinite(phi_double_prime):
        phi_double_prime = 1e5 
    return np.array([phi_prime, phi_double_prime])

# --- Event functions for solve_ivp ---
@jit(nopython=True)
def event_epsilon_H_equals_one(N, y, C, mu, omega, chi, lam, kap):
    phi, phi_prime = y
    epsilon_H_val = 0.5 * (phi_prime / M_p)**2
    return epsilon_H_val - 1.0
event_epsilon_H_equals_one.terminal = True 
event_epsilon_H_equals_one.direction = 1 

@jit(nopython=True)
def event_V_too_small_or_large(N, y, C, mu, omega, chi, lam, kap): # Uses _solver version
    phi, phi_prime = y
    V_val = potential_V_solver(phi, C, mu, omega, chi, lam, kap)
    if V_val < 1e-20 : return -1.0 
    if V_val > 1e20 : return -1.0 
    return 1.0 
event_V_too_small_or_large.terminal = True

@jit(nopython=True)
def event_H_sq_denom_too_small(N, y, C, mu, omega, chi, lam, kap):
    phi, phi_prime = y
    return (3 * M_p**2 - 0.5 * phi_prime**2) - 1e-9 
event_H_sq_denom_too_small.terminal = True

BOUNDARY_EVENT_EPSILON_SOLVER = 1e-6 # Tighter for solver

@jit(nopython=True)
def event_theta_boundary_positive(N, y, C, mu, omega, chi, lam, kap):
    phi, phi_prime = y
    if C <= 1e-9: return 0.0 
    theta = C * phi / M_p
    return (np.pi/2.0 - BOUNDARY_EVENT_EPSILON_SOLVER) - theta 
event_theta_boundary_positive.terminal = True

@jit(nopython=True)
def event_theta_boundary_negative(N, y, C, mu, omega, chi, lam, kap):
    phi, phi_prime = y
    if C <= 1e-9: return 0.0
    theta = C * phi / M_p
    return theta - (-np.pi/2.0 + BOUNDARY_EVENT_EPSILON_SOLVER) 
event_theta_boundary_negative.terminal = True


# --- Main inflation solver function ---
def run_inflation_model(params_dict, theta_initial_norm, N_target_efolds=60.0, N_max_integration=200.0):
    C = params_dict['C']
    mu = params_dict['mu']
    omega = params_dict['omega']
    chi_param = params_dict['chi'] 
    lam_param = params_dict['lambda'] # Use lam_param to avoid keyword clash
    kap = params_dict['kappa']

    theta_initial = theta_initial_norm * (np.pi/2.0 - BOUNDARY_EVENT_EPSILON_SOLVER * 20) # Start further from boundary
    
    if C <= 1e-9: 
        return 1e30
        return {"status": "failure_C_too_small", "cost": 1e10}

    phi_initial = theta_initial * M_p / C

    # Use _solver versions for initial condition checks
    V_initial = potential_V_solver(phi_initial, C, mu, omega, chi_param, lam_param, kap)
    V_phi_initial = potential_V_phi_solver(phi_initial, C, mu, omega, chi_param, lam_param, kap)

    if V_initial >= 1e29 or np.abs(V_phi_initial) >= 1e29 : 
         return {"status": "failure_initial_V_boundary", "cost": 1e10}
    if V_initial <= 1e-20 or not np.isfinite(V_initial) or not np.isfinite(V_phi_initial):
        return {"status": "failure_initial_V_unphysical", "cost": 1e10}

    dphi_dN_initial = -M_p**2 * V_phi_initial / V_initial
    
    H_sq_denom_initial = (3 * M_p**2 - 0.5 * dphi_dN_initial**2)
    if H_sq_denom_initial <= 1e-9:
         return {"status": "failure_initial_H_sq_denom", "cost": 1e10}
    
    epsilon_H_initial = 0.5 * (dphi_dN_initial / M_p)**2
    if epsilon_H_initial >= 1.0:
        return {"status": "failure_initial_epsilon_H", "cost": 1e10}

    y0 = np.array([phi_initial, dphi_dN_initial])
    N_span = (0, N_max_integration)
    
    events_to_track = [event_epsilon_H_equals_one, event_V_too_small_or_large, 
                       event_H_sq_denom_too_small, event_theta_boundary_positive,
                       event_theta_boundary_negative]
    
    try:
        sol = solve_ivp(
            ode_system, N_span, y0,
            args=(C, mu, omega, chi_param, lam_param, kap), # Pass lam_param
            method='RK45', 
            dense_output=True,
            events=events_to_track,
            rtol=1e-7, atol=1e-9 
        )
    except Exception as e:
        return {"status": "failure_solve_ivp_exception", "cost": 1e10}

    if not sol.success:
        event_names = ["eps_H=1", "V_extreme", "H_sq_small", "theta_bnd_pos", "theta_bnd_neg"]
        termination_reason = f"solve_ivp_status_{sol.status}"
        if sol.t_events: 
            for i, t_event_list in enumerate(sol.t_events):
                if len(t_event_list) > 0:
                    termination_reason = f"event_{event_names[i]}"
                    break
        return {"status": f"failure_{termination_reason}", "cost": 1e10}

    N_events_eps_H = sol.t_events[0] 
    if len(N_events_eps_H) == 0:
        other_event_triggered = False
        event_names = ["eps_H=1", "V_extreme", "H_sq_small", "theta_bnd_pos", "theta_bnd_neg"] 
        termination_reason_no_eps_H = "unknown"
        for i in range(len(sol.t_events)): 
            if len(sol.t_events[i]) > 0:
                other_event_triggered = True
                termination_reason_no_eps_H = f"event_{event_names[i]}"
                break
        if other_event_triggered:
             return {"status": f"failure_no_SR_end_{termination_reason_no_eps_H}", "cost": 1e10}
        return {"status": "failure_no_SR_end_timeout", "cost": 1e10} 

    N_end = N_events_eps_H[0]
    
    if N_end < N_target_efolds: # Ensure N_end is reasonably positive too
        return {"status": "failure_N_end_too_small", "cost": 1e10}

    N_star = N_end - N_target_efolds
    if N_star < 0: 
        return {"status": "failure_N_star_negative", "cost": 1e10}

    try:
        phi_star, phi_prime_star = sol.sol(N_star)
    except ValueError as e: 
        if N_star < sol.t.min() or N_star > sol.t.max():
             return {"status": "failure_N_star_out_of_bounds", "cost": 1e10}
        return {"status": "failure_interpolation_N_star", "cost": 1e10}

    # Use plotting versions of V, V_phi, V_phi_phi for observables at N_star
    # as these are less prone to extreme boundary values if N_star is near a boundary.
    V_star = potential_V(phi_star, C, mu, omega, chi_param, lam_param, kap)
    V_phi_star = potential_V_phi(phi_star, C, mu, omega, chi_param, lam_param, kap)
    V_phi_phi_star = potential_V_phi_phi(phi_star, C, mu, omega, chi_param, lam_param, kap)

    # Check if any returned NaN (means phi_star was at a plotting boundary)
    if np.isnan(V_star) or np.isnan(V_phi_star) or np.isnan(V_phi_phi_star):
        return {"status": "failure_V_nan_at_N_star", "cost": 1e10}
    
    # Redundant check, as NaN above would catch it, but good for clarity
    if V_star <= 1e-20 or not np.isfinite(V_star) or not np.isfinite(V_phi_star) or not np.isfinite(V_phi_phi_star):
        return {"status": "failure_V_unphysical_at_N_star", "cost": 1e10}


    epsilon_V_star = (M_p**2 / 2.0) * (V_phi_star / V_star)**2
    eta_V_star = M_p**2 * (V_phi_phi_star / V_star)

    if not np.isfinite(epsilon_V_star) or not np.isfinite(eta_V_star):
        return {"status": "failure_SR_unphysical_at_N_star", "cost": 1e10}
        
    n_s_star = 1.0 - 6.0 * epsilon_V_star + 2.0 * eta_V_star
    r_star = 16.0 * epsilon_V_star
    
    cost = (n_s_star - 0.9649)**2 * 1000 
    
    results = {
        "status": "success",
        "cost": cost,
        "params": params_dict, # This already contains 'lambda' correctly
        "theta_initial_norm": theta_initial_norm, 
        "phi_initial": phi_initial,
        "dphi_dN_initial": dphi_dN_initial,
        "N_values": sol.t,
        "phi_values": sol.y[0],
        "phi_prime_values": sol.y[1],
        "N_end": N_end,
        "N_star": N_star,
        "phi_star": phi_star,
        "phi_prime_star": phi_prime_star,
        "V_star": V_star,
        "epsilon_V_star": epsilon_V_star,
        "eta_V_star": eta_V_star,
        "n_s_star": n_s_star,
        "r_star": r_star,
        "sol_dense": sol.sol 
    }
    return results

# --- Optuna Objective Function ---
best_result_for_callback = {"cost": float('inf'), "n_s": 0.0}

def objective(trial):
    global best_result_for_callback
    params = {
        'C': trial.suggest_float('C', 0.01, 1.0), 
        'mu': trial.suggest_float('mu', -10.0, 10.0),
        'omega': trial.suggest_float('omega', -10.0, 10.0),
        'chi': trial.suggest_float('chi', -10.0, 10.0),
        'lambda': trial.suggest_float('lambda', -10.0, 10.0), 
        'kappa': trial.suggest_float('kappa', -10.0, 10.0)
    }
    theta_initial_norm = trial.suggest_float('theta_initial_norm', -0.85, 0.85) # Even further from +/-1

    results = run_inflation_model(params, theta_initial_norm)
    
    if results["status"] == "success":
        if results["cost"] < best_result_for_callback["cost"]:
            best_result_for_callback["cost"] = results["cost"]
            best_result_for_callback["n_s"] = results["n_s_star"]
    
    return results['cost']

def callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f"Trial {trial.number} finished. Value: {trial.value:.4e}, Params: {trial.params}")
        print(f"    Current best n_s from successful runs: {best_result_for_callback['n_s']:.4f} (cost: {best_result_for_callback['cost']:.4e})")
        print(f"    Best value so far in study: {study.best_value:.4e}")

# --- Helper for Parameter Text Box ---
def get_param_text(params_dict_from_results, full_results):
    textstr = '\n'.join([
        r'$C=%.3f$' % params_dict_from_results['C'],
        r'$\mu=%.3f$' % params_dict_from_results['mu'],
        r'$\omega=%.3f$' % params_dict_from_results['omega'],
        r'$\chi=%.3f$' % params_dict_from_results['chi'],
        r'$\lambda=%.3f$' % params_dict_from_results['lambda'], 
        r'$\kappa=%.3f$' % params_dict_from_results['kappa'],
        r'$\theta_{i,norm}=%.3f$' % full_results['theta_initial_norm'],
        r'$n_s^*=%.4f$' % full_results['n_s_star'],
        r'$r^*=%.4f$' % full_results['r_star'],
        r'$N_{end}=%.1f$' % full_results['N_end']
    ])
    return textstr

# --- Plotting Functions ---
def create_plots(results, folder_name):
    if results['status'] != 'success':
        print("Cannot generate plots, simulation was not successful.")
        return

    params = results['params']
    C_best = params['C'] 
    # Unpack all params for potential functions
    mu_best, omega_best, chi_best, lam_best, kap_best = params['mu'], params['omega'], params['chi'], params['lambda'], params['kappa']


    param_text_for_plots = get_param_text(params, results)
    text_props = dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.75)

    sol_dense = results['sol_dense']
    N_plot_points = np.linspace(0, results['N_end'], 500) 
    phi_plot_points = sol_dense(N_plot_points)[0]
    phi_prime_plot_points = sol_dense(N_plot_points)[1]

    phi_phys_limit = np.pi * M_p / (2 * C_best) if C_best > 1e-9 else 10.0 # Default large range if C is tiny
    # Use THETA_PLOT_EPSILON for plotting ranges, consistent with potential functions
    phi_plot_min_phys = -phi_phys_limit + (THETA_PLOT_EPSILON / C_best if C_best > 1e-9 else 0.01)
    phi_plot_max_phys =  phi_phys_limit - (THETA_PLOT_EPSILON / C_best if C_best > 1e-9 else 0.01)

    if C_best <= 1e-9 or phi_plot_min_phys >= phi_plot_max_phys : # Handle very large C or tiny C
        phi_plot_min_phys = -10.0 # Fallback range for phi
        phi_plot_max_phys = 10.0
        if C_best > 1e-9 : # if C is large, range is small
            phi_plot_min_phys = -np.pi/(2*C_best) + 1e-5
            phi_plot_max_phys =  np.pi/(2*C_best) - 1e-5


    # 1. Potential V(phi)
    plt.figure(figsize=(8, 6))
    phi_V_range = np.linspace(phi_plot_min_phys, phi_plot_max_phys, 400)
    V_plot = np.array([potential_V(p, C_best, mu_best, omega_best, chi_best, lam_best, kap_best) for p in phi_V_range])
    V_plot_finite_indices = np.isfinite(V_plot)
    
    plt.plot(phi_V_range[V_plot_finite_indices] / M_p, V_plot[V_plot_finite_indices] / M_p**4)
    plt.scatter([results['phi_star']/M_p], [results['V_star']/M_p**4], color='red', label=f'$N_*$ ({results["N_star"]:.1f} e-folds from end)', zorder=5)
    plt.xlabel('$\tilde{\varphi} / M_p$')
    plt.ylabel('$V(\tilde{\varphi}) / M_p^4$')
    plt.title('Inflaton Potential $V(\tilde{\varphi})$')
    plt.grid(True)
    plt.legend()
    V_for_ylim = V_plot[V_plot_finite_indices]
    if len(V_for_ylim) > 0:
        min_V = np.nanmin(V_for_ylim)
        max_V = np.nanmax(V_for_ylim)
        plt.ylim(min(0, min_V / M_p**4 * 0.9) if min_V < 0 else min_V / M_p**4 * 0.8, 
                 max_V / M_p**4 * 1.2 if max_V > 0 else 0.1)
    else: # Fallback if all V_plot were NaN
        plt.ylim(-0.1, 0.1)
    ax = plt.gca(); ax.text(0.02, 0.98, param_text_for_plots, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=text_props)
    plt.savefig(os.path.join(folder_name, 'potential_V_phi.png')); plt.close()

    # 2. Field phi vs N 
    plt.figure(figsize=(8, 6))
    plt.plot(N_plot_points, phi_plot_points / M_p)
    plt.axvline(results['N_star'], color='r', linestyle='--', label=f'$N_* = {results["N_star"]:.2f}$')
    plt.axvline(results['N_end'], color='k', linestyle='--', label=f'$N_{{end}} = {results["N_end"]:.2f}$')
    plt.xlabel('Number of e-folds $N$'); plt.ylabel('$\tilde{\varphi}(N) / M_p$'); plt.title('Field Evolution $\tilde{\varphi}(N)$')
    plt.legend(); plt.grid(True)
    ax = plt.gca(); ax.text(0.98, 0.98, param_text_for_plots, transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=text_props)
    plt.savefig(os.path.join(folder_name, 'phi_vs_N.png')); plt.close()

    V_traj_dense =np.array([potential_V(p, C_best, mu_best, omega_best, chi_best, lam_best, kap_best) for p in phi_plot_points])
    V_phi_traj_dense = np.array([potential_V_phi(p, C_best, mu_best, omega_best, chi_best, lam_best, kap_best) for p in phi_plot_points])
    V_phiphi_traj_dense = np.array([potential_V_phi_phi(p, C_best, mu_best, omega_best, chi_best, lam_best, kap_best) for p in phi_plot_points])
    valid_V_indices_dense = np.isfinite(V_traj_dense) & np.isfinite(V_phi_traj_dense) & np.isfinite(V_phiphi_traj_dense) & (V_traj_dense > 1e-20)
    
    epsilon_V_traj_dense = np.full_like(phi_plot_points, np.nan)
    eta_V_traj_dense = np.full_like(phi_plot_points, np.nan)
    if np.any(valid_V_indices_dense):
        epsilon_V_traj_dense[valid_V_indices_dense] = (M_p**2 / 2.0) * (V_phi_traj_dense[valid_V_indices_dense] / V_traj_dense[valid_V_indices_dense])**2
        eta_V_traj_dense[valid_V_indices_dense] = M_p**2 * (V_phiphi_traj_dense[valid_V_indices_dense] / V_traj_dense[valid_V_indices_dense])

    # 3. epsilon_V, eta_V vs phi
    plt.figure(figsize=(8,6)); plt.plot(phi_plot_points[valid_V_indices_dense]/M_p, epsilon_V_traj_dense[valid_V_indices_dense], label='$\\epsilon_V(\\tilde{\\varphi})$')
    plt.plot(phi_plot_points[valid_V_indices_dense]/M_p, eta_V_traj_dense[valid_V_indices_dense], label='$\eta_V(\tilde{\varphi})$')
    plt.xlabel('$\tilde{\varphi} / M_p$'); plt.ylabel('Slow-roll Parameters'); plt.title('$\epsilon_V$ and $\eta_V$ vs. $\tilde{\varphi}$')
    plt.legend(); plt.grid(True); plt.ylim(-2.5, 2.5) 
    ax = plt.gca(); ax.text(0.02, 0.98, param_text_for_plots, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=text_props)
    plt.savefig(os.path.join(folder_name, 'epsilon_eta_vs_phi.png')); plt.close()

    # 4. epsilon_H, eta_V(phi(N)) vs N 
    epsilon_H_traj_dense = 0.5 * (phi_prime_plot_points / M_p)**2
    plt.figure(figsize=(8,6)); plt.plot(N_plot_points, epsilon_H_traj_dense, label='$\epsilon_H(N)$')
    plt.plot(N_plot_points[valid_V_indices_dense], eta_V_traj_dense[valid_V_indices_dense], label='$\eta_V(\tilde{\varphi}(N))$ (approx $\eta_H$)')
    plt.axvline(results['N_star'], color='r', linestyle='--', label=f'$N_*$'); plt.axvline(results['N_end'], color='k', linestyle='--', label=f'$N_{{end}}$')
    plt.xlabel('Number of e-folds $N$'); plt.ylabel('Slow-roll Parameters'); plt.title('$\epsilon_H(N)$ and $\eta_V(N)$ vs. $N$')
    plt.legend(); plt.grid(True); plt.ylim(-2.5, 2.5) 
    ax = plt.gca(); ax.text(0.98, 0.98, param_text_for_plots, transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=text_props)
    plt.savefig(os.path.join(folder_name, 'epsilon_eta_vs_N.png')); plt.close()

    # 5. n_s vs N 
    n_s_traj_dense = 1.0 - 6.0 * epsilon_V_traj_dense + 2.0 * eta_V_traj_dense
    plt.figure(figsize=(8,6)); plt.plot(N_plot_points[valid_V_indices_dense], n_s_traj_dense[valid_V_indices_dense])
    plt.scatter([results['N_star']], [results['n_s_star']], color='red', label=f'$n_s(N_*) = {results["n_s_star"]:.4f}$')
    plt.axhline(0.9649, color='gray', linestyle=':', label='Target $n_s = 0.9649$')
    plt.xlabel('Number of e-folds $N$'); plt.ylabel('$n_s(N)$'); plt.title('Spectral Tilt $n_s$ vs. $N$')
    plt.legend(); plt.grid(True)
    ax = plt.gca(); ax.text(0.98, 0.05, param_text_for_plots, transform=ax.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right', bbox=text_props)
    plt.savefig(os.path.join(folder_name, 'n_s_vs_N.png')); plt.close()

    # 6. r vs N 
    r_traj_dense = 16.0 * epsilon_V_traj_dense
    plt.figure(figsize=(8,6)); plt.plot(N_plot_points[valid_V_indices_dense], r_traj_dense[valid_V_indices_dense])
    plt.scatter([results['N_star']], [results['r_star']], color='red', label=f'$r(N_*) = {results["r_star"]:.4f}$')
    plt.xlabel('Number of e-folds $N$'); plt.ylabel('$r(N)$'); plt.title('Tensor-to-Scalar Ratio $r$ vs. $N$')
    plt.legend(); plt.grid(True)
    ax = plt.gca(); ax.text(0.98, 0.98, param_text_for_plots, transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=text_props)
    plt.savefig(os.path.join(folder_name, 'r_vs_N.png')); plt.close()
    
    H_sq_denom_traj_dense = (3 * M_p**2 - 0.5 * phi_prime_plot_points**2)
    valid_H_indices_actual_traj = (H_sq_denom_traj_dense > 1e-9) & valid_V_indices_dense # V_traj_dense is already filtered
    phi_dot_actual_traj = np.full_like(phi_plot_points, np.nan) 
    if np.any(valid_H_indices_actual_traj):
        H_sq_actual_traj_valid = V_traj_dense[valid_H_indices_actual_traj] / H_sq_denom_traj_dense[valid_H_indices_actual_traj]
        H_actual_traj_valid = np.sqrt(H_sq_actual_traj_valid)
        phi_dot_actual_traj[valid_H_indices_actual_traj] = H_actual_traj_valid * phi_prime_plot_points[valid_H_indices_actual_traj]

    # 7. Phase Space Portraits
    # Adaptive y-limits for phase space plots
    default_dot_phi_max = 0.5 # Default if trajectory data is bad
    if np.any(np.isfinite(phi_dot_actual_traj)) :
        dot_phi_max_adaptive = np.nanmax(np.abs(phi_dot_actual_traj)) * 1.5 
        if not np.isfinite(dot_phi_max_adaptive) or dot_phi_max_adaptive < 1e-3: # check if too small or nan
            dot_phi_max_adaptive = default_dot_phi_max
    else:
        dot_phi_max_adaptive = default_dot_phi_max
    
    phi_grid_stream = np.linspace(phi_plot_min_phys, phi_plot_max_phys, 50) 
    dot_phi_grid_stream = np.linspace(-dot_phi_max_adaptive, dot_phi_max_adaptive, 50)
    
    PHI_m, DOT_PHI_m = np.meshgrid(phi_grid_stream, dot_phi_grid_stream)
    U_phi, V_phi = np.zeros_like(PHI_m), np.zeros_like(PHI_m)

    for i in range(PHI_m.shape[0]):
        for j in range(PHI_m.shape[1]):
            phi_ij, phidot_ij = PHI_m[i,j], DOT_PHI_m[i,j]
            U_phi[i,j] = phidot_ij
            V_val = potential_V(phi_ij, C_best, mu_best, omega_best, chi_best, lam_best, kap_best)
            V_p_val = potential_V_phi(phi_ij, C_best, mu_best, omega_best, chi_best, lam_best, kap_best)
            if np.isnan(V_val) or np.isnan(V_p_val):
                V_phi[i,j] = np.nan; continue
            H_sq_num = V_val + 0.5 * phidot_ij**2
            if H_sq_num < 0 : H_sq_val = -1.0
            else: H_sq_val = H_sq_num / (3.0 * M_p**2)
            if H_sq_val <= 1e-12 : V_phi[i,j] = np.nan; continue
            H_val = np.sqrt(H_sq_val)
            V_phi[i,j] = -3 * H_val * phidot_ij - V_p_val

    speed_phi = np.sqrt(U_phi**2 + V_phi**2)
    log_speed_phi = np.log(speed_phi + 1e-12) 
    finite_log_speeds = log_speed_phi[np.isfinite(log_speed_phi)]
    l_phi, vmin_p, vmax_p = 50, np.nanmin(finite_log_speeds) if len(finite_log_speeds)>0 else 0, np.nanmax(finite_log_speeds) if len(finite_log_speeds)>0 else 1
    if len(finite_log_speeds) > 1:
        vmin_p = np.nanpercentile(finite_log_speeds, 5)
        vmax_p = np.nanpercentile(finite_log_speeds, 95)
        if vmax_p <= vmin_p : vmin_p, vmax_p = np.nanmin(finite_log_speeds), np.nanmax(finite_log_speeds)
        if vmax_p > vmin_p : l_phi = np.linspace(vmin_p, vmax_p, 50)

    fig, ax = plt.subplots(figsize=(10,8)); 
    cont_p = ax.contourf(PHI_m/M_p, DOT_PHI_m/M_p**2, log_speed_phi, levels=l_phi, cmap='viridis', alpha=0.7, vmin=vmin_p, vmax=vmax_p, extend='both')
    ax.streamplot(PHI_m/M_p, DOT_PHI_m/M_p**2, np.nan_to_num(U_phi), np.nan_to_num(V_phi), color='k', density=1.0, linewidth=0.8, arrowsize=0.8, broken_streamlines=False) # Adjusted density
    if np.any(np.isfinite(phi_dot_actual_traj)):
        ax.plot(phi_plot_points[np.isfinite(phi_dot_actual_traj)]/M_p, phi_dot_actual_traj[np.isfinite(phi_dot_actual_traj)]/M_p**2, 
                 color='white', lw=2.0, zorder=4) # Thicker white line for trajectory
        ax.scatter(phi_plot_points[np.isfinite(phi_dot_actual_traj)]/M_p, phi_dot_actual_traj[np.isfinite(phi_dot_actual_traj)]/M_p**2, 
                   c=N_plot_points[np.isfinite(phi_dot_actual_traj)], cmap='coolwarm', s=20, zorder=5, edgecolor='black', linewidth=0.3)

    ax.set_xlabel(r'$\tilde{\varphi} / M_p$'); ax.set_ylabel(r'$\dot{\tilde{\varphi}} / M_p^2$'); ax.set_title('Uncompactified Phase Portrait')
    ax.grid(True); ax.set_xlim(phi_plot_min_phys/M_p, phi_plot_max_phys/M_p); ax.set_ylim(-dot_phi_max_adaptive/M_p**2, dot_phi_max_adaptive/M_p**2)
    div_p = make_axes_locatable(ax); cax_p = div_p.append_axes("right",size="5%",pad=0.1); plt.colorbar(cont_p,cax=cax_p,label='Log Flow Speed')
    ax.text(0.02,0.98,param_text_for_plots,transform=ax.transAxes,fontsize=8,verticalalignment='top',bbox=text_props)
    plt.savefig(os.path.join(folder_name,'phase_space_uncompactified.png')); plt.close()

    # Compactified phase space
    theta_grid_stream = np.linspace(-np.pi/2 + THETA_PLOT_EPSILON*2, np.pi/2 - THETA_PLOT_EPSILON*2, 50)
    dot_theta_max_adaptive = (C_best/M_p) * dot_phi_max_adaptive if C_best > 1e-9 else 0.1
    dot_theta_grid_stream = np.linspace(-dot_theta_max_adaptive, dot_theta_max_adaptive, 50)
    THETA_m, DOT_THETA_m = np.meshgrid(theta_grid_stream, dot_theta_grid_stream)
    U_th, V_th = np.zeros_like(THETA_m), np.zeros_like(THETA_m)
    for i in range(THETA_m.shape[0]):
        for j in range(THETA_m.shape[1]):
            th_ij, thdot_ij = THETA_m[i,j], DOT_THETA_m[i,j]
            U_th[i,j] = thdot_ij
            phi_f_th = th_ij * M_p / C_best if C_best > 1e-9 else 0
            phidot_f_thdot = thdot_ij * M_p / C_best if C_best > 1e-9 else 0
            V_val = potential_V(phi_f_th, C_best, mu_best, omega_best, chi_best, lam_best, kap_best)
            V_p_val = potential_V_phi(phi_f_th, C_best, mu_best, omega_best, chi_best, lam_best, kap_best)
            if np.isnan(V_val) or np.isnan(V_p_val): V_th[i,j]=np.nan; continue
            H_sq_num = V_val + 0.5 * phidot_f_thdot**2
            if H_sq_num < 0 : H_sq_val = -1.0
            else: H_sq_val = H_sq_num / (3.0*M_p**2)
            if H_sq_val <= 1e-12 : V_th[i,j]=np.nan; continue
            H_val = np.sqrt(H_sq_val)
            phi_ddot = -3*H_val*phidot_f_thdot - V_p_val
            V_th[i,j] = (C_best/M_p) * phi_ddot if C_best > 1e-9 else 0
    speed_th = np.sqrt(U_th**2 + V_th**2); log_speed_th = np.log(speed_th + 1e-12)
    finite_log_speeds_th = log_speed_th[np.isfinite(log_speed_th)]
    l_th,vmin_t,vmax_t=50,np.nanmin(finite_log_speeds_th) if len(finite_log_speeds_th)>0 else 0, np.nanmax(finite_log_speeds_th) if len(finite_log_speeds_th)>0 else 1
    if len(finite_log_speeds_th) > 1:
        vmin_t = np.nanpercentile(finite_log_speeds_th, 5)
        vmax_t = np.nanpercentile(finite_log_speeds_th, 95)
        if vmax_t <= vmin_t : vmin_t,vmax_t = np.nanmin(finite_log_speeds_th),np.nanmax(finite_log_speeds_th)
        if vmax_t > vmin_t : l_th = np.linspace(vmin_t,vmax_t,50)
    
    fig_t, ax_t = plt.subplots(figsize=(10,8))
    cont_t = ax_t.contourf(THETA_m, DOT_THETA_m, log_speed_th, levels=l_th, cmap='viridis', alpha=0.7, vmin=vmin_t, vmax=vmax_t, extend='both')
    ax_t.streamplot(THETA_m, DOT_THETA_m, np.nan_to_num(U_th), np.nan_to_num(V_th), color='k', density=1.2, linewidth=0.8, arrowsize=0.8, broken_streamlines=False) # Adjusted density
    theta_traj = C_best * phi_plot_points / M_p
    theta_dot_traj = (C_best/M_p) * phi_dot_actual_traj
    valid_theta_traj = np.isfinite(theta_traj) & np.isfinite(theta_dot_traj)
    if np.any(valid_theta_traj):
        ax_t.plot(theta_traj[valid_theta_traj], theta_dot_traj[valid_theta_traj], color='white', lw=2.0, zorder=4)
        ax_t.scatter(theta_traj[valid_theta_traj], theta_dot_traj[valid_theta_traj], 
                     c=N_plot_points[valid_theta_traj], cmap='coolwarm', s=20, zorder=5, edgecolor='black', linewidth=0.3)
    ax_t.set_xlabel(r'$\theta = C \tilde{\varphi} / M_p$'); ax_t.set_ylabel(r'$\dot{\theta} = (C/M_p) \dot{\tilde{\varphi}} / M_p$'); ax_t.set_title('Compactified Phase Portrait')
    ax_t.grid(True); ax_t.set_xlim(-np.pi/2, np.pi/2); ax_t.set_ylim(-dot_theta_max_adaptive, dot_theta_max_adaptive)
    div_t = make_axes_locatable(ax_t); cax_t = div_t.append_axes("right",size="5%",pad=0.1); plt.colorbar(cont_t,cax=cax_t,label='Log Flow Speed')
    ax_t.text(0.02,0.98,param_text_for_plots,transform=ax_t.transAxes,fontsize=8,verticalalignment='top',bbox=text_props)
    plt.savefig(os.path.join(folder_name,'phase_space_compactified_theta.png')); plt.close()

    # 8. n_s vs r 
    plt.figure(figsize=(8,6)); valid_r_plot = valid_V_indices_dense & np.isfinite(r_traj_dense) & (r_traj_dense>=0) & (r_traj_dense<1.0) & np.isfinite(n_s_traj_dense)
    plt.plot(r_traj_dense[valid_r_plot], n_s_traj_dense[valid_r_plot])
    plt.scatter([results['r_star']], [results['n_s_star']], color='red', label=f'$(r_*, n_{{s*}}) = ({results["r_star"]:.4f}, {results["n_s_star"]:.4f})$', zorder=5)
    plt.axhline(0.9649,color='gray',linestyle=':',label='$n_s=0.9649$'); plt.axvline(0.056,color='lightgray',linestyle=':',label='$r<0.056$')
    plt.xlabel('Tensor-to-Scalar Ratio $r$'); plt.ylabel('Spectral Tilt $n_s$'); plt.title('$n_s$ vs. $r$ Trajectory')
    plt.legend(); plt.grid(True)
    plt.xlim(0, max(0.06, results['r_star']*1.5) if np.isfinite(results['r_star']) and results['r_star'] > 0 else 0.06)
    plt.ylim(0.90, max(1.0, results['n_s_star']*1.02) if np.isfinite(results['n_s_star']) and results['n_s_star'] > 0.9 else 1.0)
    ax = plt.gca(); ax.text(0.98,0.05,param_text_for_plots,transform=ax.transAxes,fontsize=8,verticalalignment='bottom',horizontalalignment='right',bbox=text_props)
    plt.savefig(os.path.join(folder_name, 'n_s_vs_r.png')); plt.close()
    print(f"Plots saved to folder: {folder_name}")

# --- Main Execution ---
if __name__ == '__main__':
    n_trials = 1000 
    study_name = "inflation_optimizer_study_v4" # New study name
    study = optuna.create_study(direction='minimize', study_name=study_name,
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50, multivariate=True, group=True, warn_independent_sampling=False), 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=30, interval_steps=5)
                               ) 
    print(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback]) 
    print("\nOptimization Finished.")
    if study.best_trial is None:
        print("No successful trials completed. Cannot proceed.")
    else:
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best value (cost): {study.best_value}")
        print("Best parameters found:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        best_params_dict = study.best_params.copy() # Optuna params dict directly
        best_theta_initial_norm = study.best_params['theta_initial_norm']
        print("\nRunning simulation with best parameters to generate final results and plots...")
        final_results = run_inflation_model(best_params_dict, best_theta_initial_norm, N_max_integration=400) 
        if final_results['status'] == 'success':
            print(r"\nFinal run successful:")
            print(r"  n_s(N_*) = {final_results['n_s_star']:.5f}")
            print(r"  r(N_*) = {final_results['r_star']:.5f}")
            print(r"  N_end = {final_results['N_end']:.2f} e-folds")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_folder = f"inflation_results_{timestamp}"
            os.makedirs(results_folder, exist_ok=True)
            with open(os.path.join(results_folder, 'optimal_parameters.txt'), 'w') as f:
                f.write("Optimal Inflaton Parameters:\n"); 
                for key, value in best_params_dict.items(): f.write(f"  {key}: {value}\n")
                # f.write(f"  theta_initial_norm: {best_theta_initial_norm}\n") # Already in best_params_dict
                f.write(r"\nDerived Observables at N_* = N_end - 60:\n")
                f.write(r"  n_s: {final_results['n_s_star']}\n"); f.write(f"  r: {final_results['r_star']}\n")
                f.write(r"  N_end: {final_results['N_end']}\n")
            with open(os.path.join(results_folder, 'optimal_parameters.tex'), 'w') as f:
                f.write("\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage[a4paper,margin=1in]{geometry}\n")
                f.write("\\title{Optimal Inflaton Parameters and Observables}\n\\date{\\today}\n\\author{Inflation Model Optimizer}\n")
                f.write("\\begin{document}\n\\maketitle\n\\section*{Optimized Potential Parameters}\n\\begin{itemize}\n")
                for key, value in best_params_dict.items():
                    param_name_tex = key.replace('_', '\\_') if '_' in key else key
                    f.write(f"  \\item $\\texttt{{{param_name_tex}}} = {value:.6f}$\n")
                f.write("\\end{itemize}\n\\section*{Derived Observables at $N_* = N_{{end}} - 60$ e-folds}\n\\begin{itemize}\n")
                f.write(r"  \\item Spectral tilt, $n_s = {final_results['n_s_star']:.6f}$\n")
                f.write(r"  \\item Tensor-to-scalar ratio, $r = {final_results['r_star']:.6f}$\n")
                f.write(r"  \\item Total e-folds of inflation, $N_{{end}} = {final_results['N_end']:.2f}$\n")
                f.write(r"  \\item E-folds before end of inflation for CMB pivot scale, $N_* = {final_results['N_star']:.2f}$\n")
                f.write("\\end{itemize}\n\\end{document}\n")
            print(f"Parameters saved to {results_folder}")
            create_plots(final_results, results_folder)
        else:
            print("\nFinal run with best parameters failed.")
            print(f"Status: {final_results['status']}\nCost: {final_results['cost']}")
