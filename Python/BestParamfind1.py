import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from scipy.misc import derivative # For numerical derivatives if analytical are too complex

# --- Constants and Target Values ---
N_TARGET = 60.0
NS_TARGET = 0.9626
MP = 1.0 # Reduced Planck mass, set to 1 for dimensionless calculations

# --- Parameter Names (for clarity) ---
# params = [alpha, gamma, beta, k, mu, omega, chi, lambda_val, kappa]
# Note: lambda is a keyword in Python, so using lambda_val

# --- Helper Functions for Potential and Derivatives ---

def get_X_phi(phi, A_const, C_const, beta, gamma):
    """ Calculates X(phi) """
    if beta == 0: # Avoid division by zero, though constraints should prevent this
        return np.inf # Or handle as an invalid case
    # Protect against tan(pi/2)
    arg_tan = A_const * phi
    if np.abs(arg_tan) >= np.pi/2 * 0.99999: # Stay away from exact singularity
        return np.sign(arg_tan) * 1e20 # Large value
    return (C_const * np.tan(arg_tan) - gamma) / (2 * beta)

def get_P_X(X, alpha, gamma, beta):
    """ Denominator polynomial P(X) = beta*X^2 + gamma*X + alpha """
    return beta * X**2 + gamma * X + alpha

def get_N_X(X, mu, omega, chi, lambda_val, kappa):
    """ Numerator polynomial N(X) """
    return kappa + 2*omega*X + mu**2 * X**2 + 2*chi*X**3 + lambda_val*X**4

def get_V_num_from_X(X, params_dict):
    """ Calculates V_num = N(X)/P(X)^2 from X """
    alpha = params_dict['alpha']
    gamma = params_dict['gamma']
    beta = params_dict['beta']
    mu = params_dict['mu']
    omega = params_dict['omega']
    chi = params_dict['chi']
    lambda_val = params_dict['lambda_val']
    kappa = params_dict['kappa']

    PX = get_P_X(X, alpha, gamma, beta)
    NX = get_N_X(X, mu, omega, chi, lambda_val, kappa)
    
    if PX == 0: return np.inf # Singularity
    return NX / (PX**2)

# --- Derivatives (Analytical) ---
# dX/dphi
def get_dX_dphi(X, A_const, C_const, alpha, gamma, beta):
    PX = get_P_X(X, alpha, gamma, beta)
    if C_const == 0: return np.inf # Should be caught by constraints
    return (2 * A_const / C_const) * PX * MP # MP is 1 here

# dP/dX
def get_dP_dX(X, gamma, beta):
    return 2 * beta * X + gamma

# dN/dX
def get_dN_dX(X, mu, omega, chi, lambda_val):
    return 2*omega + 2*mu**2 * X + 6*chi*X**2 + 4*lambda_val*X**3

# d2N/dX2
def get_d2N_dX2(X, mu, chi, lambda_val):
    return 2*mu**2 + 12*chi*X + 12*lambda_val*X**2

# dV_num/dX and d2V_num/dX2
def get_V_num_derivatives_X(X, params_dict):
    alpha = params_dict['alpha']
    gamma = params_dict['gamma']
    beta = params_dict['beta']
    mu = params_dict['mu']
    omega = params_dict['omega']
    chi = params_dict['chi']
    lambda_val = params_dict['lambda_val']
    kappa = params_dict['kappa']

    PX = get_P_X(X, alpha, gamma, beta)
    NX = get_N_X(X, mu, omega, chi, lambda_val, kappa)
    
    if PX == 0: # Singularity
        return np.nan, np.nan 

    PX_prime = get_dP_dX(X, gamma, beta)
    NX_prime = get_dN_dX(X, mu, omega, chi, lambda_val)
    
    # dV_num/dX
    V_num_X = (NX_prime * PX - 2 * NX * PX_prime) / (PX**3)

    # For d2V_num/dX2
    PX_double_prime = 2 * beta
    NX_double_prime = get_d2N_dX2(X, mu, chi, lambda_val)

    # Numerator of dV_num/dX: Q = NX_prime * PX - 2 * NX * PX_prime
    Q_prime = (NX_double_prime * PX + NX_prime * PX_prime) - \
              2 * (NX_prime * PX_prime + NX * PX_double_prime)
    
    V_num_XX = (Q_prime * PX**3 - (NX_prime * PX - 2 * NX * PX_prime) * 3 * PX**2 * PX_prime) / (PX**6)
    V_num_XX = (Q_prime * PX - 3 * (NX_prime * PX - 2 * NX * PX_prime) * PX_prime) / (PX**4)
    
    return V_num_X, V_num_XX

# V_num_phi and V_num_phiphi
def get_V_num_phi_derivatives(phi, params_dict, A_const, C_const):
    alpha = params_dict['alpha']
    gamma = params_dict['gamma']
    beta = params_dict['beta']
    
    X_val = get_X_phi(phi, A_const, C_const, beta, gamma)
    if not np.isfinite(X_val):
        return np.nan, np.nan, np.nan # V_num, V_num_phi, V_num_phiphi

    V_num = get_V_num_from_X(X_val, params_dict)
    if not np.isfinite(V_num) or V_num <= 0: # Potential must be positive for inflation
         return np.nan, np.nan, np.nan

    V_num_X, V_num_XX = get_V_num_derivatives_X(X_val, params_dict)
    if not np.isfinite(V_num_X) or not np.isfinite(V_num_XX):
        return V_num, np.nan, np.nan

    X_phi = get_dX_dphi(X_val, A_const, C_const, alpha, gamma, beta)
    if not np.isfinite(X_phi):
        return V_num, np.nan, np.nan
        
    PX = get_P_X(X_val, alpha, gamma, beta)
    PX_prime = get_dP_dX(X_val, gamma, beta)
    
    # d2X/dphi2 = (2A/C)^2 * P(X) * P'(X) (since MP=1)
    X_phiphi = (2 * A_const / C_const)**2 * PX * PX_prime if C_const != 0 else np.inf
    if not np.isfinite(X_phiphi):
         return V_num, np.nan, np.nan

    V_num_phi = V_num_X * X_phi
    V_num_phiphi = V_num_XX * (X_phi**2) + V_num_X * X_phiphi
    
    return V_num, V_num_phi, V_num_phiphi


# --- Slow Roll Functions ---
def get_epsilon_V(phi, params_dict, A_const, C_const):
    V_num, V_num_phi, _ = get_V_num_phi_derivatives(phi, params_dict, A_const, C_const)
    if not np.isfinite(V_num) or not np.isfinite(V_num_phi) or V_num == 0:
        return np.inf 
    return 0.5 * (V_num_phi / V_num)**2

def get_eta_V(phi, params_dict, A_const, C_const):
    V_num, _, V_num_phiphi = get_V_num_phi_derivatives(phi, params_dict, A_const, C_const)
    if not np.isfinite(V_num) or not np.isfinite(V_num_phiphi) or V_num == 0:
        return np.inf
    return V_num_phiphi / V_num

# Function to find phi_e (end of inflation)
def find_phi_e(params_dict, A_const, C_const, phi_limit):
    # Search for phi_e where epsilon_V = 1
    # Assuming phi decreases, V_num_phi < 0, search phi_e in (0, phi_limit)
    # Need to determine search range carefully based on potential shape
    # For now, let's try a broad positive range if V_num_phi is negative
    
    # Test direction of roll:
    # Evaluate V_num_phi at a test point, e.g., phi_limit * 0.1
    # This is simplified; a robust solution would analyze V_num_phi behavior
    _, V_num_phi_test, _ = get_V_num_phi_derivatives(phi_limit * 0.1, params_dict, A_const, C_const)
    
    if not np.isfinite(V_num_phi_test): return np.nan

    search_interval_phi_e = None
    if V_num_phi_test < 0: # phi decreases, search (eps_phi_e_min, phi_limit * 0.99)
        search_interval_phi_e = (1e-5, phi_limit * 0.99) 
    elif V_num_phi_test > 0: # phi increases, search (-phi_limit * 0.99, -1e-5)
        search_interval_phi_e = (-phi_limit * 0.99, -1e-5)
    else: # V_num_phi_test is zero or nan
        return np.nan

    try:
        # Check if epsilon_V - 1 changes sign in the interval
        f_low = get_epsilon_V(search_interval_phi_e[0], params_dict, A_const, C_const) - 1.0
        f_high = get_epsilon_V(search_interval_phi_e[1], params_dict, A_const, C_const) - 1.0
        if not (np.isfinite(f_low) and np.isfinite(f_high) and np.sign(f_low) != np.sign(f_high)):
            # Try a wider range or different strategy if no sign change
            # This can happen if epsilon_V never reaches 1 or always > 1
            # For simplicity, we'll assume it crosses. A production system needs more robustness.
            # print(f"Warning: Epsilon_V-1 does not bracket zero in {search_interval_phi_e}. Low: {f_low}, High: {f_high}")
            # Try a few points to see behavior of epsilon_V
            # test_phis = np.linspace(search_interval_phi_e[0], search_interval_phi_e[1], 10)
            # eps_vals = [get_epsilon_V(tp, params_dict, A_const, C_const) for tp in test_phis]
            # print(f"Epsilon_V at test points: {eps_vals}")
            return np.nan
            
        phi_e_sol = optimize.brentq(
            lambda p: get_epsilon_V(p, params_dict, A_const, C_const) - 1.0,
            search_interval_phi_e[0], search_interval_phi_e[1],
            xtol=1e-7, rtol=1e-7, maxiter=100
        )
        return phi_e_sol
    except ValueError: # brentq fails if no root or other issues
        # print("Brentq failed to find phi_e")
        return np.nan


# ODE for N: dphi/dN = sqrt(2*epsilon_V) (if V_phi < 0, phi decreases as N increases from end of inflation)
# Or dphi/dN = -sqrt(2*epsilon_V) (if V_phi > 0, phi increases)
def dphi_dN_ode(N_val, phi, params_dict, A_const, C_const, roll_direction_positive):
    eps_V = get_epsilon_V(phi, params_dict, A_const, C_const)
    if not np.isfinite(eps_V) or eps_V < 0: # eps_V should be positive
        return 0 # Stop integration
    
    term = np.sqrt(2 * eps_V)
    if not np.isfinite(term): return 0

    return term if roll_direction_positive else -term


# --- Objective Function for Optimization ---
def cost_function(params_array):
    params_dict = {
        'alpha': params_array[0], 'gamma': params_array[1], 'beta': params_array[2],
        'k': params_array[3], 'mu': params_array[4], 'omega': params_array[5],
        'chi': params_array[6], 'lambda_val': params_array[7], 'kappa': params_array[8]
    }

    # --- Parameter Constraints & Validity Checks ---
    alpha, gamma, beta, k = params_dict['alpha'], params_dict['gamma'], params_dict['beta'], params_dict['k']
    
    if k <= -12.0: return 1e10 # k > -12
    A_const_squared_factor = 12.0 + k
    if A_const_squared_factor <= 1e-9 : return 1e10 
    A_const = np.sqrt(2.0 / A_const_squared_factor) / MP # MP=1

    delta_sqrt_term = 4 * alpha * beta - gamma**2
    if delta_sqrt_term <= 1e-9: return 1e10 # 4*alpha*beta - gamma^2 > 0
    C_const = np.sqrt(delta_sqrt_term)

    # Ensure alpha and beta have same sign (conventionally positive)
    if alpha <= 1e-9 or beta <= 1e-9: return 1e10 # Let's assume alpha, beta > 0

    phi_limit = (np.pi / 2.0) / A_const if A_const > 0 else np.inf

    # --- Find phi_e (end of inflation) ---
    phi_e = find_phi_e(params_dict, A_const, C_const, phi_limit)
    if not np.isfinite(phi_e):
        # print("phi_e not found or invalid.")
        return 1e9
    
    # Determine roll direction for ODE integration
    # Test V_num_phi around where inflation might happen (e.g. away from phi_e if eps_V changes fast)
    # This is a heuristic. A more robust way is to check V_num_phi(phi_e - small_delta)
    # or V_num_phi(phi_e + small_delta) depending on which side phi_star is expected.
    # Let's assume phi_star is further from origin than phi_e for now.
    # If phi_e > 0, assume phi_star > phi_e. If phi_e < 0, assume phi_star < phi_e.
    
    _, V_num_phi_at_phi_e_test_offset, _ = get_V_num_phi_derivatives(
        phi_e * (1.0 + np.sign(phi_e) * 0.1) if phi_e != 0 else 0.1, # Test point slightly away from phi_e
        params_dict, A_const, C_const
    )

    if not np.isfinite(V_num_phi_at_phi_e_test_offset): return 1e8
    
    # If V_num_phi < 0, phi decreases. dphi/dN = -sqrt(2 eps_V) to go from N=0 (phi_e) to N=60 (phi_star)
    # If V_num_phi > 0, phi increases. dphi/dN = +sqrt(2 eps_V)
    roll_direction_positive = V_num_phi_at_phi_e_test_offset > 0

    # --- Integrate for N_TARGET e-folds to find phi_star ---
    # We integrate dphi/dN from N=0 (at phi_e) to N=N_TARGET
    # The sign depends on whether phi increases or decreases during inflation
    # To get phi_*, we integrate from (N=0, phi=phi_e) to (N=N_TARGET, phi=phi_*)
    
    # Span for N: from 0 to N_TARGET
    N_eval_points = [0, N_TARGET]
    
    try:
        sol = integrate.solve_ivp(
            dphi_dN_ode,
            [0, N_TARGET], # N_span
            [phi_e],      # Initial phi
            args=(params_dict, A_const, C_const, roll_direction_positive),
            dense_output=True,
            method='RK45', # or 'LSODA' for stiffness
            rtol=1e-6, atol=1e-8
        )
        if not sol.success or sol.status != 0:
            # print(f"ODE integration failed: {sol.message}")
            return 1e7
        phi_star = sol.y[0, -1] # phi at N_TARGET
    except Exception as e:
        # print(f"Exception during ODE solve: {e}")
        return 1e7

    if not np.isfinite(phi_star) or np.abs(phi_star) >= phi_limit * 0.999:
        # print(f"phi_star invalid or too close to limit: {phi_star}")
        return 1e6
        
    # --- Calculate Observables at phi_star ---
    eps_V_star = get_epsilon_V(phi_star, params_dict, A_const, C_const)
    eta_V_star = get_eta_V(phi_star, params_dict, A_const, C_const)

    if not np.isfinite(eps_V_star) or not np.isfinite(eta_V_star):
        # print("eps_V_star or eta_V_star is not finite.")
        return 1e5

    ns_calculated = 1.0 - 6.0 * eps_V_star + 2.0 * eta_V_star
    
    # Cost: squared difference from target ns
    cost = (ns_calculated - NS_TARGET)**2
    
    # Penalty for non-physical r (tensor-to-scalar ratio)
    r_calculated = 16.0 * eps_V_star
    if r_calculated < 0 or r_calculated > 1.0 : # Broad physical range
        cost += 100.0 
        
    # Ensure V_num(phi_star) is positive
    V_num_star, _, _ = get_V_num_phi_derivatives(phi_star, params_dict, A_const, C_const)
    if not np.isfinite(V_num_star) or V_num_star <= 0:
        cost += 1000.0

    # print(f"Params: {params_array}, ns_calc: {ns_calculated:.4f}, r_calc: {r_calculated:.4f}, Cost: {cost:.4e}")
    # print(f"phi_e: {phi_e:.4f}, phi_star: {phi_star:.4f}, eps_V*: {eps_V_star:.4e}, eta_V*: {eta_V_star:.4e}")

    return cost


# --- Optimization ---
if __name__ == "__main__":
    # Parameter bounds: [alpha, gamma, beta, k, mu, omega, chi, lambda_val, kappa]
    # These bounds are crucial and may need tuning.
    bounds = [
        (-10, 10.0),    # alpha > 0
        (-10.0, 10.0),   # gamma
        (-10, 10.0),    # beta > 0
        (-11.99, 50.0),  # k > -12
        (-10.0, 10.0),     # mu (mu^2 is used)
        (-10.0, 10.0),   # omega
        (-10.0, 10.0),   # chi
        (-10.0, 10.0),   # lambda_val
        (-50.0, 50.0)    # kappa (needs to ensure V_num > 0)
    ]
    # A common strategy for kappa is to ensure V_num is positive.
    # For example, if other terms are small, kappa might need to be positive.
    # Or, one could fix kappa=1 and scale other N(X) coefficients.

    print("Starting optimization... (this may take a while)")
    # Using differential_evolution for global optimization
    # For faster but potentially local results, 'L-BFGS-B' or 'Nelder-Mead' could be tried
    # but they are more sensitive to initial guesses.
    
    # To make it run faster for demonstration, reduce maxiter and popsize
    # For a real search, increase these significantly (e.g., maxiter=500-1000, popsize=15*len(bounds))
    result = optimize.differential_evolution(
        cost_function,
        bounds,
        strategy='best1bin',
        maxiter=100, # Reduced for speed in example (e.g. 500-1000 for real search)
        popsize=20,  # Reduced for speed (e.g. 15 * num_params)
        tol=1e-4,    # Tolerance for convergence
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True, # Display progress
        workers=-1 # Use all available CPU cores
    )

    print("\nOptimization finished.")
    print(f"Best parameters found: {result.x}")
    print(f"Minimum cost (ns error squared): {result.fun}")

    best_params_array = result.x
    best_params_dict = {
        'alpha': best_params_array[0], 'gamma': best_params_array[1], 'beta': best_params_array[2],
        'k': best_params_array[3], 'mu': best_params_array[4], 'omega': best_params_array[5],
        'chi': best_params_array[6], 'lambda_val': best_params_array[7], 'kappa': best_params_array[8]
    }

    # --- Recalculate and print observables with best parameters ---
    k_opt = best_params_dict['k']
    alpha_opt, gamma_opt, beta_opt = best_params_dict['alpha'], best_params_dict['gamma'], best_params_dict['beta']
    
    A_const_opt = np.sqrt(2.0 / (12.0 + k_opt)) / MP
    C_const_opt = np.sqrt(4 * alpha_opt * beta_opt - gamma_opt**2)
    phi_limit_opt = (np.pi / 2.0) / A_const_opt

    phi_e_opt = find_phi_e(best_params_dict, A_const_opt, C_const_opt, phi_limit_opt)
    
    if not np.isfinite(phi_e_opt):
        print("Could not re-calculate phi_e with optimal parameters. Plotting aborted.")
    else:
        print(f"Optimal phi_e: {phi_e_opt}")
        _, V_num_phi_at_phi_e_test_offset_opt, _ = get_V_num_phi_derivatives(
             phi_e_opt * (1.0 + np.sign(phi_e_opt) * 0.1) if phi_e_opt != 0 else 0.1,
             best_params_dict, A_const_opt, C_const_opt
        )
        roll_direction_positive_opt = V_num_phi_at_phi_e_test_offset_opt > 0


        sol_opt = integrate.solve_ivp(
            dphi_dN_ode, [0, N_TARGET], [phi_e_opt],
            args=(best_params_dict, A_const_opt, C_const_opt, roll_direction_positive_opt),
            dense_output=True, method='RK45', rtol=1e-7, atol=1e-9
        )
        phi_star_opt = sol_opt.y[0, -1]
        
        eps_V_star_opt = get_epsilon_V(phi_star_opt, best_params_dict, A_const_opt, C_const_opt)
        eta_V_star_opt = get_eta_V(phi_star_opt, best_params_dict, A_const_opt, C_const_opt)
        ns_final = 1.0 - 6.0 * eps_V_star_opt + 2.0 * eta_V_star_opt
        r_final = 16.0 * eps_V_star_opt

        print(f"Optimal phi_star: {phi_star_opt}")
        print(f"Calculated ns with optimal parameters: {ns_final:.5f} (Target: {NS_TARGET})")
        print(f"Calculated r with optimal parameters: {r_final:.5f}")

        # --- Plot V(phi) ---
        # Plot V(tilde_phi) = (MP^4/8) * V_num(tilde_phi/MP)
        # With MP=1, this is (1/8) * V_num(phi)
        
        phi_plot_min = -phi_limit_opt * 2
        phi_plot_max = phi_limit_opt * 2
        
        # Ensure phi_star and phi_e are within the plot range, extend if necessary
        # This simple extension might not be ideal if they are very far out.
        if np.isfinite(phi_star_opt):
             phi_plot_min = min(phi_plot_min, phi_star_opt - 0.1*abs(phi_star_opt) if phi_star_opt!=0 else -1)
             phi_plot_max = max(phi_plot_max, phi_star_opt + 0.1*abs(phi_star_opt) if phi_star_opt!=0 else 1)
        if np.isfinite(phi_e_opt):
             phi_plot_min = min(phi_plot_min, phi_e_opt - 0.1*abs(phi_e_opt) if phi_e_opt!=0 else -1)
             phi_plot_max = max(phi_plot_max, phi_e_opt + 0.1*abs(phi_e_opt) if phi_e_opt!=0 else 1)
        
        # Ensure min < max
        if phi_plot_min >= phi_plot_max:
            phi_plot_min = -2 * phi_limit_opt # Fallback
            phi_plot_max = 2 * phi_limit_opt

        phi_values_plot = np.linspace(phi_plot_min, phi_plot_max, 4000)
        
        V_plot = []
        for p_val in phi_values_plot:
            X_p = get_X_phi(p_val, A_const_opt, C_const_opt, beta_opt, gamma_opt)
            if not np.isfinite(X_p):
                V_plot.append(np.nan)
                continue
            V_num_p = get_V_num_from_X(X_p, best_params_dict)
            V_plot.append((MP**4 / 8.0) * V_num_p if np.isfinite(V_num_p) else np.nan)
        
        V_plot = np.array(V_plot)

        plt.figure(figsize=(10, 6))
        plt.plot(phi_values_plot, V_plot, label=r'$V(\tilde{\varphi})$')
        
        # Mark phi_star and phi_e on the plot if they are valid
        if np.isfinite(phi_star_opt) and np.isfinite(phi_e_opt):
            V_at_phi_star = (MP**4 / 8.0) * get_V_num_from_X(get_X_phi(phi_star_opt, A_const_opt, C_const_opt, beta_opt, gamma_opt), best_params_dict)
            V_at_phi_e = (MP**4 / 8.0) * get_V_num_from_X(get_X_phi(phi_e_opt, A_const_opt, C_const_opt, beta_opt, gamma_opt), best_params_dict)
            if np.isfinite(V_at_phi_star):
                 plt.scatter([phi_star_opt], [V_at_phi_star], color='red', s=50, zorder=5, label=fr'$\tilde{{\varphi}}_*$ (N={N_TARGET})')
            if np.isfinite(V_at_phi_e):
                 plt.scatter([phi_e_opt], [V_at_phi_e], color='green', s=50, zorder=5, label=r'$\tilde{\varphi}_e$ (end of inflation)')

        plt.xlabel(r'$\tilde{\varphi}/M_p$ (assuming $M_p=1$)')
        plt.ylabel(r'$V(\tilde{\varphi}) / M_p^4$ (assuming $M_p=1$, factor 1/8 included)')
        plt.title('Inflationary Potential $V(\\tilde{\\varphi})$ with Optimized Parameters')
        
        # Determine y-axis limits, avoiding extreme values if V_plot has NaNs or Infs
        valid_V_plot = V_plot[np.isfinite(V_plot)]
        if len(valid_V_plot) > 0:
            ymin = np.min(valid_V_plot)
            ymax = np.max(valid_V_plot)
            yrange = ymax - ymin
            if yrange == 0: yrange = 1.0 # Avoid zero range
            plt.ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        else:
            plt.ylim(-1,1) # Fallback if no valid points

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

"""Optimization finished.
Best parameters found: [ 6.0952918  -1.88700513  9.39810384 45.84900061 -5.78937483 -5.38642911
  9.46398675  2.05157819 17.65923743]
  Parameter bounds: [alpha, gamma, beta, k, mu, omega, chi, lambda_val, kappa]
Minimum cost (ns error squared): 2.95148064284017e-10
Optimal phi_e: 8.028341751572691
Optimal phi_star: 0.7295265035938892
Calculated ns with optimal parameters: 0.96258 (Target: 0.9626)
Calculated r with optimal parameters: 0.01978

SECOND Run

Optimization finished.
Best parameters found: [ 7.60216141  0.38357323  4.57708382 32.07011198  0.73457071  5.80179047
  9.60123708  1.84293688 13.79875193]
Minimum cost (ns error squared): 8.292967678094659e-19
Optimal phi_e: -1.5327497888960078
Optimal phi_star: 3.4703525887326165
Calculated ns with optimal parameters: 0.96260 (Target: 0.9626)
Calculated r with optimal parameters: 0.02794

THIRD RUN

Optimization finished.
Best parameters found: [ 5.37904158 -1.2378985   4.84572706 46.47655135  7.12605145 -7.74347196
  9.67061868  0.31666247 42.92091333]
Minimum cost (ns error squared): 8.173274220072592e-14
Optimal phi_e: 7.685066100932695
Optimal phi_star: 1.0121415107832608
Calculated ns with optimal parameters: 0.96260 (Target: 0.9626)
Calculated r with optimal parameters: 0.00797
The thread 'MainThread' (1) has exited with code 0 (0x0).
The program 'python.exe' has exited with code 4294967295 (0xffffffff).

"""