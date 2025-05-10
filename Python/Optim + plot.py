import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormap module

# --- Constants and Target Values ---
N_TARGET = 60.0
NS_TARGET = 0.9626
MP = 1.0 # Reduced Planck mass, set to 1 for dimensionless calculations

# --- Parameter Names (for clarity) ---
# params = [alpha, gamma, beta, k, mu, omega, chi, lambda_val, kappa]

# --- Helper Functions for Potential and Derivatives ---

def get_X_phi(phi, A_const, C_const, beta, gamma):
    """ Calculates X(phi) """
    if beta == 0:
        return np.inf
    arg_tan = A_const * phi
    # Protect against tan(pi/2) by staying slightly away from exact singularity
    # Use a small epsilon relative to pi/2
    epsilon_singularity = 1e-6
    if np.abs(arg_tan) >= (np.pi/2 * (1.0 - epsilon_singularity)):
        # Return a very large number with the correct sign
        return np.sign(arg_tan) * 1e20
    # Normal case
    return (C_const * np.tan(arg_tan) - gamma) / (2 * beta)

def get_P_X(X, alpha, gamma, beta):
    """ Denominator polynomial P(X) = beta*X^2 + gamma*X + alpha """
    return beta * X**2 + gamma * X + alpha

def get_N_X(X, mu, omega, chi, lambda_val, kappa):
    """ Numerator polynomial N(X) """
    # mu is squared in the potential, so use mu_sq for clarity if mu is the parameter
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

    # Handle singularity in potential
    if PX == 0 or not np.isfinite(PX):
        return np.inf
    # Ensure potential is not excessively large due to numerical issues near singularity
    val = NX / (PX**2)
    if np.abs(val) > 1e30: # Cap large values that might arise near PX=0
        return np.sign(val) * 1e30
    return val

# --- Derivatives (Analytical) ---
def get_dX_dphi(X, A_const, C_const, alpha, gamma, beta):
    """ Calculates dX/dphi """
    PX = get_P_X(X, alpha, gamma, beta)
    if C_const == 0: return np.inf
    # Formula derived from the relation between X and phi:
    # X = (C_const * tan(A_const*phi) - gamma) / (2*beta)
    # dX/dphi = (C_const * A_const / (2*beta)) * sec^2(A_const*phi)
    # Using P(X) = (C_const^2 / (4*beta)) * sec^2(A_const*phi), we get:
    # dX/dphi = (2 * A_const / C_const) * P(X) * MP (with MP=1)
    return (2 * A_const / C_const) * PX * MP

def get_dP_dX(X, gamma, beta):
    """ Derivative of P(X) with respect to X """
    return 2 * beta * X + gamma

def get_dN_dX(X, mu, omega, chi, lambda_val):
    """ Derivative of N(X) with respect to X """
    return 2*omega + 2*mu**2 * X + 6*chi*X**2 + 4*lambda_val*X**3

def get_d2N_dX2(X, mu, chi, lambda_val):
    """ Second derivative of N(X) with respect to X """
    return 2*mu**2 + 12*chi*X + 12*lambda_val*X**2

def get_V_num_derivatives_X(X, params_dict):
    """ Calculates d(V_num)/dX and d^2(V_num)/dX^2 """
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

    if PX == 0 or not np.isfinite(PX):
        return np.nan, np.nan

    PX_prime = get_dP_dX(X, gamma, beta)
    NX_prime = get_dN_dX(X, mu, omega, chi, lambda_val)

    # First derivative of V_num w.r.t X: d/dX [N(X)/P(X)^2]
    # = [N'(X)P(X)^2 - N(X)*2*P(X)*P'(X)] / P(X)^4
    # = [N'(X)P(X) - 2*N(X)*P'(X)] / P(X)^3
    V_num_X = (NX_prime * PX - 2 * NX * PX_prime) / (PX**3)

    # Second derivative of V_num w.r.t X: d/dX [ (N'(X)P(X) - 2*N(X)*P'(X)) / P(X)^3 ]
    # Let Q(X) = N'(X)P(X) - 2*N(X)*P'(X)
    # V_num_XX = [Q'(X)P(X)^3 - Q(X)*3*P(X)^2*P'(X)] / P(X)^6
    # = [Q'(X)P(X) - 3*Q(X)*P'(X)] / P(X)^4
    # Q'(X) = (N''(X)P(X) + N'(X)P'(X)) - 2*(N'(X)P'(X) + N(X)P''(X))
    # Q'(X) = N''(X)P(X) - N'(X)P'(X) - 2*N(X)P''(X)
    NX_double_prime = get_d2N_dX2(X, mu, chi, lambda_val)
    PX_double_prime = 2 * beta

    Q_num = NX_prime * PX - 2 * NX * PX_prime # Numerator of V_num_X * PX^3
    Q_num_prime_X = (NX_double_prime * PX + NX_prime * PX_prime) - \
                    2 * (NX_prime * PX_prime + NX * PX_double_prime)

    V_num_XX = (Q_num_prime_X * PX - 3 * Q_num * PX_prime) / (PX**4)

    return V_num_X, V_num_XX

def get_V_num_phi_derivatives(phi, params_dict, A_const, C_const):
    """ Calculates V_num, d(V_num)/dphi, and d^2(V_num)/dphi^2 """
    alpha = params_dict['alpha']
    gamma = params_dict['gamma']
    beta = params_dict['beta']

    X_val = get_X_phi(phi, A_const, C_const, beta, gamma)
    if not np.isfinite(X_val):
        return np.nan, np.nan, np.nan

    V_num = get_V_num_from_X(X_val, params_dict)
    # Potential must be positive and finite for physical inflation calculations
    if not np.isfinite(V_num) or V_num <= 1e-30: # Using a small positive threshold
          return np.nan, np.nan, np.nan

    V_num_X, V_num_XX = get_V_num_derivatives_X(X_val, params_dict)
    if not (np.isfinite(V_num_X) and np.isfinite(V_num_XX)):
        return V_num, np.nan, np.nan

    X_phi = get_dX_dphi(X_val, A_const, C_const, alpha, gamma, beta)
    if not np.isfinite(X_phi):
        return V_num, np.nan, np.nan

    # d2X/dphi2 = d/dphi(X_phi) = d/dphi [ (2 * A_const / C_const) * P(X) ]
    # = (2 * A_const / C_const) * dP(X)/dphi
    # dP(X)/dphi = dP/dX * dX/dphi = P'(X) * X_phi
    PX_prime = get_dP_dX(X_val, gamma, beta)
    X_phiphi = (2 * A_const / C_const) * PX_prime * X_phi
    if not np.isfinite(X_phiphi):
          return V_num, V_num_X * X_phi if np.isfinite(V_num_X * X_phi) else np.nan, np.nan # Return V_num and V_num_phi if calculable

    # Chain rule for derivatives w.r.t phi
    V_num_phi = V_num_X * X_phi
    V_num_phiphi = V_num_XX * (X_phi**2) + V_num_X * X_phiphi

    return V_num, V_num_phi, V_num_phiphi

# --- Slow Roll Functions ---
def get_epsilon_V(phi, params_dict, A_const, C_const):
    """ Calculates the first slow-roll parameter epsilon_V """
    V_num, V_num_phi, _ = get_V_num_phi_derivatives(phi, params_dict, A_const, C_const)
    # V_num must be positive and finite, V_num_phi must be finite
    if not (np.isfinite(V_num) and V_num > 1e-30 and np.isfinite(V_num_phi)):
        return np.inf
    return 0.5 * (V_num_phi / V_num)**2 * MP**2 # MP=1, so factor is 1

def get_eta_V(phi, params_dict, A_const, C_const):
    """ Calculates the second slow-roll parameter eta_V """
    V_num, _, V_num_phiphi = get_V_num_phi_derivatives(phi, params_dict, A_const, C_const)
    # V_num must be positive and finite, V_num_phiphi must be finite
    if not (np.isfinite(V_num) and V_num > 1e-30 and np.isfinite(V_num_phiphi)):
        return np.inf
    return (V_num_phiphi / V_num) * MP**2 # MP=1, so factor is 1

def find_phi_e(params_dict, A_const, C_const, phi_limit, initial_guess_phi=None):
    """ Finds the field value phi_e where inflation ends (epsilon_V = 1) """
    # phi_limit is the boundary of the tan function: pi/(2*A_const)
    # Search for phi_e where epsilon_V = 1.
    # We need to determine if phi rolls left or right to set search interval.
    # Test V_num_phi at a point, e.g., phi_limit * 0.5 or -phi_limit * 0.5

    # Define the function to find the root for: epsilon_V(phi) - 1 = 0
    def epsilon_minus_one(phi):
        eps = get_epsilon_V(phi, params_dict, A_const, C_const)
        return eps - 1.0 if np.isfinite(eps) else np.inf # Return inf if epsilon is not finite

    phi_e_candidates = []

    # Try searching in both positive and negative phi ranges, away from origin and singularity
    search_ranges = [(1e-4 * phi_limit, 0.999 * phi_limit), (-0.999 * phi_limit, -1e-4 * phi_limit)]

    for search_interval in search_ranges:
        try:
            # Check if epsilon_V - 1 changes sign across the interval
            f_low = epsilon_minus_one(search_interval[0])
            f_high = epsilon_minus_one(search_interval[1])

            if np.isfinite(f_low) and np.isfinite(f_high) and np.sign(f_low) != np.sign(f_high):
                # Use brentq if a sign change is detected
                phi_e_sol = optimize.brentq(
                    epsilon_minus_one,
                    search_interval[0], search_interval[1],
                    xtol=1e-7, rtol=1e-7, maxiter=100
                )
                phi_e_candidates.append(phi_e_sol)
        except (ValueError, RuntimeError):
            # brentq failed (e.g., no root found, or function evaluation failed)
            pass

    if not phi_e_candidates:
        return np.nan # No valid phi_e found

    # If multiple candidates, return the one closest to the initial guess if provided,
    # otherwise return the first one found.
    if initial_guess_phi is not None and phi_e_candidates:
        phi_e_candidates.sort(key=lambda x: abs(x - initial_guess_phi))

    return phi_e_candidates[0] if phi_e_candidates else np.nan


def dphi_dN_ode(N_val, phi, params_dict, A_const, C_const, V_phi_sign_for_roll):
    """ ODE for phi as a function of N (e-folds) under slow roll """
    # N is e-folds *remaining*. As N increases from 0 (end) to N_target (start),
    # phi moves from phi_e "up" the potential (backwards in time).
    # dphi/dN = - MP^2 * V_phi / V (from slow roll equations)
    # Also, dphi/dN = - sqrt(2 * epsilon_V) * MP (using epsilon_V definition)
    # The sign depends on the roll direction. If V_phi < 0, phi increases naturally (rolls right).
    # To go backwards in time (increasing N), phi must decrease, so dphi/dN < 0.
    # If V_phi > 0, phi decreases naturally (rolls left). To go backwards, phi must increase, so dphi/dN > 0.
    # So, dphi/dN has the opposite sign of V_phi in forward time, which is the same sign as
    # V_phi_sign_for_roll for integration backwards in N.

    eps_V = get_epsilon_V(phi, params_dict, A_const, C_const)
    # Stop integration if eps_V is problematic (non-finite or too small/negative which shouldn't happen with proper V_num check)
    if not np.isfinite(eps_V) or eps_V <= 1e-15: # Use a very small positive threshold
        return 0

    term = MP * np.sqrt(2 * eps_V) # MP=1
    if not np.isfinite(term):
        return 0 # Stop integration

    # dphi/dN should have the same sign as V_phi_sign_for_roll for backwards integration
    return term if V_phi_sign_for_roll > 0 else -term


def cost_function(params_array):
    """ Cost function to minimize for parameter optimization """
    params_dict = {
        'alpha': params_array[0], 'gamma': params_array[1], 'beta': params_array[2],
        'k': params_array[3], 'mu': params_array[4], 'omega': params_array[5],
        'chi': params_array[6], 'lambda_val': params_array[7], 'kappa': params_array[8]
    }

    alpha, gamma, beta, k = params_dict['alpha'], params_dict['gamma'], params_dict['beta'], params_dict['k']

    # Parameter constraints based on the structure of the potential
    # Ensure terms under square roots are positive
    if alpha <= 1e-9 or beta <= 1e-9: return 1e10 # alpha and beta in denominator of A_const_squared_factor and C_const
    if (12.0 + k) <= 1e-9: return 1e10 # Term in A_const_squared
    if (4 * alpha * beta - gamma**2) <= 1e-9: return 1e10 # Term in C_const

    A_const = np.sqrt(2.0 / (12.0 + k)) / MP
    C_const = np.sqrt(4 * alpha * beta - gamma**2)

    phi_limit = (np.pi / 2.0) / A_const if A_const > 1e-9 else np.inf
    if not np.isfinite(phi_limit): return 1e10

    # --- Find phi_e (end of inflation) ---
    # Provide a rough guess for phi_e. Try both positive and negative sides.
    # A common scenario is inflation ending at positive phi if V_phi < 0 for phi>0, or negative phi if V_phi > 0 for phi<0.
    # Let's try a guess on the positive side first.
    phi_e = find_phi_e(params_dict, A_const, C_const, phi_limit, initial_guess_phi = phi_limit*0.8)
    # If positive search fails, try negative side
    if not np.isfinite(phi_e) or np.abs(phi_e) >= phi_limit * 0.9999:
         phi_e = find_phi_e(params_dict, A_const, C_const, phi_limit, initial_guess_phi = -phi_limit*0.8)

    if not np.isfinite(phi_e) or np.abs(phi_e) >= phi_limit * 0.9999:
        return 1e9 # Penalize if phi_e is not found or too close to singularity

    # Determine V_phi sign for roll direction during inflation.
    # Inflation happens "before" phi_e in time (larger N).
    # If phi_e > 0, inflation typically happens at smaller positive phi.
    # If phi_e < 0, inflation typically happens at larger negative phi (smaller magnitude).
    # Test V_phi slightly "before" phi_e along the presumed path towards the origin.
    test_phi_for_Vphi = phi_e * 0.95 # A point before phi_e along the presumed path towards origin
    if abs(test_phi_for_Vphi) < 1e-5: # Avoid testing too close to zero
        test_phi_for_Vphi = np.sign(phi_e) * 1e-5 if phi_e != 0 else 0.1 * phi_limit # Use a small value or fraction of limit

    _, V_num_phi_at_test, _ = get_V_num_phi_derivatives(test_phi_for_Vphi, params_dict, A_const, C_const)

    if not np.isfinite(V_num_phi_at_test): return 1e8 # Penalize if V_phi is not finite at test point

    # V_phi_sign_for_roll: positive if V_phi > 0 (rolls left), negative if V_phi < 0 (rolls right)
    # This is the sign of dphi/dt in forward time.
    # For backwards integration in N (from phi_e to phi_star), dphi/dN has the same sign as dphi/dt.
    V_phi_sign_for_roll = np.sign(V_num_phi_at_test)
    if V_phi_sign_for_roll == 0: return 1e8 # Penalize if potential is flat at test point

    # Integrate for N_TARGET e-folds backwards from phi_e
    try:
        # Adjust starting point slightly if phi_e is exactly at a numerical boundary
        phi_e_eff = phi_e
        if np.isfinite(phi_limit) and abs(phi_e_eff/phi_limit) > 0.999:
             phi_e_eff = np.sign(phi_e_eff) * phi_limit * 0.999

        sol = integrate.solve_ivp(
            dphi_dN_ode, [0, N_TARGET], [phi_e_eff], # Integrate from N=0 (phi_e) to N=N_TARGET (phi_star)
            args=(params_dict, A_const, C_const, V_phi_sign_for_roll),
            dense_output=True, method='RK45',
            rtol=1e-7, atol=1e-9 # Tightened tolerances
        )
        # Check if integration was successful
        if not sol.success or sol.status != 0:
            # print(f"Integration failed: {sol.message}")
            return 1e7 # Penalize failed integration

        phi_star = sol.y[0, -1] # Field value at N = N_TARGET
    except Exception as e:
        # print(f"Integration threw exception: {e}")
        return 1e7 # Penalize if integration throws an exception

    # Penalize if phi_star is not finite or too close to singularity
    if not np.isfinite(phi_star) or (np.isfinite(phi_limit) and np.abs(phi_star) >= phi_limit * 0.9999):
        return 1e6

    # --- Calculate Observables at phi_star ---
    eps_V_star = get_epsilon_V(phi_star, params_dict, A_const, C_const)
    eta_V_star = get_eta_V(phi_star, params_dict, A_const, C_const)

    # Penalize if slow-roll parameters are not finite at phi_star
    if not (np.isfinite(eps_V_star) and np.isfinite(eta_V_star)):
        return 1e5

    # Calculate ns
    ns_calculated = 1.0 - 6.0 * eps_V_star + 2.0 * eta_V_star

    # Calculate r (tensor-to-scalar ratio)
    r_calculated = 16.0 * eps_V_star

    # --- Cost Calculation ---
    # Primary cost: deviation from target ns
    cost = (ns_calculated - NS_TARGET)**2

    # Additional penalties for physical validity and slow-roll behavior at phi_star
    # Penalize if r is non-physical (negative or excessively large)
    if not (np.isfinite(r_calculated) and r_calculated > 0 and r_calculated < 1.0): # r typically < 0.1, but allow up to 1.0 as a soft constraint
         cost += 10.0 # Moderate penalty

    # Penalize if potential at phi_star is not positive and finite
    V_num_star, V_num_phi_star, _ = get_V_num_phi_derivatives(phi_star, params_dict, A_const, C_const)
    if not (np.isfinite(V_num_star) and V_num_star > 1e-30):
        cost += 100.0 # Significant penalty

    # Penalize if slow-roll conditions are not reasonably met at phi_star
    # Epsilon_V should be small, Eta_V should be small in magnitude
    if eps_V_star > 0.5 or abs(eta_V_star) > 0.5: # Thresholds for slow-roll validity
         cost += 50.0 # Moderate penalty for not being in slow-roll regime

    # Sanity check: V_num_phi_star should have the expected sign for rolling towards phi_e
    # This checks consistency between the assumed roll direction and the potential slope at phi_star
    if np.isfinite(V_num_phi_star) and V_phi_sign_for_roll != 0 and np.sign(V_num_phi_star) != V_phi_sign_for_roll:
         cost += 10.0 # Penalize if roll direction is inconsistent

    # Optional: print progress during optimization
    # print(f"Params: {params_array[:4]}..., ns: {ns_calculated:.4f}, r: {r_calculated:.4f}, cost: {cost:.2e}, phi_e: {phi_e:.2f}, phi*: {phi_star:.2f}, eps*: {eps_V_star:.2f}, eta*: {eta_V_star:.2f}")

    return cost

# --- Phase Portrait ODE system ---
def phase_portrait_odes(t, y, params_dict_plot, A_const_plot, C_const_plot):
    """ System of ODEs for phase portrait (phi, phi_dot) """
    phi, phi_dot = y[0], y[1]

    # Avoid evaluating too close to singularity
    if np.isfinite(A_const_plot) and abs(A_const_plot * phi) >= (np.pi/2 * 0.9999):
         return [0, 0] # Treat as a boundary/wall

    V_num, V_num_phi, _ = get_V_num_phi_derivatives(phi, params_dict_plot, A_const_plot, C_const_plot)

    # If potential or derivative is invalid, stop the trajectory
    if not (np.isfinite(V_num) and V_num > 1e-30 and np.isfinite(V_num_phi)):
        return [0, 0]

    # Equations of motion: phi_ddot + 3H * phi_dot + V_full_prime = 0
    # V_full = (MP^4/8) * V_num. With MP=1, V_full = V_num/8.
    # V_full_prime = d(V_full)/dphi = (MP^4/8) * V_num_phi = V_num_phi / 8.
    # H^2 = V_full / (3*MP^2) = (V_num/8) / 3 = V_num / 24. (with MP=1)
    # 3H = 3 * sqrt(V_num / 24) = sqrt(9 * V_num / 24) = sqrt(3 * V_num / 8).

    # Ensure V_num is positive for sqrt(H^2)
    three_H_term = np.sqrt(3 * V_num / 8.0) if V_num > 0 else 0
    if not np.isfinite(three_H_term): three_H_term = 0 # Avoid issues if V_num is problematic

    phi_ddot = -three_H_term * phi_dot - (1/8.0) * V_num_phi

    return [phi_dot, phi_ddot]


# --- Main Execution ---
if __name__ == "__main__":
    # Option to use user-provided parameters directly for plotting or run optimization
    # Set to True to use the user's last provided parameters for plotting
    # Set to False to run the optimization first
    USE_USER_PARAMS = False # Changed to False to run optimization as requested

    # User's last provided parameters (if USE_USER_PARAMS is True)
    user_params_array = np.array([
        6.0952918, -1.88700513, 9.39810384, 45.84900061,
        -5.78937483, -5.38642911, 9.46398675, 2.05157819, 17.65923743
    ])
    # User's last provided phi_e and phi_star (if USE_USER_PARAMS is True)
    user_phi_e = 8.028341751572691
    user_phi_star = 0.7295265035938892


    if USE_USER_PARAMS:
        print("Using user-provided parameters for plotting.")
        best_params_array = user_params_array
        # Calculate cost for user params to see how good they are with current cost function
        cost_val = cost_function(user_params_array)
        print(f"Cost for user parameters with updated function: {cost_val}")

    else:
        print("Starting optimization... (this may take a while)")
        # Define bounds for the parameters
        bounds = [
            (-20.0, 20.0),    # alpha (must be positive)
            (-20.0, 20.0),   # gamma
            (-20.0, 20.0),    # beta (must be positive)
            (-11.99, 100.0), # k (12+k > 0)
            (-10.0, 10.0),   # mu (mu^2 used, so sign doesn't matter for potential shape)
            (-20.0, 20.0),   # omega
            (-20.0, 20.0),   # chi
            (-20.0, 20.0),   # lambda_val
            (-100.0, 100.0)  # kappa
        ]
        # Add initial guess based on user's previous parameters to help the optimizer
        initial_population = [user_params_array]
        # Add some random variations around the initial guess
        for _ in range(10):
             initial_population.append(user_params_array + np.random.randn(9) * 0.1) # Add small random noise
        # Ensure initial population stays within bounds (optional, but good practice)
        initial_population = [np.clip(p, [b[0] for b in bounds], [b[1] for b in bounds]) for p in initial_population]


        result = optimize.differential_evolution(
            cost_function, bounds,
            strategy='best1bin', maxiter=200, popsize=30, # Increased iterations and popsize
            tol=1e-6, mutation=(0.5, 1.5), recombination=0.8,
            disp=True, workers=-1, polish=True,
            init=initial_population # Use the initial population
        )
        print("\nOptimization finished.")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Best parameters found by optimizer: {result.x}")
        print(f"Minimum cost (ns error squared + penalties): {result.fun}")
        best_params_array = result.x

    # --- Setup for plotting with the chosen parameters ---
    params_plot_dict = {
        'alpha': best_params_array[0], 'gamma': best_params_array[1], 'beta': best_params_array[2],
        'k': best_params_array[3], 'mu': best_params_array[4], 'omega': best_params_array[5],
        'chi': best_params_array[6], 'lambda_val': best_params_array[7], 'kappa': best_params_array[8]
    }
    k_plot = params_plot_dict['k']
    alpha_plot, gamma_plot, beta_plot = params_plot_dict['alpha'], params_plot_dict['gamma'], params_plot_dict['beta']

    # Recalculate constants based on best parameters, handling potential issues
    A_const_plot = np.sqrt(2.0 / (12.0 + k_plot)) / MP if (12.0 + k_plot) > 1e-9 else np.inf
    C_const_plot = np.sqrt(4 * alpha_plot * beta_plot - gamma_plot**2) if (4*alpha_plot*beta_plot - gamma_plot**2)>1e-9 else 0
    phi_limit_plot = (np.pi / 2.0) / A_const_plot if A_const_plot > 1e-9 else np.inf

    # Recalculate phi_e and phi_star for the chosen parameters
    print("\nRecalculating phi_e and phi_star for plotting...")
    phi_e_plot = find_phi_e(params_plot_dict, A_const_plot, C_const_plot, phi_limit_plot, initial_guess_phi = phi_limit_plot*0.8 if np.isfinite(phi_limit_plot) else 5)
    if not np.isfinite(phi_e_plot): # Try negative side if positive fails
         phi_e_plot = find_phi_e(params_plot_dict, A_const_plot, C_const_plot, phi_limit_plot, initial_guess_phi = -phi_limit_plot*0.8 if np.isfinite(phi_limit_plot) else -5)

    phi_star_plot = np.nan # Initialize phi_star_plot

    if np.isfinite(phi_e_plot):
        # Determine V_phi sign for roll direction for integration
        test_phi_for_Vphi_plot = phi_e_plot * 0.95
        if abs(test_phi_for_Vphi_plot) < 1e-5:
             test_phi_for_Vphi_plot = np.sign(phi_e_plot) * 1e-5 if phi_e_plot != 0 else (0.1 * phi_limit_plot if np.isfinite(phi_limit_plot) else 1.0)

        _, V_num_phi_at_test_plot, _ = get_V_num_phi_derivatives(test_phi_for_Vphi_plot, params_plot_dict, A_const_plot, C_const_plot)
        V_phi_sign_for_roll_plot = np.sign(V_num_phi_at_test_plot) if np.isfinite(V_num_phi_at_test_plot) else 0

        if V_phi_sign_for_roll_plot == 0:
            print("Warning: V_phi_sign for roll is zero. Cannot determine integration direction.")
        else:
            try:
                phi_e_eff_plot = phi_e_plot
                if np.isfinite(phi_limit_plot) and abs(phi_e_eff_plot/phi_limit_plot) > 0.999:
                     phi_e_eff_plot = np.sign(phi_e_eff_plot) * phi_limit_plot * 0.999

                sol_plot = integrate.solve_ivp(
                    dphi_dN_ode, [0, N_TARGET], [phi_e_eff_plot],
                    args=(params_plot_dict, A_const_plot, C_const_plot, V_phi_sign_for_roll_plot),
                    dense_output=True, method='RK45', rtol=1e-7, atol=1e-9
                )
                if sol_plot.success:
                    phi_star_plot = sol_plot.y[0, -1]
                else:
                    print(f"Integration for plotting failed: {sol_plot.message}")
            except Exception as e:
                 print(f"Integration for plotting threw exception: {e}")


    # Calculate final observables for the chosen parameters
    ns_final, r_final = np.nan, np.nan
    if np.isfinite(phi_star_plot) and np.isfinite(phi_e_plot):
        eps_V_star_plot = get_epsilon_V(phi_star_plot, params_plot_dict, A_const_plot, C_const_plot)
        eta_V_star_plot = get_eta_V(phi_star_plot, params_plot_dict, A_const_plot, C_const_plot)
        if np.isfinite(eps_V_star_plot) and np.isfinite(eta_V_star_plot):
            ns_final = 1.0 - 6.0 * eps_V_star_plot + 2.0 * eta_V_star_plot
            r_final = 16.0 * eps_V_star_plot

    print(f"\n--- Observables for Plotted Parameters ---")
    print(f"Parameters used for plots: {best_params_array}")
    print(f"phi_e: {phi_e_plot:.4f}, phi_star: {phi_star_plot:.4f} (for N={N_TARGET})")
    print(f"Calculated ns: {ns_final:.5f} (Target: {NS_TARGET})")
    print(f"Calculated r: {r_final:.5f}")


    # --- Plot V(phi) ---
    # Determine plot range based on phi_limit, phi_star, and phi_e
    plot_phi_min = -phi_limit_plot * 0.999 if np.isfinite(phi_limit_plot) else -10
    plot_phi_max = phi_limit_plot * 0.999 if np.isfinite(phi_limit_plot) else 10

    # Adjust plot range to include phi_star and phi_e if they are outside the initial range
    if np.isfinite(phi_star_plot):
        plot_phi_min = min(plot_phi_min, phi_star_plot - 0.2 * abs(phi_star_plot) if phi_star_plot != 0 else -0.2)
        plot_phi_max = max(plot_phi_max, phi_star_plot + 0.2 * abs(phi_star_plot) if phi_star_plot != 0 else 0.2)
    if np.isfinite(phi_e_plot):
        plot_phi_min = min(plot_phi_min, phi_e_plot - 0.2 * abs(phi_e_plot) if phi_e_plot != 0 else -0.2)
        plot_phi_max = max(plot_phi_max, phi_e_plot + 0.2 * abs(phi_e_plot) if phi_e_plot != 0 else 0.2)

    # Ensure min < max and a reasonable range
    if plot_phi_min >= plot_phi_max:
        # Fallback to a default range if calculated range is invalid
        default_range = 10
        plot_phi_min = -default_range
        plot_phi_max = default_range
        if np.isfinite(phi_limit_plot): # If phi_limit exists, use a range relative to it
             plot_phi_min = -phi_limit_plot * 1.1
             plot_phi_max = phi_limit_plot * 1.1

    if plot_phi_max - plot_phi_min < 1e-2: # Ensure a minimal range
         center = (plot_phi_min + plot_phi_max) / 2.0
         plot_phi_min = center - 0.5
         plot_phi_max = center + 0.5


    phi_values_for_plot = np.linspace(plot_phi_min, plot_phi_max, 500)
    V_values_plot = []
    for p_val in phi_values_for_plot:
        # Check if p_val is too close to singularity for X(phi) before calculation
        arg_tan_plot = A_const_plot * p_val
        if np.isfinite(A_const_plot) and np.abs(arg_tan_plot) >= (np.pi/2 * 0.9999):
             V_values_plot.append(np.nan)
             continue

        X_p = get_X_phi(p_val, A_const_plot, C_const_plot, beta_plot, gamma_plot)
        if not np.isfinite(X_p):
            V_values_plot.append(np.nan)
            continue
        V_num_p = get_V_num_from_X(X_p, params_plot_dict)
        V_values_plot.append((MP**4 / 8.0) * V_num_p if np.isfinite(V_num_p) else np.nan)
    V_values_plot = np.array(V_values_plot)

    plt.figure(figsize=(12, 7))
    plt.plot(phi_values_for_plot, V_values_plot, label=r'$V(\tilde{\varphi})/M_p^4$', color='dodgerblue', linewidth=2)

    # Asymptotic value of the potential for large |X| (which corresponds to phi approaching +/- phi_limit)
    # As |X| -> inf, N(X)/P(X)^2 -> (lambda_val * X^4) / (beta * X^2)^2 = lambda_val / beta^2
    if beta_plot != 0 and np.isfinite(params_plot_dict['lambda_val']):
        asymptote_V_num = params_plot_dict['lambda_val'] / (beta_plot**2)
        asymptote_V_plot = (MP**4 / 8.0) * asymptote_V_num
        if np.isfinite(asymptote_V_plot):
             plt.axhline(asymptote_V_plot, linestyle='--', color='gray', label=f'Asymptote $\\lambda/(8\\beta^2) \\approx {asymptote_V_plot:.2e}$')


    if np.isfinite(phi_star_plot):
        V_at_phi_star = (MP**4/8.0) * get_V_num_from_X(get_X_phi(phi_star_plot, A_const_plot, C_const_plot, beta_plot, gamma_plot), params_plot_dict)
        if np.isfinite(V_at_phi_star):
            plt.scatter([phi_star_plot], [V_at_phi_star], color='red', s=80, zorder=5, edgecolor='black', label=fr'$\tilde{{\varphi}}_*$ (N={N_TARGET})')
    if np.isfinite(phi_e_plot):
        V_at_phi_e = (MP**4/8.0) * get_V_num_from_X(get_X_phi(phi_e_plot, A_const_plot, C_const_plot, beta_plot, gamma_plot), params_plot_dict)
        if np.isfinite(V_at_phi_e):
            plt.scatter([phi_e_plot], [V_at_phi_e], color='green', s=80, zorder=5, edgecolor='black', label=r'$\tilde{\varphi}_e$ (End of Inflation)')

    plt.xlabel(r'$\tilde{\varphi}/M_p$ (assuming $M_p=1$)', fontsize=14)
    plt.ylabel(r'$V(\tilde{\varphi}) / M_p^4$', fontsize=14)
    plt.title('Inflationary Potential $V(\\tilde{\\varphi})$', fontsize=16)

    # Adjust y-axis limits based on valid potential values
    valid_V_plot = V_values_plot[np.isfinite(V_values_plot)]
    if len(valid_V_plot) > 1: # Need at least two points to determine range
        ymin, ymax = np.percentile(valid_V_plot, [1, 99]) # Use percentiles to avoid extreme outliers
        # Ensure ymin is not excessively negative if potential should be positive
        if np.any(valid_V_plot > 0): # If there are positive values, ensure ymin is not too low
             ymin = max(ymin, -0.1 * np.max(valid_V_plot[valid_V_plot > 0])) # Allow slight dip below zero if needed

        # Include asymptote in y-range if defined
        if 'asymptote_V_plot' in locals() and np.isfinite(asymptote_V_plot):
            ymin = min(ymin, asymptote_V_plot)
            ymax = max(ymax, asymptote_V_plot)

        yrange = ymax - ymin
        if yrange < 1e-9: yrange = abs(ymin)*0.2 if ymin !=0 else 0.1 # Handle flat or zero potential
        plt.ylim(ymin - 0.15 * yrange, ymax + 0.15 * yrange)
    elif len(valid_V_plot) == 1: # Only one valid point
         plt.ylim(valid_V_plot[0] - 0.1, valid_V_plot[0] + 0.1)
    else: # No valid points
        plt.ylim(-0.01, 0.01) # Fallback to a small default range

    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # --- Plot Phase Portrait (Streamplot) ---
    phi_stream_min = plot_phi_min
    phi_stream_max = plot_phi_max

    # Estimate typical phi_dot range based on slow-roll velocity at phi_star
    phi_dot_range_abs = 0.05 # Default range
    if np.isfinite(phi_star_plot):
         V_num_at_phi_star_plot = get_V_num_from_X(get_X_phi(phi_star_plot, A_const_plot, C_const_plot, beta_plot, gamma_plot), params_plot_dict)
         eps_V_star_plot = get_epsilon_V(phi_star_plot, params_plot_dict, A_const_plot, C_const_plot)
         # Slow roll phi_dot^2 ~ epsilon_V * V_num / 4
         if np.isfinite(eps_V_star_plot) and np.isfinite(V_num_at_phi_star_plot) and V_num_at_phi_star_plot > 0:
             phi_dot_sr_sq = eps_V_star_plot * V_num_at_phi_star_plot / 4.0
             if phi_dot_sr_sq > 0:
                  # Set the range to be wide enough to see the slow-roll trajectory and surrounding dynamics
                  phi_dot_range_abs = max(np.sqrt(phi_dot_sr_sq) * 20, 0.01) # Multiplied by 20 for wider view

    phi_stream_vals = np.linspace(phi_stream_min, phi_stream_max, 40) # Increased grid points
    phi_dot_stream_vals = np.linspace(-phi_dot_range_abs, phi_dot_range_abs, 40) # Increased grid points

    PHI, PHI_DOT = np.meshgrid(phi_stream_vals, phi_dot_stream_vals)

    u_vel = np.zeros_like(PHI) # Stores d(phi)/dt = phi_dot
    v_vel = np.zeros_like(PHI_DOT) # Stores d(phi_dot)/dt = phi_ddot
    speed = np.zeros_like(PHI) # Stores the speed of the vector field

    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            phi_val = PHI[i, j]
            phi_dot_val = PHI_DOT[i, j]

            # Check if phi_val is too close to singularity for X(phi)
            arg_tan_stream = A_const_plot * phi_val
            if np.isfinite(A_const_plot) and np.abs(arg_tan_stream) >= (np.pi/2 * 0.9999):
                 u_vel[i,j], v_vel[i,j] = 0, 0 # Treat as a wall
                 speed[i,j] = 0
                 continue

            derivs = phase_portrait_odes(0, [phi_val, phi_dot_val], params_plot_dict, A_const_plot, C_const_plot)
            u_vel[i,j] = derivs[0]
            v_vel[i,j] = derivs[1]
            speed[i,j] = np.sqrt(u_vel[i,j]**2 + v_vel[i,j]**2) # Calculate speed

    plt.figure(figsize=(10, 8))
    # Streamplot with adjusted density, arrowsize, and colored by speed
    strm = plt.streamplot(PHI, PHI_DOT, u_vel, v_vel, density=1.5, linewidth=0.8,
                          color=speed, cmap='viridis', arrowsize=0.8) # Use 'color=speed'

    # Add a color bar to show speed mapping
    cbar = plt.colorbar(strm.lines)
    cbar.set_label('Speed $d\tilde{\varphi}/dt$', fontsize=12)

    # Overlay inflationary trajectory points if calculable
    if np.isfinite(phi_star_plot) and np.isfinite(phi_e_plot):
        # Estimate phi_dot at phi_star based on slow roll
        V_num_at_phi_star_plot = get_V_num_from_X(get_X_phi(phi_star_plot, A_const_plot, C_const_plot, beta_plot, gamma_plot), params_plot_dict)
        eps_V_star_plot = get_epsilon_V(phi_star_plot, params_plot_dict, A_const_plot, C_const_plot)

        phi_dot_star_val = 0 # Default if calculation fails
        if np.isfinite(eps_V_star_plot) and np.isfinite(V_num_at_phi_star_plot) and V_num_at_phi_star_plot > 0:
            # Sign of phi_dot_star: if V_num_phi_star < 0, phi increases (rolls right), so phi_dot > 0
            # If V_num_phi_star > 0, phi decreases (rolls left), so phi_dot < 0
            # This is the forward-in-time sign.
            _, V_num_phi_star_val, _ = get_V_num_phi_derivatives(phi_star_plot, params_plot_dict, A_const_plot, C_const_plot)
            sign_phi_dot = -np.sign(V_num_phi_star_val) if np.isfinite(V_num_phi_star_val) and V_num_phi_star_val !=0 else 1 # Default sign if V_phi is zero
            phi_dot_star_val = sign_phi_dot * np.sqrt(eps_V_star_plot * V_num_at_phi_star_plot / 4.0) # Slow roll phi_dot magnitude

        plt.scatter([phi_star_plot], [phi_dot_star_val], color='red', s=100, zorder=10, edgecolor='black', label=fr'$Phase space$')

        # Estimate phi_dot at phi_e (where epsilon_V = 1)
        V_num_at_phi_e_plot = get_V_num_from_X(get_X_phi(phi_e_plot, A_const_plot, C_const_plot, beta_plot, gamma_plot), params_plot_dict)
        if np.isfinite(V_num_at_phi_e_plot) and V_num_at_phi_e_plot > 0:
            # Sign of phi_dot_e: determined by the potential slope at phi_e
            _, V_num_phi_e_val, _ = get_V_num_phi_derivatives(phi_e_plot, params_plot_dict, A_const_plot, C_const_plot)
            sign_phi_dot_e = -np.sign(V_num_phi_e_val) if np.isfinite(V_num_phi_e_val) and V_num_phi_e_val !=0 else 1
            phi_dot_e_val = sign_phi_dot_e * np.sqrt(V_num_at_phi_e_plot / 4.0) # Since epsilon_V = 1, phi_dot^2 = V/4

            plt.scatter([phi_e_plot], [phi_dot_e_val], color='green', s=100, zorder=10, edgecolor='black', label=r'$(\tilde{\varphi}_e, d\tilde{\varphi}/dt)$')


    plt.xlabel(r'$\tilde{\varphi}/M_p$', fontsize=14)
    plt.ylabel(r'$d\tilde{\varphi}}/dt$', fontsize=14) # Units assuming t is in Mp^-1
    plt.title('Phase Portrait', fontsize=16)
    plt.xlim(phi_stream_min, phi_stream_max)
    plt.ylim(-phi_dot_range_abs, phi_dot_range_abs) # Set y-limits based on calculated range
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

        # --- Streamplot Visualization (Fixed and Beautified) ---
    # Build a fine mesh for (phi, dphi/dN)
    phi_vals   = np.linspace(phi_stream_min, phi_stream_max, 300)
    dphi_vals  = np.linspace(-phi_dot_range_abs, phi_dot_range_abs, 300)
    PHI, DPHI  = np.meshgrid(phi_vals, dphi_vals)

    # Use the same phase_equations function you already have:
    def d2phi_dN2(phi, dphi, params_dict, A_const, C_const):
        """ Wrapper around your phase_portrait_odes to return just φ''(N) """
        _, phi_ddot = phase_portrait_odes(0, [phi, dphi], params_dict, A_const, C_const)
        return phi_ddot

    # Compute the vector field and speed
    U = DPHI
    V = np.zeros_like(PHI)
    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            V[i,j] = d2phi_dN2(PHI[i,j], DPHI[i,j], params_plot_dict, A_const_plot, C_const_plot)
    speed = np.sqrt(U**2 + V**2)

    # Line‑width scaled by speed (for visual punch)
    lw = 1.2

    plt.figure(figsize=(10,6))
    # coloured contour of log‑speed
    plt.contourf(PHI, DPHI, np.log10(speed+1e-12),
                 levels=50, cmap='viridis', alpha=0.8)

    # streamlines on top
    strm = plt.streamplot(PHI, DPHI, U, V,
                          color='k',
                          density=1.2,
                          linewidth=lw,
                          arrowsize=1)

    cbar = plt.colorbar(label=r'$\log_{10}(\mathrm{flow\ speed})$')
    plt.scatter([phi_star_plot], [0], color='white', edgecolor='k', s=100,
                label=r'Inflationary trajectory ($N=60$ start)')
    plt.scatter([phi_e_plot], [0], color='cyan', edgecolor='k', s=100,
                label=r'End of inflation ($\varepsilon=1$)')

    plt.xlabel(r'$\tilde\varphi$')
    plt.ylabel(r'$d\tilde\varphi/dN$')
    plt.title('Phase Portrait with Flow‑Speed Contour')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
