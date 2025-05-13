import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import time

# Constants
M_p = 1.0  # Planck mass in natural units

# === 1) Parameter packing/unpacking ===
def pack_params(x):
    # x = [log_alpha, log_kplus12, mu, omega, chi, log_kappa]
    log_alpha, log_kp12, mu, omega, chi, log_kappa = x
    return {
        'alpha': np.exp(log_alpha),
        'k': max(np.exp(log_kp12) - 12.0, 1e-6),
        'mu': mu,
        'omega': omega,
        'chi': chi,
        'kappa': np.exp(log_kappa),
    }

def unpack_params(params):
    return [
        np.log(params['alpha']), 
        np.log(params['k'] + 12.0),
        params['mu'],
        params['omega'],
        params['chi'],
        np.log(params['kappa'])
    ]

# === 2) Potential and its derivatives ===
def V_Vp_Vpp(phi, params):
    α, k = params['alpha'], params['k']
    μ, ω, χ, κ = params['mu'], params['omega'], params['chi'], params['kappa']
    
    Ck = np.sqrt(2.0/(12.0 + k)) / M_p
    t = Ck*phi
    D = np.sqrt(4*α)
    X = (D * np.tan(t)) / 2.0
    
    Np = κ + 2*ω*X + μ*X**2 + 2*χ*X**3 + X**4
    D2 = (X**2 + α)**2
    V = (M_p**4/8.0) * Np / D2
    
    dNp = 2*ω + 2*μ*X + 6*χ*X**2 + 4*X**3
    dD2 = 4*X*(X**2 + α)
    dVdX = (M_p**4/8.0)*(dNp*D2 - Np*dD2)/D2**2
    
    dXdφ = (D/2.0)*Ck/np.cos(t)**2
    Vp = dVdX * dXdφ
    
    d2Np = 2*μ + 12*χ*X + 12*X**2
    d2D2 = 12*X**2 + 4*α
    d2VdX2 = (M_p**4/8.0)*(d2Np*D2 - 2*dNp*dD2 - Np*d2D2 + 2*Np*(dD2**2)/D2)/D2**2
    
    d2Xdφ2 = (D/2.0)*Ck**2 * 2*np.tan(t)/np.cos(t)**2
    Vpp = d2VdX2*dXdφ**2 + dVdX*d2Xdφ2
    
    return V, Vp, Vpp

def phi_max(params):
    """Maximum allowed value of canonical field phi"""
    k = params['k']
    return np.sqrt((12.0 + k)/2.0) * M_p * np.pi/2.0 * 0.999  # Slightly less than theoretical maximum

def compute_field_mapping(phi, params):
    """Compute X(phi) mapping from canonical to non-canonical field"""
    α, k = params['alpha'], params['k']
    Ck = np.sqrt(2.0/(12.0 + k)) / M_p
    D = np.sqrt(4*α)
    return (D * np.tan(Ck*phi)) / 2.0

# === 3) Slow-roll parameters ===
def slow_roll_params(phi, params):
    V, Vp, Vpp = V_Vp_Vpp(phi, params)
    
    # First slow-roll parameter epsilon
    epsilon = 0.5 * M_p**2 * (Vp/V)**2
    
    # Second slow-roll parameter eta
    eta = M_p**2 * Vpp/V
    
    return epsilon, eta

# === 4) Compute N(phi) - Number of e-folds ===
def compute_N(phi_values, params):
    """
    Compute number of e-folds from phi_end to phi_star
    phi_values should be in ascending order
    """
    N_values = np.zeros_like(phi_values)
    
    for i in range(1, len(phi_values)):
        phi_curr, phi_prev = phi_values[i], phi_values[i-1]
        
        # Get slow-roll parameters
        epsilon_curr, _ = slow_roll_params(phi_curr, params)
        epsilon_prev, _ = slow_roll_params(phi_prev, params)
        
        # Compute dN/dphi and integrate
        dN_prev = 1.0/np.sqrt(2*epsilon_prev)
        dN_curr = 1.0/np.sqrt(2*epsilon_curr)
        
        # Trapezoidal integration
        N_values[i] = N_values[i-1] + 0.5 * (dN_prev + dN_curr) * (phi_curr - phi_prev)
    
    return N_values

def find_phi_end(params):
    """Find phi value where epsilon = 1 (end of inflation)"""
    phi_max_val = phi_max(params)
    
    # Search from small phi to find where epsilon reaches 1
    phi_test = np.linspace(0.001, phi_max_val * 0.95, 1000)
    
    for phi in phi_test:
        epsilon, _ = slow_roll_params(phi, params)
        if epsilon >= 1.0:
            return phi
    
    # If not found, return a small value
    return 0.001

def find_phi_star(params, N_target=60):
    """
    Find phi_star that gives N_target e-folds before end of inflation
    """
    phi_end_val = find_phi_end(params)
    phi_max_val = phi_max(params)
    
    # Create a grid from phi_end to phi_max
    phi_grid = np.linspace(phi_end_val, 0.95*phi_max_val, 1000)
    
    # Compute N values along this grid
    N_values = compute_N(phi_grid, params)
    
    # Find phi value that gives N_target
    idx = np.argmin(np.abs(N_values - N_target))
    
    return phi_grid[idx], N_values[idx]

# === 5) Compute scalar spectral index n_s ===
def compute_ns(phi, params):
    """Compute scalar spectral index n_s at given phi"""
    epsilon, eta = slow_roll_params(phi, params)
    return 1.0 - 6*epsilon + 2*eta

# === 6) Optimization functions ===
def objective_function(x):
    """
    Objective function to minimize: |n_s - target| at N=60
    x = [log_alpha, log_kplus12, mu, omega, chi, log_kappa]
    """
    params = pack_params(x)
    
    try:
        # Check parameter constraints
        if params['k'] <= -12.0 or params['alpha'] <= 0:
            return 10.0  # High penalty for invalid parameters
        
        # Find phi_star for N=60
        phi_star, actual_N = find_phi_star(params, N_target=60)
        
        # If we couldn't achieve N=60, penalize
        if abs(actual_N - 60) > 5:
            return 5.0
        
        # Compute n_s
        n_s = compute_ns(phi_star, params)
        
        # Return difference from target
        target_ns = 0.9626
        return abs(n_s - target_ns)
    
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 10.0  # High penalty on error

# === 7) Hybrid optimization approach ===
def hybrid_optimization():
    """
    Hybrid optimization approach combining Differential Evolution with local search
    """
    # Parameter bounds: [log_alpha, log_kplus12, mu, omega, chi, log_kappa]
    bounds = [
        (-3.0, 2.0),      # log_alpha
        (-6.0, 2.0),      # log_kplus12
        (-10.0, 10.0),    # mu
        (-10.0, 10.0),    # omega
        (-10.0, 10.0),    # chi
        (-3.0, 2.0)       # log_kappa
    ]
    
    # Target n_s value
    TARGET_NS = 0.9626
    
    # Storage for promising solutions
    promising_regions = []
    best_solutions = []
    
    # Create a callback function to monitor DE progress and collect promising regions
    def callback(xk, convergence):
        params = pack_params(xk)
        try:
            phi_star, actual_N = find_phi_star(params, N_target=60)
            n_s = compute_ns(phi_star, params)
            diff = abs(n_s - TARGET_NS)
            
            # Store best solutions
            solution = {
                'params': params.copy(),
                'phi_star': phi_star,
                'N': actual_N,
                'n_s': n_s,
                'diff': diff
            }
            
            best_solutions.append(solution)
            best_solutions.sort(key=lambda x: x['diff'])
            if len(best_solutions) > 20:  # Keep only top 20
                best_solutions.pop()
            
            print(f"N={actual_N:.1f}, n_s={n_s:.6f}, diff={diff:.6f}, params={params}")
            
            # Check if this is a promising region (close to target)
            if diff < 0.01:
                # Check if we already have a similar point
                is_new = True
                for existing_point in promising_regions:
                    if np.linalg.norm(np.array(xk) - np.array(existing_point['x'])) < 0.5:
                        is_new = False
                        break
                
                if is_new:
                    print(f"Found promising region: n_s={n_s:.6f}")
                    promising_regions.append({'x': xk.copy(), 'n_s': n_s})
        except Exception as e:
            print(f"Error in callback: {e}")
        
        # Don't stop DE - let it run its course
        return False
    
    print("Starting Differential Evolution...")
    
    # Run differential evolution
    de_result = differential_evolution(
        objective_function, 
        bounds,
        strategy='best1bin',
        popsize=20,
        mutation=(0.5, 1.0),
        recombination=0.7,
        maxiter=100,
        callback=callback,
        disp=True,
        polish=False  # We'll do our own polishing
    )
    
    print("\nDE finished. Running local searches on promising regions...")
    
    # Run local searches on all promising regions
    local_results = []
    
    # First, try the DE result
    local_result = minimize(
        objective_function,
        de_result.x,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    local_results.append(local_result)
    
    # Then try all promising regions
    for region in promising_regions:
        local_result = minimize(
            objective_function,
            region['x'],
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        local_results.append(local_result)
    
    # Find best result across all optimizations
    best_result = min(local_results, key=lambda x: x.fun)
    best_params = pack_params(best_result.x)
    
    # Compute phi_star and n_s for the best result
    phi_star, N = find_phi_star(best_params, N_target=60)
    n_s = compute_ns(phi_star, best_params)
    
    print("\nBest Parameters Found:")
    for key, value in best_params.items():
        print(f"{key} = {value:.6f}")
    
    print(f"\nAt N = {N:.2f}, n_s = {n_s:.6f}")
    
    return best_params, best_solutions

# === 8) Plotting functions ===
def plot_potential(params):
    """Plot canonical potential V(phi)"""
    phi_max_val = phi_max(params)
    phi_values = np.linspace(-0.95*phi_max_val, 0.95*phi_max_val, 1000)
    
    # Calculate potential values
    V_values = np.array([V_Vp_Vpp(phi, params)[0] for phi in phi_values])
    
    # Normalize potential for better visualization
    V_norm = V_values / np.max(V_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(phi_values, V_norm, 'b-', linewidth=2)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Find phi_end and phi_star for N=60
    phi_end = find_phi_end(params)
    phi_star, _ = find_phi_star(params, N_target=60)
    
    # Mark phi_end and phi_star on the plot
    plt.axvline(x=phi_end, color='r', linestyle='--', label=r'$\phi_{end}$')
    plt.axvline(x=phi_star, color='g', linestyle='--', label=r'$\phi_{*}$ (N=60)')
    
    plt.title('Canonical Potential V(φ)')
    plt.xlabel('Canonical Field φ')
    plt.ylabel('Normalized Potential V(φ)/V_max')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_ns_vs_N(params):
    """Plot n_s vs N"""
    phi_end = find_phi_end(params)
    phi_max_val = phi_max(params)
    
    # Create phi grid from phi_end to near phi_max
    phi_grid = np.linspace(phi_end, 0.95*phi_max_val, 500)
    
    # Compute N values
    N_values = compute_N(phi_grid, params)
    
    # Filter to keep only meaningful N values (e.g., up to 100)
    mask = N_values <= 100
    phi_grid = phi_grid[mask]
    N_values = N_values[mask]
    
    # Compute n_s values
    ns_values = np.array([compute_ns(phi, params) for phi in phi_grid])
    
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, ns_values, 'b-', linewidth=2)
    
    # Mark n_s at N=60
    idx = np.argmin(np.abs(N_values - 60))
    if idx < len(N_values):
        plt.plot(N_values[idx], ns_values[idx], 'ro', markersize=8, 
                 label=f'N=60, n_s={ns_values[idx]:.6f}')
    
    # Add target line
    plt.axhline(y=0.9626, color='g', linestyle='--', label='Target n_s = 0.9626')
    
    plt.title('Scalar Spectral Index n_s vs. Number of e-folds N')
    plt.xlabel('Number of e-folds N')
    plt.ylabel('Scalar Spectral Index n_s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_phase_space(params, compactified=False):
    """
    Plot phase space portrait (φ, φ̇)
    If compactified=True, use compactified coordinates
    """
    phi_max_val = phi_max(params)
    
    # Create phi grid
    if compactified:
        # Use theta as parameter to create more points near boundaries
        theta = np.linspace(0, np.pi, 500)
        phi_grid = phi_max_val * np.sin(theta)
    else:
        phi_grid = np.linspace(-0.95*phi_max_val, 0.95*phi_max_val, 100)
    
    # Create phi_dot grid
    phi_dot_max = 5.0  # Arbitrary maximum for phi_dot
    if compactified:
        theta = np.linspace(0, np.pi, 500)
        phi_dot_grid = phi_dot_max * np.sin(theta)
    else:
        phi_dot_grid = np.linspace(-phi_dot_max, phi_dot_max, 100)
    
    # Create mesh grid
    PHI, PHI_DOT = np.meshgrid(phi_grid, phi_dot_grid)
    
    # Calculate derivatives for phase space
    dPHI_dt = PHI_DOT
    
    # Calculate dPHI_DOT/dt = -3H*PHI_DOT - V'(PHI)/M_p^2
    dPHI_DOT_dt = np.zeros_like(PHI)
    
    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            phi = PHI[i, j]
            phi_dot = PHI_DOT[i, j]
            
            # Calculate V and V'
            V, Vp, _ = V_Vp_Vpp(phi, params)
            
            # Calculate Hubble parameter H
            H = np.sqrt(V + 0.5 * phi_dot**2) / np.sqrt(3) / M_p
            
            # Calculate dPHI_DOT/dt
            dPHI_DOT_dt[i, j] = -3 * H * phi_dot - Vp
    
    # Normalize vectors for streamplot
    magnitude = np.sqrt(dPHI_dt**2 + dPHI_DOT_dt**2)
    magnitude[magnitude < 1e-10] = 1e-10  # Avoid division by zero
    
    dPHI_dt_norm = dPHI_dt / magnitude
    dPHI_DOT_dt_norm = dPHI_DOT_dt / magnitude
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Define a custom colormap from blue to red
    colors = [(0, 0, 0.8), (0, 0, 1), (0, 0.5, 1), (0, 0.8, 0.8), 
              (0.5, 1, 0.5), (0.8, 0.8, 0), (1, 0.5, 0), (1, 0, 0), (0.8, 0, 0)]
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
    
    # Create streamplot
    plt.streamplot(PHI, PHI_DOT, dPHI_dt_norm, dPHI_DOT_dt_norm, 
                   density=1.5, color=magnitude, cmap=custom_cmap, 
                   linewidth=1.5, arrowsize=1.2)
    
    # Add colorbar
    cbar = plt.colorbar(label='Normalized Magnitude')
    
    # Find phi_end and phi_star (N=60)
    phi_end = find_phi_end(params)
    phi_star, _ = find_phi_star(params, N_target=60)
    
    # Mark phi_end and phi_star
    plt.axvline(x=phi_end, color='r', linestyle='--', label=r'$\phi_{end}$')
    plt.axvline(x=phi_star, color='g', linestyle='--', label=r'$\phi_{*}$ (N=60)')
    
    # Add labels and title
    if compactified:
        plt.title('Compactified Phase Space Portrait')
    else:
        plt.title('Phase Space Portrait')
    
    plt.xlabel('Canonical Field φ')
    plt.ylabel('Field Velocity φ̇')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    return plt.gcf()

# === 9) Main function to run optimization and generate plots ===
def main():
    start_time = time.time()
    
    print("Starting optimization...")
    best_params, best_solutions = hybrid_optimization()
    
    print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot potential
    fig_potential = plot_potential(best_params)
    fig_potential.savefig('canonical_potential.png', dpi=300, bbox_inches='tight')
    
    # Plot n_s vs N
    fig_ns_N = plot_ns_vs_N(best_params)
    fig_ns_N.savefig('ns_vs_N.png', dpi=300, bbox_inches='tight')
    
    # Plot phase space portraits
    fig_phase_space = plot_phase_space(best_params, compactified=False)
    fig_phase_space.savefig('phase_space.png', dpi=300, bbox_inches='tight')
    
    fig_phase_space_comp = plot_phase_space(best_params, compactified=True)
    fig_phase_space_comp.savefig('compactified_phase_space.png', dpi=300, bbox_inches='tight')
    
    print("\nAll plots saved successfully.")
    
    # Plot evolution of best solutions during optimization
    plt.figure(figsize=(12, 8))
    
    # Extract data from best_solutions
    ns_values = [sol['n_s'] for sol in best_solutions]
    diff_values = [sol['diff'] for sol in best_solutions]
    
    plt.subplot(2, 1, 1)
    plt.plot(ns_values, 'b.-')
    plt.axhline(y=0.9626, color='r', linestyle='--', label='Target')
    plt.title('Evolution of n_s During Optimization')
    plt.ylabel('n_s')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.semilogy(diff_values, 'g.-')
    plt.title('Evolution of Error |n_s - 0.9626| During Optimization')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Error (log scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
    
    # Return best parameters and n_s value
    phi_star, N = find_phi_star(best_params, N_target=60)
    n_s = compute_ns(phi_star, best_params)
    
    print("\nSummary:")
    print(f"Found n_s = {n_s:.6f} at N = {N:.2f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"{key} = {value:.6f}")

if __name__ == "__main__":
    main()