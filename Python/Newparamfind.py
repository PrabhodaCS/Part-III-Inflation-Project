import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root
from scipy.integrate import solve_ivp
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# Configure matplotlib for publication quality plots
mpl.rcParams['axes.linewidth'] = 3.0
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
})

# Constants
M_p = 1.0  # Planck mass (set to 1 for natural units)
phi_0 = 1.0  # Field scale

# Model parameters (initial guess)
class Parameters:
    def __init__(self):
        self.M = 1.0      # Energy scale
        self.alpha = 1.0  # Kinetic coefficient parameters
        self.beta = 0.5
        self.gamma = 0.1
        self.k = 6.0      # Extra parameter in kinetic term
        self.kappa = 1.0  # Potential parameters
        self.omega = 0.5
        self.mu = 0.8
        self.chi = 0.3
        self.lam = 0.2    # lambda is a reserved keyword in Python

# Create global parameters object
params = Parameters()

def kinetic_coefficient(phi, params):
    """Calculate K(phi) from equation"""
    num = params.M**2 * (12 + params.k) * (4 * params.alpha * params.beta - params.gamma**2) * phi_0**2
    denom = 2 * (params.beta * phi**2 + params.gamma * phi_0 * phi + params.alpha * phi_0**2)
    return num / denom

def potential_non_canonical(phi, params):
    """Calculate V(phi) for the non-canonical field"""
    num = (params.kappa * phi_0**4 + 
           2 * params.omega * phi_0**3 * phi + 
           params.mu**2 * phi_0**2 * phi**2 + 
           2 * params.chi * phi_0 * phi**3 + 
           params.lam * phi**4)
    
    denom = 2 * (params.beta * phi**2 + params.gamma * phi_0 * phi + params.alpha * phi_0**2)**2
    
    return params.M**4 * num / denom

def phi_to_phi_tilde(phi, params):
    """Transform from phi to phi_tilde (canonical field)"""
    # Check stability condition
    if 4 * params.alpha * params.beta <= params.gamma**2:
        raise ValueError("Ghost instability: 4αβ <= γ²")
        
    return np.sqrt((12 + params.k) / 2) * M_p * np.arctan(
        (2 * phi * params.beta + params.gamma * phi_0) / 
        (phi_0 * np.sqrt(4 * params.alpha * params.beta - params.gamma**2))
    )

def phi_tilde_to_phi(phi_tilde, params):
    """Transform from phi_tilde (canonical field) to phi"""
    return (phi_0 / (2 * params.beta)) * (
        np.sqrt(4 * params.alpha * params.beta - params.gamma**2) * 
        np.tan(np.sqrt(2 / (12 + params.k)) * phi_tilde / M_p) - 
        params.gamma
    )

def V_canonical(phi_tilde, params):
    """Calculate canonical potential V(phi(phi_tilde))"""
    try:
        phi = phi_tilde_to_phi(phi_tilde, params)
        return potential_non_canonical(phi, params)
    except:
        # Return a large value for numerical stability if there's an issue
        return 1e10

def dV_dphi_tilde(phi_tilde, params, epsilon=1e-8):
    """Calculate dV/dphi_tilde numerically"""
    return (V_canonical(phi_tilde + epsilon, params) - V_canonical(phi_tilde - epsilon, params)) / (2 * epsilon)

def d2V_dphi_tilde2(phi_tilde, params, epsilon=1e-6):
    """Calculate d²V/dphi_tilde² numerically"""
    return (dV_dphi_tilde(phi_tilde + epsilon, params) - dV_dphi_tilde(phi_tilde - epsilon, params)) / (2 * epsilon)

def slow_roll_epsilon(phi_tilde, params):
    """Calculate first slow-roll parameter epsilon"""
    deriv = dV_dphi_tilde(phi_tilde, params)
    V = V_canonical(phi_tilde, params)
    return 0.5 * M_p**2 * (deriv / V)**2

def slow_roll_eta(phi_tilde, params):
    """Calculate second slow-roll parameter eta"""
    second_deriv = d2V_dphi_tilde2(phi_tilde, params)
    V = V_canonical(phi_tilde, params)
    return M_p**2 * second_deriv / V

def spectral_index(phi_tilde, params):
    """Calculate spectral index ns"""
    eps = slow_roll_epsilon(phi_tilde, params)
    eta = slow_roll_eta(phi_tilde, params)
    return 1 - 6*eps + 2*eta

def N_efolds(phi_tilde_start, phi_tilde_end, params):
    """Calculate number of e-folds from phi_tilde_start to phi_tilde_end"""
    def integrand(phi_t):
        V = V_canonical(phi_t, params)
        dV = dV_dphi_tilde(phi_t, params)
        return V / (M_p**2 * dV)
    
    # Simple numerical integration using trapezoidal rule
    phi_values = np.linspace(phi_tilde_start, phi_tilde_end, 1000)
    integrand_values = [integrand(phi) for phi in phi_values]
    return -np.trapz(integrand_values, phi_values)

def find_phi_end(phi_tilde_guess, params):
    """Find field value where inflation ends (epsilon = 1)"""
    def epsilon_minus_1(phi_t):
        eps = slow_roll_epsilon(phi_t, params)
        return eps - 1.0
    
    result = root(epsilon_minus_1, phi_tilde_guess)
    if result.success:
        return result.x[0]
    else:
        raise ValueError("Could not find phi_end where epsilon = 1")

def find_phi_N_efolds(phi_tilde_end, N_target, params):
    """Find field value N e-folds before the end of inflation"""
    def N_diff(phi_tilde_start):
        N = N_efolds(phi_tilde_start, phi_tilde_end, params)
        return N - N_target
    
    # Try different initial guesses if needed
    guesses = [phi_tilde_end + 1.0, phi_tilde_end + 5.0, phi_tilde_end + 10.0]
    
    for guess in guesses:
        try:
            result = root(N_diff, guess)
            if result.success:
                return result.x[0]
        except:
            continue
    
    raise ValueError("Could not find phi_N_efolds")

def optimize_parameters_for_ns(target_ns=0.9626, target_N=60):
    """
    Optimize model parameters to hit target spectral index at target N e-folds
    """
    def objective(x):
        # Unpack parameters (keeping some fixed for simplicity)
        params.alpha = np.abs(x[0])  # Keep positive to avoid instabilities
        params.beta = np.abs(x[1])   # Keep positive
        params.gamma = x[2]
        params.k = np.abs(x[3])      # Keep positive
        params.omega = x[4]
        params.mu = x[5]
        params.chi = x[6]
        params.lam = x[7]
        
        # Check stability condition
        if 4 * params.alpha * params.beta <= params.gamma**2:
            return 1e10  # Penalty for ghost instability
        
        try:
            # Find end of inflation
            phi_tilde_end = find_phi_end(0.1, params)
            
            # Find field value at N=60
            phi_tilde_N60 = find_phi_N_efolds(phi_tilde_end, target_N, params)
            
            # Calculate ns at N=60
            ns_calculated = spectral_index(phi_tilde_N60, params)
            
            # Return error (difference from target)
            return abs(ns_calculated - target_ns)
        except:
            # If any calculation fails, return large penalty
            return 1e10
    
    # Initial parameter guesses
    x0 = [params.alpha, params.beta, params.gamma, params.k, 
          params.omega, params.mu, params.chi, params.lam]
    
    # Run optimization
    print("Starting parameter optimization...")
    result = minimize(objective, x0, method='Nelder-Mead', 
                     options={'maxiter': 1000, 'disp': True})
    
    # Update parameters with optimized values
    params.alpha = np.abs(result.x[0])
    params.beta = np.abs(result.x[1])
    params.gamma = result.x[2]
    params.k = np.abs(result.x[3])
    params.omega = result.x[4]
    params.mu = result.x[5]
    params.chi = result.x[6]
    params.lam = result.x[7]
    
    print("Optimization complete:")
    print(f"α = {params.alpha:.6f}")
    print(f"β = {params.beta:.6f}")
    print(f"γ = {params.gamma:.6f}")
    print(f"k = {params.k:.6f}")
    print(f"ω = {params.omega:.6f}")
    print(f"μ = {params.mu:.6f}")
    print(f"χ = {params.chi:.6f}")
    print(f"λ = {params.lam:.6f}")
    
    # Verify results
    try:
        phi_tilde_end = find_phi_end(0.1, params)
        phi_tilde_N60 = find_phi_N_efolds(phi_tilde_end, target_N, params)
        ns_achieved = spectral_index(phi_tilde_N60, params)
        eps = slow_roll_epsilon(phi_tilde_N60, params)
        eta = slow_roll_eta(phi_tilde_N60, params)
        
        print("\nResults verification:")
        print(f"Field value at end of inflation: φ_tilde_end = {phi_tilde_end:.6f}")
        print(f"Field value at N=60: φ_tilde_N60 = {phi_tilde_N60:.6f}")
        print(f"Target ns = {target_ns}, Achieved ns = {ns_achieved:.6f}")
        print(f"Slow-roll parameters at N=60: ε = {eps:.6f}, η = {eta:.6f}")
        
        return phi_tilde_end, phi_tilde_N60
    except Exception as e:
        print(f"Verification failed: {e}")
        return None, None

def plot_canonical_potential(phi_tilde_end=None, phi_tilde_N60=None):
    """Plot the canonical potential V(φ_tilde)"""
    # Define phi_tilde range
    phi_tilde_range = np.linspace(-15, 15, 1000)
    
    # Filter out points that would cause singularities
    safe_points = []
    potential_values = []
    
    for phi_t in phi_tilde_range:
        try:
            pot = V_canonical(phi_t, params)
            if not np.isnan(pot) and not np.isinf(pot) and pot < 1e9:
                safe_points.append(phi_t)
                potential_values.append(pot)
        except:
            continue
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(safe_points, potential_values, 'b-', linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$\tilde{\varphi}/M_p$', fontsize=14)
    ax.set_ylabel(r'$V(\tilde{\varphi})/M^4$', fontsize=14)
    ax.set_title('Canonical Inflaton Potential', fontsize=16)
    
    # Mark important points if provided
    if phi_tilde_end is not None:
        V_end = V_canonical(phi_tilde_end, params)
        ax.plot(phi_tilde_end, V_end, 'ro')
        ax.annotate(r'$\tilde{\varphi}_{\mathrm{end}}$', 
                    xy=(phi_tilde_end, V_end),
                    xytext=(phi_tilde_end+1, V_end+0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12)
    
    if phi_tilde_N60 is not None:
        V_N60 = V_canonical(phi_tilde_N60, params)
        ax.plot(phi_tilde_N60, V_N60, 'go')
        ax.annotate(r'$\tilde{\varphi}_{N=60}$', 
                    xy=(phi_tilde_N60, V_N60),
                    xytext=(phi_tilde_N60+1, V_N60+0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12)
    
    # Set y-axis limit to focus on the interesting part
    ax.set_ylim(0, max(potential_values) * 1.1)
    
    # Find region of interest for x-axis
    if phi_tilde_end is not None and phi_tilde_N60 is not None:
        x_min = min(phi_tilde_end, phi_tilde_N60) - 5
        x_max = max(phi_tilde_end, phi_tilde_N60) + 5
        ax.set_xlim(x_min, x_max)
    
    fig.tight_layout()
    return fig

def phase_equations(t, state):
    """
    Define the system of first-order ODEs for the phase portrait
    state[0] = phi_tilde
    state[1] = dphi_tilde/dt
    
    Using:
    d²φ_tilde/dt² + 3H dφ_tilde/dt + dV/dφ_tilde = 0
    3M_p²H² = (1/2)(dφ_tilde/dt)² + V(φ_tilde)
    """
    phi_tilde, dphi_tilde_dt = state
    
    # Calculate potential and derivative
    V = V_canonical(phi_tilde, params)
    dV = dV_dphi_tilde(phi_tilde, params)
    
    # Calculate Hubble parameter
    H_squared = (0.5 * dphi_tilde_dt**2 + V) / (3 * M_p**2)
    H = np.sqrt(max(H_squared, 1e-10))  # Ensure H is positive
    
    # Calculate acceleration
    d2phi_tilde_dt2 = -3 * H * dphi_tilde_dt - dV
    
    return [dphi_tilde_dt, d2phi_tilde_dt2]

def plot_phase_portrait_uncompactified():
    """Create an uncompactified phase portrait"""
    # Define grid range
    phi_range = np.linspace(-10, 10, 100)
    dphi_range = np.linspace(-2, 2, 100)
    PHI, DPHI = np.meshgrid(phi_range, dphi_range)
    
    # Calculate vector field
    U = np.zeros_like(PHI)
    V = np.zeros_like(DPHI)
    
    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            phi_tilde = PHI[i, j]
            dphi_tilde_dt = DPHI[i, j]
            
            try:
                derivatives = phase_equations(0, [phi_tilde, dphi_tilde_dt])
                U[i, j] = derivatives[0]  # dphi_tilde/dt
                V[i, j] = derivatives[1]  # d²phi_tilde/dt²
            except:
                U[i, j] = 0
                V[i, j] = 0
    
    # Normalize vector field for better visualization
    speed = np.sqrt(U**2 + V**2)
    speed_norm = np.tanh(speed)  # maps values to (-1, 1)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Make divider for colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    # Contour of normalized speed
    contour = ax.contourf(
        PHI, DPHI, speed_norm,
        levels=50,
        cmap='viridis',
        vmin=0, vmax=1,
        alpha=0.8,
        extend='both',
        antialiased=False
    )
    
    # Add colorbar
    cbar = fig.colorbar(contour, cax=cax)
    cbar.set_label(r"Normalized flow speed")
    
    # Add streamlines
    ax.streamplot(
        phi_range, dphi_range, U, V,
        density=1.5,
        color='black',
        linewidth=0.7,
        arrowsize=1
    )
    
    # Set labels and title
    ax.set_xlim(min(phi_range), max(phi_range))
    ax.set_ylim(min(dphi_range), max(dphi_range))
    ax.set_xlabel(r"$\tilde{\varphi}/M_p$", fontsize=14)
    ax.set_ylabel(r"$d\tilde{\varphi}/dt /M_p^2$", fontsize=14)
    ax.set_title(r"Uncompactified Phase Portrait", fontsize=16)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    return fig

def compact_transform(x):
    """Transform to compactified coordinates: maps (-∞,∞) to (-1,1)"""
    return np.tanh(x)

def inverse_compact_transform(x):
    """Inverse transform: maps (-1,1) to (-∞,∞)"""
    return np.arctanh(x)

def plot_phase_portrait_compactified():
    """Create a phase portrait with compactified axes"""
    # Create mesh with compactified coordinates
    x_compact = np.linspace(-0.98, 0.98, 100)  # Avoid exact ±1 which causes issues
    y_compact = np.linspace(-0.98, 0.98, 100)
    X_compact, Y_compact = np.meshgrid(x_compact, y_compact)
    
    # Transform back to original coordinates
    X_orig = inverse_compact_transform(X_compact)
    Y_orig = inverse_compact_transform(Y_compact)
    
    # Calculate vector field
    U = np.zeros_like(X_compact)
    V = np.zeros_like(Y_compact)
    
    for i in range(X_compact.shape[0]):
        for j in range(X_compact.shape[1]):
            phi_tilde = X_orig[i, j]
            dphi_tilde_dt = Y_orig[i, j]
            
            try:
                # Get derivatives from our phase equations
                derivatives = phase_equations(0, [phi_tilde, dphi_tilde_dt])
                
                # These are dphi_tilde/dt and d²phi_tilde/dt²
                dphi = derivatives[0]
                d2phi = derivatives[1]
                
                # Calculate changes in compactified coordinates using chain rule
                jacobian_x = 1 - X_compact[i, j]**2
                U[i, j] = jacobian_x * dphi
                
                jacobian_y = 1 - Y_compact[i, j]**2
                V[i, j] = jacobian_y * d2phi
                
            except:
                U[i, j] = 0
                V[i, j] = 0
    
    # Compute speed and normalize
    speed = np.sqrt(U**2 + V**2)
    speed_norm = np.tanh(speed)  # maps values to (-1, 1)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Make divider for colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    # Contour of normalized speed
    contour = ax.contourf(
        X_compact, Y_compact, speed_norm,
        levels=50,
        cmap='viridis',
        vmin=0, vmax=1,
        alpha=0.8,
        extend='both',
        antialiased=False
    )
    
    # Add colorbar
    cbar = fig.colorbar(contour, cax=cax)
    cbar.set_label(r"Normalized flow speed")
    
    # Add streamlines
    ax.streamplot(
        x_compact, y_compact, U, V,
        density=1.5,
        color='black',
        linewidth=0.7,
        arrowsize=1
    )
    
    # Seed additional streamlines from a grid
    seed_x = np.linspace(-0.9, 0.9, 15)
    seed_y = np.linspace(-0.9, 0.9, 15)
    SX, SY = np.meshgrid(seed_x, seed_y)
    SEED_POINTS = np.vstack((SX.ravel(), SY.ravel())).T
    
    ax.streamplot(
        x_compact, y_compact, U, V,
        start_points=SEED_POINTS,
        integration_direction='both',
        color='black',
        linewidth=0.7,
        arrowsize=1
    )
    
    # Set labels and title
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"$\tilde{\varphi}/M_p$ (compactified)", fontsize=14)
    ax.set_ylabel(r"$d\tilde{\varphi}/dt/M_p^2$ (compactified)", fontsize=14)
    ax.set_title(r"Compactified Phase Portrait", fontsize=16)
    
    # Add custom ticks for reference
    compact_ticks = [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]
    orig_ticks = [inverse_compact_transform(x) for x in compact_ticks]
    orig_tick_labels = [f"{x:.1f}" for x in orig_ticks]
    
    ax.set_xticks(compact_ticks)
    ax.set_xticklabels(orig_tick_labels)
    ax.set_yticks(compact_ticks)
    ax.set_yticklabels(orig_tick_labels)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    return fig

def main():
    # First, optimize parameters to hit ns=0.9626 at N=60
    phi_tilde_end, phi_tilde_N60 = optimize_parameters_for_ns(target_ns=0.9626, target_N=60)
    
    # Plot the canonical potential
    fig_potential = plot_canonical_potential(phi_tilde_end, phi_tilde_N60)
    fig_potential.savefig('canonical_inflaton_potential.png', dpi=300, bbox_inches='tight')
    
    # Plot the uncompactified phase portrait
    fig_phase_uncomp = plot_phase_portrait_uncompactified()
    fig_phase_uncomp.savefig('uncompactified_phase_portrait.png', dpi=300, bbox_inches='tight')
    
    # Plot the compactified phase portrait
    fig_phase_comp = plot_phase_portrait_compactified()
    fig_phase_comp.savefig('compactified_phase_portrait.png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main()