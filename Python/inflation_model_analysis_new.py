import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# Configure matplotlib for publication quality plots
mpl.rcParams['axes.linewidth'] = 3.0   # make all axes spines 3 points thick
mpl.rcParams.update({
    "font.family":        "serif",       # for non‐math text
    "mathtext.fontset":   "cm",          # Computer Modern font
    "mathtext.rm":        "serif",       # roman (upright) math font
    "mathtext.it":        "serif:italic",
    "mathtext.bf":        "serif:bold",
})
"""Optimized parameters: [ 7.88416532  8.85847023 15.87685664 39.72118433  9.27322142  8.20535856
  2.06594146  0.71201353  5.84319039  3.13182862]
Final n_s= 0.9627741531363996"""

# Constants (these are parameters you might want to adjust)
M_p   = 1.0
M     = np.sqrt(2) * M_p
alpha = 5.37904158 
beta  = 4.84572706 
gamma = -1.2378985
phi_0  = 10
k     = 46.47655135  

kappa = 42.92091333
omega = -7.74347196
mu    = 7.12605145 
chi   = 9.67061868
lam   = 0.31666247 
#params = [ 7.12605145,  1, 5.37904158, 4.84572706,  -1.2378985,  46.47655135, -7.74347196,  42.92091333, 0.31666247,  7.12605145]
"""[ 5.37904158 -1.2378985   4.84572706 46.47655135  7.12605145 -7.74347196
  9.67061868  0.31666247 42.92091333]
[alpha, gamma, beta, k, mu, omega, chi, lambda_val, kappa]"""
# [m, phi0, a, b, gamma, k, li, ka, la, om]
#mu, phi0, alpha, beta, gamma, k , li, ka, la, om = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9]
def phi_tilde(phi):
    """Transform phi to phi_tilde according to the given equation"""
    term1 = np.sqrt(4*alpha*beta - gamma**2) * np.tan(np.sqrt(2/(12+k)) * phi/M_p)
    return (phi_0/(2*beta)) * (term1 - gamma)

def potential(phi):
    """Calculate V(phi) by first transforming phi to phi_tilde"""
    phi_t = phi_tilde(phi)
    
    # Numerator terms
    term1 = kappa * phi_0**4
    term2 = 2 * omega * phi_0**3 * phi_t
    term3 = mu**2 * phi_0**2 * phi_t**2
    term4 = 2 * chi * phi_0 * phi_t**3
    term5 = lam * phi_t**4
    
    # Denominator
    denom = 2 * (beta * phi_t**2 + gamma * phi_0 * phi_t + alpha * phi_0**2)**2
    
    return M**4 * (term1 + term2 + term3 + term4 + term5) / denom

def dV_dphi(phi):
    """Numerical derivative of V with respect to phi"""
    epsilon = 1e-6
    return (potential(phi + epsilon) - potential(phi - epsilon)) / (2 * epsilon)

def phase_equations(N, state):
    """
    Define the system of first-order ODEs for the phase portrait
    state[0] = phi
    state[1] = dphi/dN
    
    From your equation with K=1:
    d^2φ/dN^2 + 3 dφ/dN - (1/2M_p^2)(dφ/dN)^3 + (dK/dφ)/2 * (dφ/dN)^2 + (3M_p^2 - (1/2)(dφ/dN)^2) * d(ln V)/dφ = 0
    
    Since K=1, dK/dφ = 0, the equation simplifies to:
    d^2φ/dN^2 + 3 dφ/dN - (1/2M_p^2)(dφ/dN)^3 + (3M_p^2 - (1/2)(dφ/dN)^2) * d(ln V)/dφ = 0
    """
    phi, dphi_dN = state
    
    # Calculate d(ln V)/dφ = (1/V) * dV/dφ
    V_val = potential(phi)
    dV_val = dV_dphi(phi)
    dln_V = dV_val / V_val if abs(V_val) > 1e-10 else 0
    
    # Calculate d^2φ/dN^2 from your equation
    d2phi_dN2 = (
        -3 * dphi_dN 
        + (1/(2*M_p**2)) * dphi_dN**3
        - (3*M_p**2 - 0.5 * dphi_dN**2) * dln_V
    )
    
    return [dphi_dN, d2phi_dN2]

def plot_potential():
    """Plot the inflaton potential V(φ) versus φ"""
    prefac = np.sqrt((12 + k) / 2) * M_p
    # Define phi range
    # We need to avoid values that make tan(phi) singular
    phi_range = np.linspace(-prefac * np.pi/2, prefac* np.pi/2, 100000)
    
    potential_values = potential(phi_range)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(phi_range, potential_values, 'b-', linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$\varphi/M_p$', fontsize=14)
    ax.set_ylabel(r'$V(\varphi)/M^4$', fontsize=14)
    ax.set_title('Inflaton Potential', fontsize=16)
    
    # Find potential minimum for visual reference
    min_idx = np.argmin(potential_values)
    ax.plot(phi_range[min_idx], potential_values[min_idx], 'ro')
    ax.annotate(f'Minimum at φ ≈ {phi_range[min_idx]:.3f}', 
                xy=(phi_range[min_idx], potential_values[min_idx]),
                xytext=(phi_range[min_idx]+0.2, potential_values[min_idx]+0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12)
    
    fig.tight_layout()
    return fig

def compact_transform(x):
    """Transform to compactified coordinates: maps (-∞,∞) to (-1,1)"""
    return np.tanh(x)

def inverse_compact_transform(x):
    """Inverse transform: maps (-1,1) to (-∞,∞)"""
    return np.arctanh(x)

def plot_phase_portrait():
    """Create a phase portrait with compactified axes"""
    # Create mesh with compactified coordinates
    x_compact = np.linspace(-0.98, 0.98, 200)  # Avoid exact ±1 which causes issues
    y_compact = np.linspace(-0.98, 0.98, 200)
    X_compact, Y_compact = np.meshgrid(x_compact, y_compact)
    
    # Transform back to original coordinates
    X_orig = inverse_compact_transform(X_compact)
    Y_orig = inverse_compact_transform(Y_compact)
    
    # Calculate vector field
    U = np.zeros_like(X_compact)
    V = np.zeros_like(Y_compact)
    
    for i in range(X_compact.shape[0]):
        for j in range(X_compact.shape[1]):
            phi = X_orig[i, j]
            dphi_dN = Y_orig[i, j]
            
            try:
                # Get derivatives from our phase equations
                derivatives = phase_equations(0, [phi, dphi_dN])
                
                # These are the actual dφ/dN and d²φ/dN²
                dphi = dphi_dN
                d2phi = derivatives[1]
                
                # Calculate how these map to our compactified coordinates
                # Chain rule: dx_compact/dN = (dx_compact/dx_orig) * (dx_orig/dN)
                # For tanh(x): d/dx[tanh(x)] = 1 - tanh²(x)
                
                # Calculate U component (change in compactified phi)
                jacobian_x = 1 - X_compact[i, j]**2
                U[i, j] = jacobian_x * dphi
                
                # Calculate V component (change in compactified dphi/dN)
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
    cbar.set_label(r"Flow speed")
    
    # Add streamlines
    ax.streamplot(
        x_compact, y_compact, U, V,
        density=1.5,
        color='black',
        linewidth=0.7,
        arrowsize=1,
        maxlength=0.1
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
    ax.set_xlabel(r"$\varphi/M_p$ (compactified)", fontsize=14)
    ax.set_ylabel(r"$\frac{d\varphi}{dN}$ (compactified)", fontsize=14)
    ax.set_title(r"Phase Portrait of Inflaton Dynamics", fontsize=16)
    
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
    # Plot the potential
    fig_potential = plot_potential()
    fig_potential.savefig('inflaton_potential.png', dpi=300, bbox_inches='tight')
    
    # Plot the phase portrait
    fig_phase = plot_phase_portrait()
    fig_phase.savefig('inflaton_phase_portrait.png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main()