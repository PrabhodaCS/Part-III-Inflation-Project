import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sympy import *

# Define model parameters (set these to your model's numbers)


Nend = 70         # End of efolds
m = 1             # mass of inflaton
M = 20            # Planck mass
Mp = sqrt(2)*M   # Actual Planck mass (lol)
phi0 = 3        # non-propagating field 
gamma = 20             # model parameter gamma
alpha = -50            # model parameter alpha
beta = 0.00001             # model parameter beta
D = np.sqrt(gamma**2 - 4*alpha*beta) 
k = 10             # model parameter k
sf = 1000000      # overall scale factor in potential
c = -2            # integration constant 

# Define A(phi) = beta*phi^2 + gamma*phi0*phi + alpha*phi0^2.
def A_func(phi):
    return beta * phi**2 + gamma * phi0 * phi + alpha * phi0**2

# Define K(phi) as given
def K(phi):
    A_val = A_func(phi)
    return M**2 * ( -36*beta/A_val + (phi0**2*(gamma**2 - 4*beta*alpha)*(k-6))/(2*A_val**2) )

# Define the ODE for the canonical field: dtilde_phi/dphi = sqrt(K(phi))
def dtilde_dphi(phi, tilde):
    return np.sqrt(K(phi))

# Choose a range for phi over which you want to integrate:
phi_min = -10
phi_max = 10

# Solve the ODE numerically to get tilde_phi as a function of phi
phi_span = (phi_min, phi_max)
# We'll solve the ODE using solve_ivp. We use phi as the independent variable.
sol = solve_ivp(dtilde_dphi, phi_span, [0.0], t_eval=np.linspace(phi_min, phi_max, 1000))

print(K(phi_span[0]), K(phi_span[1]))

phi_vals = sol.t         # these are the phi values
tilde_vals = sol.y[0]     # corresponding canonical field values

print("beginning value of canonical phi : ",tilde_vals[0],"\n","End value  of canonical phi : ", tilde_vals[-1])

# Now, suppose we have a potential V(phi). For demonstration, let’s assume
def V(phi):
    # Example: a simple quadratic potential (replace with your model)
    return 0.5 * (m*phi0)**2 * (phi**2)/A_func(phi)**2

# If you want to plot V as a function of the canonical field, you need to know phi as a function of tilde.
# We can build an interpolation function from tilde_vals vs. phi_vals.
phi_of_tilde = interp1d(tilde_vals, phi_vals, kind='cubic', fill_value="extrapolate")

# Specify the range for the canonical field you want to plot:
tilde_start = -40 # set your desired lower limit for canonical field
tilde_end   = 40  # set your desired upper limit for canonical field

# Generate canonical field values in that range:
tilde_plot = np.linspace(tilde_start, tilde_end, 500)
phi_plot = phi_of_tilde(tilde_plot)
V_plot   = V(phi_plot)

# Plot the potential as a function of the canonical field:
plt.figure(figsize=(8,6))
plt.plot(tilde_plot, V_plot, label=r"$V(\tilde{\varphi})$")
plt.xlabel(r"Canonical field $\tilde{\varphi}$")
plt.ylabel(r"$V$")
plt.title("Potential vs. Canonical Field")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
