import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sympy import *

Nend = 70         # End of efolds
m = 1             # mass of inflaton
M = 20            # Planck mass
Mp = sqrt(2)*M   # Actual Planck mass (lol)
phi0 = 30        # non-propagating field 
gamma = 100             # model parameter gamma
alpha = -50            # model parameter alpha
beta = 0.1             # model parameter beta
D = np.sqrt(gamma**2 - 4*alpha*beta) 
k = 1000             # model parameter k
sf = 1000000      # overall scale factor in potential
c = 0            # integration constant 

p_s = symbols('phi')

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
phi_min = 0
phi_max = 40

# Solve the ODE numerically to get tilde_phi as a function of phi
phi_span = (phi_min, phi_max)
# We'll solve the ODE using solve_ivp. We use phi as the independent variable.
sol = solve_ivp(dtilde_dphi, phi_span, [c], t_eval=np.linspace(phi_min, phi_max, 1001), method='DOP853', rtol=1e-9, atol=1e-12)

print(K(phi_span[0]), K(phi_span[1]))

phi_vals = sol.t         # these are the phi values
tilde_vals = sol.y[0]     # corresponding canonical field values

print("beginning value of canonical phi : ",tilde_vals[0],"\n","End value  of canonical phi : ", tilde_vals[-1])

def V(phi):
    return 0.5 * (m*phi0)**2 * (phi**2)/A_func(phi)**2

# If you want to plot V as a function of the canonical field, you need to know phi as a function of tilde.
# We can build an interpolation function from tilde_vals vs. phi_vals.
phi_of_tilde = interp1d(tilde_vals, phi_vals, kind='cubic', fill_value="extrapolate")

# Specify the range for the canonical field you want to plot:
tilde_start = -3000 # lower limit for canonical field
tilde_end   = 500  # upper limit for canonical field

if not np.all(np.diff(tilde_vals) > 0):
    print("Warning: tilde_vals is not strictly increasing!")
else:
    print("tilde_vals is strictly increasing.")

# Generate canonical field values in that range:
tilde_plot = np.linspace(tilde_start, tilde_end, 500)
phi_plot = phi_of_tilde(tilde_plot)
V_plot   = V(phi_plot)

phi_vals1 = np.linspace(-50, 30, 1001)

V_normal = V(phi_vals1)

tvals = np.linspace(0, 1, 101)
"""print(phi_vals)
print("\n")
print(tilde_vals)
"""

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(phi_vals, tilde_vals)
ax.set_xlabel(r"Field $\varphi$")
ax.set_ylabel(r"Canonical field $\tilde{\varphi}$")
ax.set_title(r"$\tilde{\varphi}$ vs. $\varphi$")
ax.legend()
ax.grid(True)
plt.tight_layout()

# Parameters to display
parameters = [
    (r'$\beta$', beta),
    (r"$c'$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', gamma),
    (r'$\alpha$', alpha),
    (r'$D$', D)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')


plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\Field vs Canonical Field.png")
plt.show()


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tilde_plot, V_plot, label=r"$V(\tilde{\varphi})$")
ax.set_xlabel(r"Canonical field $\tilde{\varphi}$")
ax.set_ylabel(r"$V(\tilde{\varphi})$")
ax.set_title("Potential vs. Canonical Field")
ax.grid(True)
plt.tight_layout()

# Parameters to display
parameters = [
    (r'$\beta$', beta),
    (r"$c'$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', gamma),
    (r'$\alpha$', alpha),
    (r'$D$', D)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')


plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\Full Lagrangian potential.png")
plt.show()



"""
plt.figure(figsize=(8,6))
plt.plot(phi_vals1, V_normal, label=r"$V(\varphi)$")
plt.xlabel(r"Field $\varphi$")
plt.ylabel(r"$V$")
plt.title("Potential vs. Field")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""