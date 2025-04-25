"""
/**
 * @author Prabhoda CS
 * @email pcs52@cam.ac.uk
 * @create date 15-03-2025 20:28:03
 * @modify date 15-03-2025 20:28:03
 * @desc [description]
 */
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sympy import *

Nend = 70         # End of efolds
m = 100             # mass of inflaton
M = 20            # Planck mass
Mp = sqrt(2)*M   # Actual Planck mass (lol)
phi0 = 1        # non-propagating field 
gamma = 1             # model parameter gamma
alpha = 1            # model parameter alpha
beta = -1             # model parameter beta
D = np.sqrt(gamma**2 - 4*alpha*beta) 
k = 6             # model parameter k
sf = 1000000      # overall scale factor in potential
c = 0            # integration constant 
la = -10
ka = -1

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
sol = solve_ivp(dtilde_dphi, phi_span, [c], t_eval=np.linspace(phi_min, phi_max, 10001), method='DOP853', rtol=1e-9, atol=1e-12)

print(K(phi_span[0]), K(phi_span[1]))

phi_vals = sol.t         # these are the phi values
tilde_vals = sol.y[0]     # corresponding canonical field values

print("beginning value of canonical phi : ",tilde_vals[0],"\n","End value  of canonical phi : ", tilde_vals[-1])

def V(phi):
    return  0.5 * ((m*phi0*phi)**2 + la*phi**4 + ka*phi0**4) /A_func(phi)**2

# If you want to plot V as a function of the canonical field, you need to know phi as a function of tilde.
# We can build an interpolation function from tilde_vals vs. phi_vals.
phi_of_tilde = interp1d(tilde_vals, phi_vals, kind='cubic', fill_value="extrapolate")

# Specify the range for the canonical field you want to plot:
tilde_start = 140 # lower limit for canonical field
tilde_end   = 350  # upper limit for canonical field

if not np.all(np.diff(tilde_vals) > 0):
    print("Warning: tilde_vals is not strictly increasing!")
else:
    print("tilde_vals is strictly increasing.")

# Generate canonical field values in that range:
tilde_plot = np.linspace(tilde_start, tilde_end, 50000)
phi_plot = phi_of_tilde(tilde_plot)
V_plot   = V(phi_plot)

"""print(phi_vals)
print("\n")
print(tilde_vals)
"""

dV = np.gradient(V_plot, tilde_plot)
ddV = np.gradient(dV, tilde_plot)

ratio = max(dV)/max(V_plot)
ratio2 = max(ddV)/max(V_plot)

print("Ratio of dV to V maxima is: ", ratio)
print("Ratio of d^2V to V maxima is: ", ratio2)
print("Ratio of d^2V to dV maxima is: ", ratio2/ratio)

#Defining slow roll parameters : 
epsilon = 0.5 * (dV/V_plot)**2
eta = abs(ddV/V_plot)

print(max(epsilon), max(eta))

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
    (r"$c$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', gamma),
    (r'$\alpha$', alpha),
    (r'$D$', D),
    (r'$k$', k),
    (r'$\lambda$', la),
    (r'$\kappa$', ka)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')


plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\Field vs Canonical Field.png")
plt.show()

Nr = []
for i in range(len(epsilon)):
    if epsilon[i] < 1:
       Nr.append(tilde_plot[i])

print(len(Nr),len(tilde_plot))

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tilde_plot, V_plot, label=r"$V(\tilde{\varphi})$")
ax.plot(tilde_plot, dV/ratio, label=r"$dV/d\tilde{\varphi}$  (Scaled)")
ax.plot(tilde_plot, ddV/ratio2, label=r"$d^2V/d\tilde{\varphi}^2$  (Scaled)")
ax.set_xlabel(r"Canonical field $\tilde{\varphi}$")
ax.set_ylabel(r"$V(\tilde{\varphi})$")
ax.set_title("Potential vs. Canonical Field")
ax.legend()
ax.grid(True)

mask = (epsilon > 1) & (eta > 1)

# Check that there is at least one point satisfying the condition
if np.any(mask):
    slow_roll_min = np.min(tilde_plot[mask])
    slow_roll_max = np.max(tilde_plot[mask])
    ax.axvspan(slow_roll_min, slow_roll_max, color='grey', alpha=0.3, label="Slow-roll region")
    print(f"Slow-roll region: {slow_roll_min:.2f} < tilde_phi < {slow_roll_max:.2f}")
else:
    print("No slow-roll region found in the specified canonical range.")


plt.tight_layout()

# Parameters to display
parameters = [
    (r'$\beta$', beta),
    (r"$c$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', gamma),
    (r'$\alpha$', alpha),
    (r'$D$', D),
    (r'$k$', k),
    (r'$\lambda$', la),
    (r'$\kappa$', ka)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')


plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\Full Lagrangian potential.png")
plt.show()

ep_start = -50
ep_end = 50

mask = (epsilon < 1) & (eta < 1)
epsilon_masked = np.ma.masked_where(~mask, epsilon)
eta_masked = np.ma.masked_where(~mask, eta)

# --- Plot Slow-Roll Parameters vs Canonical Field ---
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tilde_plot, epsilon_masked, label=r"$\epsilon$")
ax.plot(tilde_plot, eta_masked, label=r"$\eta$")
ax.set_xlabel(r"Canonical field $\tilde{\varphi}$")
ax.set_ylabel("Slow-roll parameters")
ax.set_title("Slow-roll Parameters vs. Canonical Field")
ax.legend()
ax.grid(True)

# Parameters to display
parameters = [
    (r'$\beta$', beta),
    (r"$c$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', gamma),
    (r'$\alpha$', alpha),
    (r'$D$', D),
    (r'$k$', k),
    (r'$\lambda$', la),
    (r'$\kappa$', ka)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')


plt.tight_layout()
plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\Full Slow Roll Parameters.png")
plt.show()