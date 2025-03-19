"""
/**
 * @author Prabhoda CS
 * @email pcs52@cam.ac.uk
 * @create date 15-03-2025 20:29:16
 * @modify date 15-03-2025 20:29:16
 * @desc [description]
 */
"""

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec
import math

# Parameters
Nend = 60         # End of efolds
m = 1             # mass of inflaton
M = 1            # Planck mass
Mp = math.sqrt(2)*M   # Actual Planck mass (lol)
phi0 = -529.7076        # non-propagating field 
g = -185.1555             # model parameter gamma
a = -52.4048            # model parameter alpha
b = -0.4628           # model parameter beta
D = np.sqrt(g**2 - 4*a*b)  
k = 27.3976             # model parameter k
sf = 1000000      # overall scale factor in potential
c = 0            # integration constant 

eh = 0.02 #initial slow roll parameter (1/2)*(dp/dN)**2

# Symbolic expressions for potential and derivatives
p_s = symbols('phi')
A = a*phi0**2 + g*phi0*p_s + b*p_s**2

K = - 36*(M*b)**2/A**2 + (k-6)*(phi0*M*D)**2/(2*A**2)
V =  (m*phi0)**2 * (p_s/A)**2 / 2

Kp = diff(K, p_s)
V_ = diff(V, p_s)
V__ = diff(V_,p_s)
lnV = ln(V)
dlnV = diff(lnV, p_s)
print("First derivative of log(V):", dlnV)

dK = lambdify(p_s, Kp, 'numpy')
dV = lambdify(p_s, V_, 'numpy')
ddV = lambdify(p_s, V__, 'numpy')

# Convert symbolic expressions to numerical functions
Ki = lambdify(p_s, K, 'numpy')
Vi = lambdify(p_s, V, 'numpy')
dlnV = lambdify(p_s, dlnV, 'numpy')  # derivative of log(V)


# Define the system of differential equations where N = #of efolds and V is the vector containing [phi, dphi]
def de(N, V):
    p, p_ = V
    lv = dlnV(p)
    Ka = Ki(p)
    Kd = dK(p)

    dp2_dN2 = 1/Ka * (-3 * Ka * p_ + 0.5*Ka**2/Mp**2 * p_**3 - 0.5 * Kd * p_**2 + (3*Mp**2 - Ka* 0.5 * p_**2) * lv)
    return [p_, dp2_dN2]

# Initial conditions
p0 = 1000  # Initial field value
p_0 = 1  # Initial dphi/dN (velocity)

V0 = [p0, p_0]  # Initial conditions vector

# Solve the system using solve_ivp
Nr = np.linspace(0, Nend, 1000)  # Number of e-folds
sol = solve_ivp(de, (0, Nend), V0, t_eval=Nr, method='RK45', rtol=1e-6, atol=1e-8)

# Extract the solutions
ps = np.array(sol.y[0])  # φ(N)
ps_ = np.array(sol.y[1])  # dφ/dN

V_n = np.array([float(Vi(p)) for p in ps], dtype=np.float64)  # Potential as function of φ
K_n = np.array([float(Ki(p)) for p in ps], dtype=np.float64)  # K as function of φ

H = np.sqrt(V_n/(3*Mp**2 - 0.5*K_n*ps_**2))  # Hubble parameter as function of N


epsilonh = Mp**2/2 * K_n * ps_**2
etah = epsilonh - 1/(2*epsilonh) * np.gradient(epsilonh, Nr)


# Find index closest to N = 60
idx_expN = np.argmin(np.abs(Nr - Nend))

    # Convert from Hubble slow-roll parameters to Potential slow-roll parameters
epsilonV = epsilonh / (3 - etah)**2
etaV = (etah + epsilonh) / (3 - etah)

# Compute the final spectral tilt
ns_potential = 1 - 6 * epsilonV[idx_expN] + 2 * etaV[idx_expN]

print(f"Final tilt value (Potential Slow-Roll): ns = {ns_potential:.6f}")

# Main plot in the first column:# Define the starting point
N_start = 10

# Find the index where Nr >= N_start
start_index = np.searchsorted(Nr, N_start)

# Subset the arrays from start_index onwards
Nr_subset = Nr[start_index:]

"""
epsilon_subset = epsilon[start_index:]
eta_subset = eta[start_index:]

# Plotting the slow-roll parameters from N = 10 onwards
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(Nr_subset, epsilon_subset, label=r"$\epsilon$")
ax.plot(Nr_subset, eta_subset, label=r"$\eta$")
ax.set_xlabel("Number of e-folds ($N$)")
ax.set_ylabel("Slow-roll parameters")
ax.set_title("Slow-roll Parameters vs. Number of e-folds ($N$) from $N=10$ onwards")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Slow-Roll Parameters vs Canonical Field ---
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(Nr, epsilon, label=r"$\epsilon$")
ax.plot(Nr, eta, label=r"$\eta$")
ax.set_xlabel("Number of e-folds ($N$)")
ax.set_ylabel("Slow-roll parameters")
ax.set_title("Slow-roll Parameters vs. Number of e-folds ($N$)")
ax.legend()
ax.grid(True)


# Parameters to display
parameters = [
    (r'$\beta$', b),
    (r"$c$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', g),
    (r'$\alpha$', a),
    (r'$D$', D),
    (r'$k$', k)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')


plt.tight_layout()
plt.show()

"""

# Main plot in the first column:# Define the starting point
N_start = 1
epsilonh_subset = epsilonh[start_index:]
etah_subset = etah[start_index:]


# ---Hubble Slow-Roll Parameters vs Canonical Field ---
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(Nr_subset, epsilonh_subset, label=r"$\epsilon$")
ax.plot(Nr_subset, etah_subset, label=r"$\eta$")
ax.set_xlabel("Number of e-folds ($N$)")
ax.set_ylabel("Slow-roll parameters")
ax.set_title("Slow-roll Parameters vs. Number of e-folds ($N$)")
ax.legend()
ax.grid(True)


# Parameters to display
parameters = [
    (r'$\beta$', b),
    (r"$c$", c),
    (r'$\mu$', m),
    (r'$\varphi_0$', phi0),
    (r'$\gamma$', g),
    (r'$\alpha$', a),
    (r'$D$', D),
    (r'$k$', k)
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    ax.text(0.02, 0.95 - i*0.05, f'{name} = {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\Full Slow Roll Parameters as Func of N.png")

plt.tight_layout()
plt.show()