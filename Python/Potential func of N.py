"""
/**
 * @author Prabhoda CS
 * @email pcs52@cam.ac.uk
 * @create date 10-03-2025 18:12:54
 * @modify date 10-03-2025 18:12:54
 * @desc [description]
 */
"""

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.integrate import solve_ivp

# Parameters
Nend = 70         # End of efolds
m = 1             # mass of inflaton
M = 20            # Planck mass
Mp = sqrt(2)*M   # Actual Planck mass (lol)
phi0 = 3        # non-propagating field 
g = 20             # model parameter gamma
a = -50            # model parameter alpha
b = 1             # model parameter beta
D = np.sqrt(g**2 - 4*a*b)  
k_ = 1             # k_ = sqrt(2\beta / k-6\beta
sf = 1000000      # overall scale factor in potential
c = -2            # integration constant 

eh = 0.02 #initial slow roll parameter (1/2)*(dp/dN)**2

# Symbolic expressions for potential and derivatives
p_s = symbols('phi')
A = a*phi0**2 + g*phi0*p_s + b*p_s**2

K = - 36*(M*b)**2/A**2 + (phi0*M*k_*D)**2/(2*A**2)
V = - (m*phi0)**2 * (p_s/A)**2 / 2

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
p0 = 1.5  # Initial field value
p_0 = 0.1  # Initial dphi/dN (velocity)


V0 = [p0, p_0]  # Initial conditions vector

# Solve the system using solve_ivp
Nr = np.linspace(0, Nend, 1000)  # Number of e-folds
sol = solve_ivp(de, (0, Nend), V0, t_eval=Nr, method='RK45', rtol=1e-6, atol=1e-8)

# Extract the solutions
ps = sol.y[0]  # φ(N)
ps_ = sol.y[1]  # dφ/dN


# Plot the field evolution φ(N)
plt.figure(figsize=(10, 6))
plt.plot(Nr, ps, label=r"$\phi(N)$")
#plt.plot(Nr, sr, 'o', label = "Slow roll conditions met")
plt.title("Field Evolution: $\phi$ vs. Number of e-folds ($N$)")
plt.xlabel("N")
plt.ylabel(r"$\phi$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\New Field as a function of N")
plt.show()
