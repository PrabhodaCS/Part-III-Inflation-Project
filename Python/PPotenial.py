"""/**
 * @author Prabhoda CS
 * @email pcs52@cam.ac.uk
 * @create date 01-03-2025 18:44:19
 * @modify date 01-03-2025 18:44:19
 * @desc [description]
 */"""

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.integrate import solve_ivp

# Parameters
Nend = 70 #end of efolds
# --- Your fixed model parameters ---
m    = 1.0
M    = 7
phi0 = 20
a    = 50.0
b    = 10.0
c    = 0    # your 'd' constant from above
# Domain for phi
g = 30
D = np.sqrt( - g**2 / 4 + a*b)
k = 1 # this is sqrt((k-6\beta)/2\beta)
mo = 1000000
c =  0  # integration constant (minima @ 0c + displacement)  #  add this later -f*phi0*np.arcsinh(-s/(2*f)) +

eh = 0.2 #initial slow roll parameter (1/2)*(dp/dN)**2

# integration constant (minima @ 0c + displacement)  #  add this later -f*phi0*np.arcsinh(-s/(2*f)) +

# Symbolic expressions for potential and derivatives
p_s = symbols('phi')
x = p_s /(np.sqrt(6)*M) + c
Xp = (1/(4*b)) * exp(-x)*(exp(2*x) - 2*exp(x)*g*phi0 - phi0**2 * D**2)
Vp = (m**2 * phi0**2 * Xp**2)/(2* (b*Xp**2 + g*phi0*Xp + a*phi0**2)**2)

print("The potential V(φ) is:", Vp)

print("\n")


V_ = diff(Vp, p_s)
V__ = diff(V_,p_s)
lnV = ln(Vp)
dlnV = diff(lnV, p_s)
print("First derivative of log(V):", dlnV)

print("\n")

dV = lambdify(p_s, V_, 'numpy')
ddV = lambdify(p_s, V__, 'numpy')

# Convert symbolic expressions to numerical functions
Vi = lambdify(p_s, Vp, 'numpy')
dlnV = lambdify(p_s, dlnV, 'numpy')  # derivative of log(V)

# Define the system of differential equations where N = #of efolds and V is the vector containing [phi, dphi]
def de(N, V):
    p, p_ = V
    lv = dlnV(p)
    dp2_dN2 = -3 * p_ - 0.5 * p_**3 + (3 - 0.5 * p_**2) * lv
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

e = 0.5 * ps_**2 # Hubble slow-roll parameter

sr = []
sr_ = []
for i in range(len(e)):
    if e[i] < 1:
        sr.append(ps[i])
        sr_.append(ps_[i])

sr = np.array(sr)
sr0 = np.zeros(len(Nr)-len(sr))
sr = np.append(sr,sr0)
sr_ = np.array(sr_)
sr_0 = np.zeros(len(Nr)-len(sr_))
sr_ = np.append(sr_,sr_0)

# Define a range of phi values
phi = np.linspace(0, 100, 5000)

exit_index = np.argmax(e >= 1)  # first index where slow-roll fails
phi_exit_H = ps[exit_index]
print("Slow-roll exit (Hubble parameter) occurs at φ =", phi_exit_H)

print("\n")

# Define slow-roll parameters
def epsilon(phi):
    return 0.5 * (dV(phi) / Vi(phi))**2 

def eta(phi):
    return ddV(phi) / Vi(phi)

# Compute values for potential and slow-roll parameters
v1 = Vi(phi)
ep = epsilon(phi) 
et = abs(eta(phi))

# Isolating parts of graph s.t., slow roll conditions are satisfied
L = []
for i in phi:
    if epsilon(i) < 1 and abs(eta(i)) < 1:
        L.append(i)


L = np.array(L)
Mv = Vi(L)

vmin = np.min(v1)


# Compute slow-roll parameters over the range
eps_values = epsilon(phi)
eta_values = np.abs(eta(phi))

# Define slow-roll condition: both epsilon < 1 and |eta| < 1
slow_mask = (eps_values < 1) & (eta_values < 1)


plt.plot(phi, v1, label="V(φ)")
plt.title("Potential and Slow-Roll Parameters")

"""
plt.plot(L, Mv,'o', label="Slow rolling V(φ)")
plt.title("Potential and Slow-Roll Parameters")
"""

# Highlight the slow-roll region using fill_between
plt.fill_between(phi, v1, vmin, where=slow_mask, color='orange', alpha=0.3, label="Slow Roll Region")


plt.xlabel("φ")
plt.ylabel("V(φ)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\New Potenial with gravity")
plt.show()

#Plot slow roll parameters, epsilon and eta
plt.figure(figsize=(10, 6))
plt.plot(phi, ep, label=r"$\epsilon$")
#plt.plot(phi, et, label=r"$\eta$")
plt.title("Slow Roll Parameter: $\epsilon$ vs. Field ($\phi$)")
plt.xlabel("$\phi$")
plt.ylabel(r"$\epsilon$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot the field evolution φ(N)
plt.figure(figsize=(10, 6))
plt.plot(Nr, ps, label=r"$\phi(N)$")
plt.plot(Nr, sr, 'o', label = "Slow roll conditions met")
plt.title("Field Evolution: $\phi$ vs. Number of e-folds ($N$)")
plt.xlabel("N")
plt.ylabel(r"$\phi$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\NewField as a function of N")
plt.show()

# Plot the velocity evolution dφ(N)
plt.figure(figsize=(10, 6))
plt.plot(Nr, ps_, label=r"$\phi(N)$")
#plt.plot(Nr, sr_, 'o', label = "Slow roll conditions met")
plt.title("Field Evolution: d$\phi$/dN vs. Number of e-folds ($N$)")
plt.xlabel("N")
plt.ylabel(r"$\phi$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\NewField velocity as a function of N")
plt.show()
