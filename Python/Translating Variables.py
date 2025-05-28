import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.integrate import quad

a = 11.5 
b = 1.5   # PLOT EDGE

# Define a range of φ and ω values
omega = np.linspace(-b, a, 500)
#phi = omega + f*phi0*np.arctanh(4*b/np.sqrt(Mp**4 + 16*b**2)) + c  # chosen f*phi0 = 1, so that (d phi)**2 = (d omega)**2
phi = np.linspace(-b, a, 500)

# Free Parameters
Mp = 1                  #PLANCK MASS = 1

b = -8 * Mp**2                #$\beta$
c_ = 2                  #c' as defined in Silvio

#Derived Parameters [Salvio --> Barker]
"""m = Mp**2 / (24*np.sqrt(2*c_))
phi0 = np.sqrt(24) / Mp 
v = Mp**4 / 16 + b**2
s = -2*b
f = Mp**2 / 4       #derived from v and s as f = np.sqrt (v - s**2 / 4)
c = -Mp*np.sqrt(3/2)*np.arctanh(4 * b / np.sqrt(Mp ** 4 + 16 * b ** 2))
"""
#Another set of defns (Aligned units) (with an extra g factor, ambiguity in variables)
g = np.sqrt(1/3)               #ratio of phi0/(np.sqrt(3/2)*Mp) (without numerical factor, it relates what fraction of the planck mass phi0 is)      

# g < np.sqrt(2/3) =>  phi0 < Mp

m = 1 / (6 * np.sqrt(2*c_)) * g**(-1)
phi0 = np.sqrt(3/2) * Mp * g**(1)
v = (1 + 16*b**2/Mp**4) * g**(-2)
s = -8*b/Mp**2 * g**(-1)
f = g**(-1)       #derived from v and s as f = np.sqrt (v - s**2 / 4)
c = -Mp*np.sqrt(3/2)*np.arctanh(4 * b / np.sqrt(Mp ** 4 + 16 * b ** 2))

# Define potentials
p_s = symbols('phi')
V1 = (m ** 2) * (phi0 ** 4) / 2 * (s / 2 + f * sinh((p_s - c) / (f * phi0))) ** 2
print("The potential V(φ) is:", V1)

w_s = symbols('omega')
X = np.sqrt(2 / 3) * w_s / Mp + np.arctanh(4 * b / np.sqrt(Mp ** 4 + 16 * b ** 2))
V2 = 1 / (4 * c_) * (Mp ** 2 / 4 * sinh(X) - b) ** 2
print("The potential V(ω) is:", V2)

# Computing first and second derivatives of V
V_ = diff(V1, p_s)
V__ = diff(V_, p_s)

# Convert symbolic expressions to numerical functions
V1p = lambdify(p_s, V1, 'numpy')
dV = lambdify(p_s, V_, 'numpy')
ddV = lambdify(p_s, V__, 'numpy')


# Define slow-roll parameters
def epsilon(phi):
    return 0.5 * (dV(phi) / V1p(phi))**2

def eta(phi):
    return ddV(phi) / V1p(phi)

ep = epsilon(phi)
et = abs(eta(phi))

# Isolating parts of graph s.t., slow roll conditions are satisfied
L = []
for i in phi:
    if epsilon(i) < 1 and abs(eta(i)) < 1:
        L.append(i)

L = np.array(L)         #Slow roll region of field
Mv = V1p(L)             #Slow roll region of potential

def dN(phi):
    N = V1p(phi) / dV(phi)
    return N

Nf, _ = quad(dN, L[0], L[-1])  # Returns the integral and an estimate of the error
print(f"Number of e-folds (numerical integration): {Nf:.3f}")

"""
def Ni(s,e):     #Calculating efolds from phi_s to phi_e
    N = V1 / V_.simplify()
    Ni = integrate(N, (p_s, e, s))  # Sympy integration
    return Ni.evalf()

Nf = Ni(L[-1], L[0])
print(f"Number of e-folds (symbolic integration): {Nf:.3f}")
"""

# Convert symbolic expressions to numerical functions
Vi = lambdify(p_s, V1, 'numpy')
Vf = lambdify(w_s, V2, 'numpy')

# Evaluate potentials
v_1 = Vi(phi)
v_2 = Vf(omega)

# Normalize V(ω) to match V(φ)
scal = (np.max(v_1) / np.max(v_2))
print("The scaling factor is : ", scal)
v_2 = v_2 * scal


# Plot potentials
plt.figure(figsize=(10, 6))
plt.plot(phi, v_1, label="V(φ)", color="blue")
plt.plot(omega, v_2, label="V(ω)", color="orange", linestyle="dashed")
plt.plot(L, Mv, label = "Slow Roll Region", color = "green")

# Adding parameter values as text annotations
yj = 5
xd = 5
cd = max(v_2)/2
plt.text(-yj, cd + 7*xd, f'β = {b:.2f}', fontsize=10)
plt.text(-yj, cd + 6*xd, f"$c'$ = {c_:.2f}", fontsize=10)

plt.text(-yj, cd + 2*xd, f'μ = {m:.2f}', fontsize=10)
plt.text(-yj, cd + xd, f'φ₀ = {phi0:.2f}', fontsize=10)
plt.text(-yj, cd, f'σ = {s:.2f}', fontsize=10)
plt.text(-yj, cd - xd, f'ν = {v:.2f}', fontsize=10)
plt.text(-yj, cd - 2*xd, f'c = {c:.2f}', fontsize=10)


plt.title("Potential Comparison: $V(φ)$ vs $V(ω)$")
plt.xlabel("Field Values (φ/ω)")
plt.ylabel("Potential $V(φ/ω)$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Python\Figures\Comparing_Silvio_Barker")
plt.show()