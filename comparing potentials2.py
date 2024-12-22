import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sinh, lambdify
from sympy import simplify

# Free Parameters
Mp = 1
b = 1.5
c_ = 2



"""# above from Salvio, below from Barker
phi0 = 1.5
#dependant parameters
c = - np.sqrt(3/2) * Mp * np.arctanh(4*b/np.sqrt(Mp**4 + 16 * b**2))    #const of integration
m = np.sqrt(1 / (2 * c_)) / phi0                          #mu
#m = 1/phi0 * np.sqrt(1/(2*c_))
v = (3/2 * Mp**2/phi0**2 + 24 * b**2 * phi0**2 / Mp**2)
s = -4*np.sqrt(6)*b*phi0/Mp

f = np.sqrt(3/2)*Mp/phi0
"""

#Derived Parameters [Salvio --> Barker]
m = Mp**2 / (24*np.sqrt(2*c_))
phi0 = np.sqrt(24) / Mp 
v = Mp**4 / 16 + b**2
s = -2*b
f = Mp**2 / 4       #derived from v and s as f = np.sqrt (v - s**2 / 4)
c = -Mp*np.sqrt(3/2)*np.arctanh(4 * b / np.sqrt(Mp ** 4 + 16 * b ** 2))

# Define potentials
p_s = symbols('phi')
Vp = (m ** 2) * (phi0 ** 4) / 2 * (s / 2 + f * sinh((p_s - c) / (f * phi0))) ** 2
print("The potential V(φ) is:", Vp)

w_s = symbols('omega')
X = np.sqrt(2 / 3) * w_s / Mp + np.arctanh(4 * b / np.sqrt(Mp ** 4 + 16 * b ** 2))
Vp1 = 1 / (4 * c_) * (Mp ** 2 / 4 * sinh(X) - b) ** 2
print("The potential V(ω) is:", Vp1)



# Simplify symbolic expressions for comparison
are_equivalent = simplify(Vp / Vp1)
print("Are V(φ) and V(ω) equivalent?", are_equivalent)


# Convert symbolic expressions to numerical functions
Vi = lambdify(p_s, Vp, 'numpy')
Vf = lambdify(w_s, Vp1, 'numpy')

# Define a range of φ and ω values
omega = np.linspace(0.5, 5, 500)
phi = omega + f*phi0*np.arctanh(4*b/np.sqrt(Mp**4 + 16*b**2)) + c

# Evaluate potentials
v1 = Vi(phi)
v2 = Vf(omega)

# Normalize V(ω) to match V(φ)
scal = (np.max(v1) / np.max(v2))
print("The scaling factor is : ", scal)
v2 = v2 * (np.max(v1) / np.max(v2))


# Plot potentials
plt.figure(figsize=(10, 6))
plt.plot(phi, v1, label="V(φ)", color="blue")
plt.plot(omega, v2, label="V(ω)", color="orange", linestyle="dashed")
plt.title("Potential Comparison: $V(φ)$ vs $V(ω)$")
plt.xlabel("Field Values (φ/ω)")
plt.ylabel("Potential $V(φ/ω)$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()