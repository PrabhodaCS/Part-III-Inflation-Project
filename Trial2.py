import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, sinh

m = 1 #mass of inflaton
phi0 = 5 #non-propagating field

s = -2 #model param1 (sigma)
v = 2 #model param2 (nu)
c = 0 #integration constant

f = np.sqrt(v-s**2/4)

p_s = symbols('phi')

V = (m**2) * (phi0**4)/2 * (s/2 + f*sinh((p_s-c)/(f*phi0)))**2

# Computing first and second derivatives of V
V_ = diff(V, p_s)
V__ = diff(V_, p_s)

# Convert symbolic expressions to numerical functions
V = lambdify(p_s, V, 'numpy')
dV = lambdify(p_s, V_, 'numpy')
ddV = lambdify(p_s, V__, 'numpy')

# Define slow-roll parameters
def epsilon(phi):
    return 0.5 * (dV(phi) / V(phi))**2

def eta(phi):
    return ddV(phi) / V(phi)

# Define a range of phi values
phi = np.linspace(1, 8, 500)

# Compute values for potential and slow-roll parameters
v1 = V(phi)
ep = epsilon(phi)
et = abs(eta(phi))

# Isolating parts of graph s.t., slow roll conditions are satisfied
L = []
for i in phi:
    if epsilon(i) < 1 and abs(eta(i)) < 1:
        L.append(i)

L = np.array(L)
Mv = V(L)

plt.figure(figsize=(10, 6))

# Plot potential
plt.plot(phi, v1, label="V(φ)")
plt.title("Potential and Slow-Roll Parameters")
plt.xlabel("φ")
plt.ylabel("V(φ)")
plt.legend()
plt.grid()

plt.plot(L, Mv,'o', label="Slow rolling V(φ)")
plt.title("Potential and Slow-Roll Parameters")
plt.xlabel("φ")
plt.ylabel("V(φ)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()