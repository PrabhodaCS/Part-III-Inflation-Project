import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.integrate import odeint

m = 1 #mass of inflaton
phi0 = 1.5 #non-propagating field (Dimensions of Mass [phi0]=M)

s = 11 #model param1 (sigma)
v = 50 #model param2 (nu)
c = 0 #integration constant

f = np.sqrt(v-s**2/4)

p_s = symbols('phi')

V = (m**2) * (phi0**4)/2 * (s/2 + f*sinh((p_s-c)/(f*phi0)))**2
print("The potential V(φ) is : ", V)

# Computing first and second derivatives of V
V_ = diff(V, p_s)
print("The first derivative of V(φ) is : ", V_)
V__ = diff(V_, p_s)

# Convert symbolic expressions to numerical functions
V0 = lambdify(p_s, V, 'numpy')
dV = lambdify(p_s, V_, 'numpy')
ddV = lambdify(p_s, V__, 'numpy')


# Define slow-roll parameters
def epsilon(phi):
    return 0.5 * (dV(phi) / V0(phi))**2

def eta(phi):
    return ddV(phi) / V0(phi)

# Define a range of phi values
phi = np.linspace(0, 10, 500)

# Compute values for potential and slow-roll parameters
v1 = V0(phi)
ep = epsilon(phi)
et = abs(eta(phi))

# Isolating parts of graph s.t., slow roll conditions are satisfied
L = []
for i in phi:
    if epsilon(i) < 1 and abs(eta(i)) < 1:
        L.append(i)

L = np.array(L)
Mv = V0(L)

# Defining Hubble Constant given slow roll conditions (dφ/dt)**2 << V(φ)
f = 8*pi/(3*v1)
f = f.astype(float)
H = np.sqrt(f)

# Therefore, EoM becomes (where dp = dφ/dt = V,φ/3H)
def dp(phi, t):
    H_val = np.sqrt((8*np.pi/3)*V0(phi))
    return -dV(phi) / (3 * H_val)

# Calculating number of e-folds
def ef(phi_start, phi_end):
    N = V / V_.simplify()
    Ni = integrate(N, (p_s, phi_end, phi_start))  # Sympy integration
    Nfunc = lambdify([p_s], N, 'numpy')  # Numerical representation
    return Ni.evalf(), Nfunc

p0 = L[0] #Initial condition is beginning of slow roll

t=np.linspace(0,10,100)


ef1, efolds_func = ef(L[-1], L[0])
print(ef1)

pt=odeint(dp,p0,t)
#print(f"φ(t) Evolution: Final φ = {pt[-1][0]:.3f}")  (No need?) ,,check
print(f"Number of e-folds (symbolic integration): {ef1:.3f}")


# Plot potential
plt.figure(figsize=(10, 6))

plt.plot(phi, v1, label="V(φ)")
plt.title("Potential and Slow-Roll Parameters")

plt.plot(L, Mv,'o', label="Slow rolling V(φ)")
plt.title("Potential and Slow-Roll Parameters")

plt.xlabel("φ")
plt.ylabel("V(φ)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Potential and Slow-Roll Parameters")
plt.show()

# isolating values of slow-rolling φ
def csr(phi):
    return epsilon(phi) < 1 and abs(eta(phi)) < 1


### NOT REQUIRED

"""
fp = []
ft = []
for i in range(len(t)):
    if csr(pt[i, 0]):
        fp.append(pt[i,0])
        ft.append(t[i])

fp1 = np.array(fp)
t1 = np.array(ft)
"""

# Plot the phase diagram: φ vs dφ/dt
dphi_dt = np.array([dp(p, 0) for p in L])  # Calculate dφ/dt for slow-rolling φ values
plt.figure(figsize=(10, 6))
plt.plot(L, dphi_dt, 'o', label="Phase Diagram: φ vs dφ/dt")
plt.title("Phase Diagram: φ vs dφ/dt")
plt.xlabel("φ")
plt.ylabel("dφ/dt")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Phase Diagram p vs dpdt")
plt.show()

#Plot the diagram of φ(t)
plt.figure(figsize=(10, 6))
plt.plot(t, pt, label="φ(t)")
plt.title("φ as a function of t")
plt.xlabel("t")
plt.ylabel("φ")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Field as a function of t")
plt.show()