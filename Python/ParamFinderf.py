"""
/**
 * @author Prabhoda CS
 * @email pcs52@cam.ac.uk
 * @create date 01-05-2025 20:28:34
 * @modify date 01-05-2025 20:28:34
 * @desc [description]
 */
"""

from ast import Num
import numba

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.optimize import OptimizeResult, minimize
from scipy.integrate import solve_ivp

from scipy.interpolate import interp1d
from sympy import *
import time
start_time = time.time()

#JUST CHOICE OF SCALE
M_p = 1.0
M = np.sqrt(2)*M_p

Nend = 70
ns_exp = 0.9626


Nr = np.linspace(0, Nend, 1000)  # Number of e-folds
params = [-2.0, -5.0,0.1, 0.1,0.1, 0.1, 1.0, 0.1, -1.0, 0.1]
#irrparams = [la, ka, q, li] = [1.0, 0.1, -1.0, 0.1]

# Initial conditions
p0 = 100  # Initial field value
p_0 = 0.6  # Initial dphi/dN (velocity)

V0 = [p0, p_0]  # Initial conditions vector


"""#OPTIMAL VALUES
alpha = params[0]
beta = params[1]
gamma =params[2]
k = params[3]
phi0 = params[4]
mu = params[5]         #quadratic coeff
"""



@numba.njit
def fast_gradient(y, x):        #I am finding the gradient of two arrays/vectors (not functions)
    n = len(x)
    grad = np.empty(n)
    for i in range(n):
        if i == 0:
            grad[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        elif i == n - 1:
            grad[i] = (y[i] - y[i-1]) / (x[i] - x[i-1])
        else:
            grad[i] = (y[i+1] - y[i-1]) / ((x[i+1] - x[i-1]))
    return grad

@numba.njit
def K(x,params):
    alpha, beta, gamma,k ,phi0, mu, la, ka, q, li = params

    K_l = M**2 * (-36*beta/(alpha*x + beta*x**2 + gamma*phi0*x) + (phi0**2*(gamma**2 - 4*alpha*beta)*(k - 6))/(2*(alpha*phi0 + beta*x**2 + gamma*phi0*x)**2))
    return K_l

@numba.njit
def V(x,params):
    alpha, beta, gamma,k ,phi0, mu, la, ka, q, li = params
    V_l = M**4 * (li*phi0*x + mu**2*phi0**2*x**2 + q*phi0*x**3 + la*x**4 + ka*phi0**4)/(2*(alpha*phi0 + beta*x**2 + gamma*phi0*x)**2)

    return V_l


#The code for phi as a function of N (I = [phi, dphi]))
@numba.njit
def de(N, I, params):
    p, p_ = I
    h = 1e-6

    dV = (V(p + h, params) - V(p - h, params)) / (2 * h)
    dK = (K(p + h, params) - K(p - h, params)) / (2 * h)

    lv = dV/V(p,params)

    Ka = K(p, params)


    dp2_dN2 = 1/Ka * (-3 * Ka * p_ + 0.5*Ka**2/M_p**2 * p_**3 - 0.5 * dK * p_**2 + (3*M_p**2 - Ka* 0.5 * p_**2) * lv)
    return [p_, dp2_dN2]



@numba.njit
def find_ns(x, params):
    alpha, beta, gamma,k ,phi0, mu, la, ka, q, li = params
    
    V_l = M**4 * (li*phi0*x + mu**2*phi0**2*x**2 + q*phi0*x**3 + la*x**4 + ka*phi0**4)/(2*(alpha*phi0 + beta*x**2 + gamma*phi0*x)**2)
    K_l = M**2 * (-36*beta/(alpha*x + beta*x**2 + gamma*phi0*x) + (phi0**2*(gamma**2 - 4*alpha*beta)*(k - 6))/(2*(alpha*phi0 + beta*x**2 + gamma*phi0*x)**2))

    dK = fast_gradient(K_l, x)
    dV = fast_gradient(V_l, x)
    dV2 = fast_gradient(dV, x)

    epsilon = 0.5 * (M_p**2) * (dV / V_l) ** 2
    eta = (M_p**2) * np.abs(dV2 / V_l)
    lam = (M_p**2) * (dK / K_l)

    ns = 1 + (2*eta - 6*epsilon - np.sqrt(2*epsilon)*lam)/K_l

    return ns


def full_ns(p):
    sol = solve_ivp(lambda N, I: de(N, I, p),
                    (0, Nend),
                    V0,
                    t_eval=Nr,
                    rtol=1e-6, atol=1e-8)
    phi = sol.y[0]
    pi = sol.y[1]
    ns = find_ns(phi, p)
    return ns, sol.t, phi, pi


def ns_cost(p):
    ns_arr, N_vals, phi, pi = full_ns(p)
    idx = np.argmin(np.abs(N_vals - 60))
    
    # Debug plot of ns for this trial
    if np.random.rand() < 0.01:  # only plot 1% of trials to avoid flood
        plt.plot(N_vals, ns_arr, label=f"Trial (ns60={ns_arr[idx]:.3f})")
        plt.axhline(ns_exp, color='r', linestyle='--')
        plt.axvline(60, linestyle=':', color='gray')
        plt.xlabel("N")
        plt.ylabel("ns")
        plt.title("ns(N) during optimization")
        plt.grid(True)
        plt.legend()
        plt.pause(0.1)

    if np.any(ns_arr[N_vals < 55] < 0.96):
        return 1e10
    return (ns_arr[idx] - ns_exp)**2

"""

# Solve the system using solve_ivp
sol = solve_ivp(de, (0, Nend), V0, t_eval=Nr,args = (params,), method='RK45', rtol=1e-6, atol=1e-8)

# Extract the solutions
ps = sol.y[0]  # φ(N)
ps_ = sol.y[1]  # dφ/dN"""

# --- Optimize ---
result = minimize(ns_cost, params, method='Nelder-Mead', options={'xatol': 1e-6, 'disp': True})
opt_params = result.x

# --- Final results ---
ns_final, Nvals, phi, pi = full_ns(opt_params)
idx_final = np.argmin(np.abs(Nvals - 60))

### I HAVE EXTRACTED THE FIELD PHI AND VELOCITY FROM NS

(alpha, beta, gamma, k, phi_0, mu, la, ka, q, li) = opt_params

print("\nOptimal parameters:")
print(f"alpha = {opt_params[0]:.4f}, beta = {opt_params[1]:.4f}, gamma = {opt_params[2]:.4f}")
print(f"k = {opt_params[3]:.4f}, phi0 = {opt_params[4]:.4f}, mu = {opt_params[5]:.4f}")
print(f"\nOptimal irrparams:, la = {opt_params[6]:.4f}, ka = {opt_params[7]:.4f}, q = {opt_params[8]:.4f},li = {opt_params[9]:.4f}")
print(f"\nFinal ns at N=60: {ns_final[idx_final]:.5f}")

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(Nvals, ns_final, label=r'$n_s(N)$')
plt.axhline(ns_exp, color='r', linestyle='--', label=f'Observed $n_s = {ns_exp}$')
plt.axvline(Nvals[idx_final], color='gray', linestyle=':', label='N = 60')
plt.xlabel("e-folds $N$")
plt.ylabel(r"Spectral index $n_s$")
plt.title("Evolution of $n_s$ with $N$")
plt.legend()
plt.grid(True)
plt.tight_layout()
parameters = [
    (r'$\beta$', beta),
    (r'$\mu$', mu),
    (r'$\phi_0$', phi_0),
    (r'$\gamma$', gamma),
    (r'$\alpha$', alpha),
    (r'$k$', k),
    (r"$\lambda$", la),
    (r"$\kappa$", ka),
    (r"$q$", q),
    (r"$li$", li),
]

# Display parameters on the plot
for i, (name, value) in enumerate(parameters):
    plt.text(0.02, 0.95 - i*0.05, f'{name} = {value:.5f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.show()

K_vals = K(phi, params)
V_vals = V(phi, params)
dV = np.gradient(V_vals, phi)
dV2 = np.gradient(dV, phi)
Vprime2 = dV2 / V_vals

epsilon = 0.5 * (M_p**2) * (dV / V_vals) ** 2 / K_vals
r = 16*epsilon

plt.figure(figsize=(6,6))
plt.plot(ns_final, r, '-', lw=2)
plt.axhline(0, color='k', lw=1)
plt.axvline(1, color='gray', lw=1, ls='--')
plt.xlabel(r'$n_s$')
plt.ylabel(r'$r$')
plt.title(r'Tensor-to-scalar ratio $r$ vs.\ spectral index $n_s$')
plt.grid(True)
plt.show()


"""
plt.plot(phi,V_vals, label=r"Potential")
plt.xlabel(r'$\phi$')
plt.ylabel(r'$V(\phi)$')
plt.title(r'V(\phi) vs \phi')
plt.grid(True)
plt.show()

cph = np.linspace(-600, -1, 1000)
Vi = V(cph, opt_params)
plt.plot(cph,Vi, label=r"Extended Potential")
plt.xlabel(r'$\phi$')
plt.ylabel(r'$V(\phi)$')
plt.title(r'V(\phi) vs \phi')
plt.grid(True)
plt.show()
"""
"""
Optimal parameters:
alpha = -1.9272, beta = -4.7452, gamma = 0.0893
k = 0.1244, phi0 = 0.1304, mu = 0.1079

Optimal irrparams:, la = -0.0004, ka = 0.1056, q = -1.2946, li = 0.1169

Final ns at N=60: 0.96260
"""
"""
Optimal parameters:
alpha = -1.9272, beta = -4.7452, gamma = 0.0893
k = 0.1244, phi0 = 0.1304, mu = 0.1079

Optimal irrparams:, la = -0.0004, ka = 0.1056, q = -1.2946,li = 0.1169

Final ns at N=60: 0.96260
"""