#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import time
from numba import jit

# -------------------------------------------------------------------
# 1) Core model definitions
# -------------------------------------------------------------------
@jit(nopython=False, forceobj=True)
def K_func(phi, alpha, beta, gamma, phi0, M, k):
    A = beta*phi**2 + gamma*phi0*phi + alpha*phi0**2
    term1 = -36 * beta * M**2 / A
    term2 = (M**2 * phi0**2 * (gamma**2 - 4*beta*alpha) * (k - 6)) / (2 * A**2)
    return term1 + term2

@jit(nopython=False, forceobj=True)
def V_func(phi, alpha, beta, gamma, phi0, M, mu, kappa, omega, chi, lambda_):
    A = beta*phi**2 + gamma*phi0*phi + alpha*phi0**2
    num = M**4 * (kappa*phi0**4 + 2*omega*phi0**3*phi
                 + mu**2*phi0**2*phi**2 + 2*chi*phi0*phi**3
                 + lambda_*phi**4)
    return num / (2 * A**2)

class NonCanonicalInflationModel:
    def __init__(self, params):
        self.alpha, self.beta, self.gamma, mu_phi0, self.k, \
        self.kappa, self.omega, self.chi, self.lambda_ = params
        self.M_pl = 1.0
        self.phi0 = 1.0
        self.M = 0.1 * self.M_pl
        self.mu = mu_phi0 / self.phi0

    def K(self, phi):
        return K_func(phi, self.alpha, self.beta, self.gamma,
                      self.phi0, self.M, self.k)

    def V(self, phi):
        return V_func(phi, self.alpha, self.beta, self.gamma,
                      self.phi0, self.M, self.mu,
                      self.kappa, self.omega, self.chi, self.lambda_)

    def dK_dphi(self, phi):
        h = 1e-6
        return (self.K(phi+h) - self.K(phi-h)) / (2*h)

    def dV_dphi(self, phi):
        h = 1e-6
        return (self.V(phi+h) - self.V(phi-h)) / (2*h)

    def epsilon_V(self, phi):
        dV = self.dV_dphi(phi)
        return 0.5 * (self.M_pl**2) * (dV/self.V(phi))**2 / self.K(phi)

    def eta_V(self, phi):
        h = 1e-6
        d2V = (self.dV_dphi(phi+h) - self.dV_dphi(phi-h)) / (2*h)
        return self.M_pl**2 * d2V / (self.K(phi)*self.V(phi))

    def ns(self, phi):
        return 1 - 6*self.epsilon_V(phi) + 2*self.eta_V(phi)

    def r(self, phi):
        return 16*self.epsilon_V(phi)

    def find_phi_end(self, phi_start):
        phis = np.linspace(0.1*phi_start, phi_start, 1000)
        eps = np.array([self.epsilon_V(p) for p in phis])
        return phis[np.argmin(np.abs(eps - 1))]

    def N_efolds(self, phi_start, phi_end):
        phis = np.linspace(phi_end, phi_start, 1000)
        integrand = [self.K(p)*self.V(p)/self.dV_dphi(p) for p in phis]
        return np.trapz(integrand, phis) / self.M_pl**2

    def find_phi_N(self, N, phi_end):
        lo, hi = phi_end, phi_end*5
        while hi - lo > 1e-6:
            mid = 0.5*(lo+hi)
            if self.N_efolds(mid, phi_end) < N:
                lo = mid
            else:
                hi = mid
        return 0.5*(lo+hi)

    def equations_of_motion(self, N, y):
        phi, phidot = y
        Kφ = self.K(phi)
        dK = self.dK_dphi(phi)
        dV = self.dV_dphi(phi)
        H2 = (Kφ*phidot**2/2 + self.V(phi)) / (3*self.M_pl**2)
        H = np.sqrt(H2)
        phiddot = -(3*H*phidot + (dK/(2*Kφ))*phidot**2 + dV/Kφ)
        return [phidot, phiddot]

    def simulate_dynamics(self, phi0, phidot0, Nmax=100, pts=1000):
        t = np.linspace(0, Nmax, pts)
        sol = solve_ivp(self.equations_of_motion,
                        (0, Nmax), [phi0, phidot0],
                        t_eval=t, method='LSODA',
                        rtol=1e-8, atol=1e-8)
        return sol.t, sol.y[0], sol.y[1]

# -------------------------------------------------------------------
# 2) Attractor‐finding, parameter scan & optimization
# -------------------------------------------------------------------
def evaluate_params(params):
    model = NonCanonicalInflationModel(params)
    φ_end = model.find_phi_end(5.0)
    φ60  = model.find_phi_N(60, φ_end)
    ns60 = model.ns(φ60)
    r60  = model.r(φ60)
    score = (ns60 - 0.965)**2/0.01**2 + max(0, (r60 - 0.036))**2/0.01**2
    print(f"ns={ns60:.4f}, r={r60:.4f}, score={score:.2f}")
    return score

# run a quick Nelder‑Mead from a reasonable guess
initial = [ 9.76991997e-01,  1.20185872e+00, -4.84313361e-05 , 1.05437209e+00, 9.76036980e+00,  1.10688054e+00,  1.58990146e-04 , 9.10220288e-04,6.56043077e-01]
res = minimize(evaluate_params, initial, method='Nelder-Mead',
               options={'maxiter':10})
best = res.x
print("Best params:", best)

# -------------------------------------------------------------------
# 3) Detailed analysis & plotting
# -------------------------------------------------------------------
model = NonCanonicalInflationModel(best)
φ_end = model.find_phi_end(5.0)
Ns   = np.linspace(50,70,200)
φs   = [model.find_phi_N(N, φ_end) for N in Ns]
ns_s = [model.ns(φ) for φ in φs]
r_s  = [model.r(φ)  for φ in φs]

fig, axes = plt.subplots(2,2, figsize=(10,8))

# (1) ns vs N
ax = axes[0,0]
ax.plot(Ns, ns_s, 'b', label=r'$n_s$')
ax.axhline(0.965, color='r', linestyle='--', label='Planck $n_s$')
ax.set_xlabel('e‑fold N'); ax.set_ylabel(r'$n_s$')
ax.legend(); ax.grid(True)

# (2) r vs N
ax = axes[0,1]
ax.plot(Ns, r_s, 'g-s', label='$r$')
ax.axhline(0.036, color='orange', linestyle='--', label='$r_{max}$')
ax.set_xlabel('e‑fold N'); ax.set_ylabel('$r$')
ax.legend(); ax.grid(True)

# (3) potential & slow‑roll
phi_plot = np.linspace(φ_end*0.5, φs[0]*1.5, 500)
V_plot   = [model.V(p) for p in phi_plot]
eps_plot = [model.epsilon_V(p) for p in phi_plot]
eta_plot = [model.eta_V(p) for p in phi_plot]

ax = axes[1,0]
ax.plot(phi_plot, V_plot, label='$V(\\phi)$')
ax.plot(phi_plot, eps_plot, label=r'$\epsilon_V$')
ax.plot(phi_plot, eta_plot, label=r'$\eta_V$')
ax.set_xlabel(r'$\phi$')
ax.legend(); ax.grid(True)

# (4) sample background trajectories
ax = axes[1,1]
t, φtraj, φdot = model.simulate_dynamics(φs[-1], 0.0, Nmax=80)
ax.plot(t, φtraj, 'k-', label='$\\phi(N)$')
ax.set_xlabel('e‑fold N'); ax.set_ylabel(r'$\phi$')
ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig('inflation_summary.png')
plt.show()

"""Best params: [ 9.76991997e-01  1.20185872e+00 -4.84313361e-05  1.05437209e+00
  9.76036980e+00  1.10688054e+00  1.58990146e-04  9.10220288e-04
  6.56043077e-01]"""