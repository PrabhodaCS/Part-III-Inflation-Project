import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- Fixed background parameters ---
M_p   = 1.0
M     = np.sqrt(2) * M_p
alpha = 5.0
beta  = 10.0
phi0  = 10.0
k     = 1.0

# Parameter bounds for sampling
bounds = np.array([
    [2*np.sqrt(alpha*beta)+1e-3, 50.0],  # gamma
    [-10.0, 10.0],  # kappa
    [-10.0, 10.0],  # omega
    [-10.0, 10.0],  # mu2
    [-10.0, 10.0],  # chi
    [  1.0, 50.0],  # lam
])

# Field mapping prefactor
prefac = np.sqrt((12 + k) / 2) * M_p
tmin, tmax = -0.98 * (np.pi/2) * prefac, 0.98 * (np.pi/2) * prefac
tilde = np.linspace(tmin, tmax, 1001)

# Inverse map: canonical -> original φ
def phi_of_tilde(t, gamma):
    D = np.sqrt(np.maximum(gamma**2 - 4*alpha*beta, 0))
    return (phi0/(2*beta)) * (D * np.tan(t/prefac) - gamma)

# Canonical potential
def V_can(tilde, params):
    gamma, kappa, omega, mu2, chi, lam = params
    phi = phi_of_tilde(tilde, gamma)
    A   = beta*phi**2 + gamma*phi0*phi + alpha*phi0**2
    num = (kappa*phi0**4
         + 2*omega*phi0**3*phi
         +   mu2*phi0**2*phi**2
         + 2*chi*phi0*phi**3
         +   lam*phi**4)
    return M**4 * num / (2 * A**2)

# Metrics: spike half-width and valley->plateau diff
def metrics(params):
    V = V_can(tilde, params)
    # Spike half-width
    i_peak = np.argmax(V)
    half   = V[i_peak] / 2
    left   = tilde[:i_peak][V[:i_peak] < half]
    right  = tilde[i_peak:][V[i_peak:] < half]
    width = (right[0] - left[-1]) if left.size and right.size else np.nan
    # Valley depth vs plateau height
    valley_region  = tilde > (tmin + 0.05*(tmax - tmin))
    plateau_region = tilde > (tmax - 0.1*(tmax - tmin))
    if valley_region.any() and plateau_region.any():
        valley_min  = V[valley_region].min()
        plateau_avg = V[plateau_region].mean()
        diff = plateau_avg - valley_min
    else:
        diff = np.nan
    return width, diff

# 1) Random sampling
N = 1000
params_samples = np.random.rand(N, 6) * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

# 2) Evaluate metrics
metrics_vals = np.array([metrics(p) for p in params_samples])
widths = metrics_vals[:,0]
diffs  = metrics_vals[:,1]

# 3) Mask valid entries
mask = np.isfinite(widths) & np.isfinite(diffs)
print(f"Valid samples: {mask.sum()} / {N}")

# 4) Compute correlations on valid samples
corr_width = [pearsonr(params_samples[mask,i], widths[mask])[0] for i in range(6)]
corr_diff  = [pearsonr(params_samples[mask,i], diffs[mask])[0]  for i in range(6)]
names = ['gamma','kappa','omega','mu2','chi','lam']

# 5) Plot correlations
x = np.arange(6)
plt.figure(figsize=(8,4))
plt.bar(x-0.15, np.abs(corr_width), 0.3, label='|corr| width')
plt.bar(x+0.15, np.abs(corr_diff), 0.3, label='|corr| diff')
plt.xticks(x, names)
plt.ylabel('Absolute Pearson r')
plt.title('Parameter Sensitivity via Correlation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print top influencers
top_width = np.argsort(np.abs(corr_width))[::-1][:3]
top_diff  = np.argsort(np.abs(corr_diff))[::-1][:3]
print("Top 3 params for spike width:", [names[i] for i in top_width])
print("Top 3 params for valley->plateau diff:", [names[i] for i in top_diff])
