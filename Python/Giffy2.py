import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from sympy import *

# --- Your fixed model parameters ---
m    = 1.0
M    = 7
phi0 = 20
a    = 50.0
b    = 10.0
c    = 0    # your 'd' constant from above
# Domain for phi
phi = np.linspace(0, 100, 1000)

# GIF sweep settings
g_min     = 0.0
g_max     = 30.0    # adjust to whatever max you want
n_frames  = 40
gif_file  = "potential_vs_g.gif"

# Prepare sympy for your V(φ;g)
p_s = symbols('phi')
x   = p_s/(np.sqrt(6)*M) + c
# Xp(g,φ):
Xp_expr = (1/(4*b)) * exp(-x)*(exp(2*x) - 2*exp(x)*symbols('g')*phi0 - phi0**2 * sqrt(-symbols('g')**2/4 + a*b)**2)
# full Vp(g,φ):
V_expr  = (m**2 * phi0**2 * Xp_expr**2) / (2*(b*Xp_expr**2 + symbols('g')*phi0*Xp_expr + a*phi0**2)**2)

# turn it into a numpy function of (φ,g)
V_func = lambdify((p_s, symbols('g')), V_expr, 'numpy')

# Frame directory
frame_dir = "gif_frames_g"
os.makedirs(frame_dir, exist_ok=True)
frames = []

for i, g_val in enumerate(np.linspace(g_min, g_max, n_frames)):
    V_vals = V_func(phi, g_val)

    plt.figure(figsize=(6,4))
    plt.plot(phi, V_vals, lw=2)
    plt.ylim(np.min(V_vals)*1.1, np.max(V_vals)*1.1)
    plt.title(r"$V(\phi)$ at $g={:.2f}$".format(g_val))
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$V$")
    plt.grid(True)
    plt.tight_layout()

    fname = os.path.join(frame_dir, f"g_frame_{i:03d}.png")
    plt.savefig(fname)
    plt.close()
    frames.append(fname)

# Build the GIF
with imageio.get_writer(gif_file, mode='I', duration=0.1) as writer:
    for fname in frames:
        img = imageio.imread(fname)
        writer.append_data(img)

print(f"Animated GIF written to {gif_file}")
