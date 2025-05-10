import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# --- Model parameters (other than gamma) ---
m    = 1.0    # inflaton mass
M    = 1.0    # Planck mass
phi0 = 1.5    # non‑propagating field
a    = 50.0   # alpha
b    = 5.0    # beta
k = 1
# note: D depends on gamma, so compute inside loop

li = 1
ka = 0.01
la = 1
om = 1


# Domain for phi
phi = np.linspace(-3.0, 3.0, 1000)

# GIF settings
gamma_max = 30.0
n_frames  = 60
output_gif = "pvg.gif"

# Temporary directory for frames
frame_dir = "gif_frames"
os.makedirs(frame_dir, exist_ok=True)
frame_files = []

# Sweep gamma
for i, g in enumerate(np.linspace(0, gamma_max, n_frames)):
    D = np.sqrt(-g**2/4 + a*b)
    # define X(φ) and V(φ;γ)
    Xp =  (phi0/(2*b)) * (-g  + D * np.tan(phi /(np.sqrt(k+12)*M)))
    Vp =  - (M**4)/(2) *((ka*phi0**4 +li*phi0**3*Xp +m**2 * Xp**2 + om*phi0*Xp**3 + la*Xp**4) /(b*Xp**2 + g*phi0*Xp + a*phi0**2)**2)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(phi, Vp, lw=2)
    plt.ylim(np.min(Vp)*1.1, np.max(Vp)*20+0.1)
    plt.title(r"$V(\phi)$ at $\gamma={:.2f}$".format(g))
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$V$")
    plt.grid(True)
    plt.tight_layout()

    # Save frame
    fname = os.path.join(frame_dir, f"frame_{i:03d}.png")
    plt.savefig(fname)
    plt.close()
    frame_files.append(fname)

# Build GIF
with imageio.get_writer(output_gif, mode='I', duration=0.1) as writer:
    for fname in frame_files:
        image = imageio.imread(fname)
        writer.append_data(image)

print(f"GIF saved to {output_gif}")
