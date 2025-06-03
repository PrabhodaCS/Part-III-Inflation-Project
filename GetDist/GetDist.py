from getdist import plots
from getdist.mcsamples import loadMCSamples
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Point each loadMCSamples(...) at the folder where your chains actually live.
# In your case, everything is under "...GetDist\Code\base_r\…", not "...GetDist\PlanckData\…"
# ----------------------------------------------------------------

# (1) “Planck only” (no BK15, no lensing) lives in:
planck_only = loadMCSamples(
    r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\GetDist\Code\base_r\plikHM_TTTEEE_lowl_lowE\base_r_plikHM_TTTEEE_lowl_lowE"
)

# (2) “Planck + BK15” (but no lensing) lives in:
planck_plus_BK15 = loadMCSamples(
    r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\GetDist\Code\base_r\plikHM_TTTEEE_lowl_lowE_BK15\base_r_plikHM_TTTEEE_lowl_lowE_BK15"
)

# (3) “Planck + lensing” lives in:
planck_plus_lensing = loadMCSamples(
    r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\GetDist\Code\base_r\plikHM_TTTEEE_lowl_lowE_lensing\base_r_plikHM_TTTEEE_lowl_lowE_lensing"
)

# (4) “Planck + BK15 + lensing” lives in:
planck_BK15_lensing = loadMCSamples(
    r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\GetDist\Code\base_r\plikHM_TTTEEE_lowl_lowE_BK15_lensing\base_r_plikHM_TTTEEE_lowl_lowE_BK15_lensing"
)

# Now you can over‐plot n_s vs r for whichever combinations you like. For example:
plotter = plots.get_subplot_plotter()
plotter.plot_2d(
    [planck_only, planck_BK15_lensing],
    "ns", "r",
    filled=True,
    colors=["green", ("#F7BAA6", "#E03424")]
)
plotter.add_legend([
    "Planck only",
    "Planck + lensing + BICEP/Keck"
])

# …and if you want to mark a particular inflation model:
plt.scatter([0.965], [0.005], marker="*", s=150, color="red")
plt.xlabel(r"$n_s$")
plt.ylabel(r"$r$")
plt.gcf().set_size_inches(8, 8)
plt.tight_layout()
plt.savefig("ns_r_plot.pdf", dpi=300)
plt.show()
