from getdist import plots   # Prabhoda: you should install getdist via cobaya
from getdist.mcsamples import loadMCSamples 
import matplotlib.pyplot as plt

# Prabhoda: you don't have this but I'll give it to you.
samples = loadMCSamples('/home/barker/Downloads/base_r/plikHM_TTTEEE_lowl_lowE_BK15_lensing/base_r_plikHM_TTTEEE_lowl_lowE_BK15_lensing')

# Grab the slices.
plotter = plots.get_subplot_plotter()
plotter.plot_2d(samples, 'ns', 'r', filled=True, legend_labels=['Planck 2018'])

# Prabhoda's fancy inflation model on top.
plt.scatter([0.965], [0.005], marker='*', s=150, color='red', label='Sarjapur inflation')
plt.legend()
plt.xlabel(r'$n_s$')
plt.ylabel(r'$r$')
plt.tight_layout()
plt.gcf().set_size_inches(10, 8)
plt.savefig('ns_r_plot.png', dpi=300)
