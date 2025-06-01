from getdist import plots   # Prabhoda: you should install getdist via cobaya
from getdist.mcsamples import loadMCSamples 
import matplotlib.pyplot as plt

# Prabhoda: please explore the directory of chains and add extra combinations if you like
base_r_BK15_lensing = loadMCSamples('base_r/plikHM_TTTEEE_lowl_lowE_BK15_lensing/base_r_plikHM_TTTEEE_lowl_lowE_BK15_lensing')
base_r_lensing = loadMCSamples('base_r/plikHM_TTTEEE_lowl_lowE_lensing/base_r_plikHM_TTTEEE_lowl_lowE_lensing')
base_r = loadMCSamples('base_r/plikHM_TTTEEE_lowl_lowE/base_r_plikHM_TTTEEE_lowl_lowE')

# Grab the slices.
plotter = plots.get_subplot_plotter()
plotter.plot_2d([base_r,base_r_BK15_lensing], 'ns', 'r', filled=True,colors=['green', ('#F7BAA6', '#E03424')])  #   Prabhoda: make the colors nicer
plotter.add_legend(['Planck','Planck+lensing+BICEP/Keck'])   # Prabhoda: make the legend more informative 

# Prabhoda: add your model here
plt.scatter([0.965], [0.005], marker='*', s=150, color='red')
plt.xlabel(r'$n_s$')
plt.ylabel(r'$r$')
plt.gcf().set_size_inches(8, 8)
plt.savefig('ns_r_plot.pdf', dpi=300)
