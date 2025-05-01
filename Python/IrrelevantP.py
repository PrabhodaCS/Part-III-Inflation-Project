"""
/**
 * @author Prabhoda CS
 * @email pcs52@cam.ac.uk
 * @create date 15-03-2025 20:28:34
 * @modify date 15-03-2025 20:28:34
 * @desc [description]
 */
"""

import numba

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from scipy.interpolate import interp1d
from sympy import *

M_p = 1.0
M = np.sqrt(2)*M_p

alpha = -52.4048, 
beta = -0.4628
gamma = -185.1555
k_param = 27.3976
phi0 = -529.7076

mu = 10         #quadratic coeff
la = 0          #Quartic Coeff 
ka = 0          #Quartic coeff of \phi 
q = 0           #Cubic term
li = 0          #linear term
