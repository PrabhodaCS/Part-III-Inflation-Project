"""
/**
 * @author Prabhoda CS
 * @email pcs52@cam.ac.uk
 * @create date 20-11-2024 01:25:37
 * @modify date 20-11-2024 01:25:37
 * @desc [description]
 */
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
 
 
def DE(x,t,k1,k2):
    dxdt=[x[1],-k1*x[1]-k2*np.sin(x[0])]
    return dxdt


 
# set the initial conditions
x0=[0,1]
 
# define the discretization points
t=np.linspace(0,100,300)
 
#define the constants 
k1=0.1
k2=0.5
 
sde=odeint(DE,x0,t, args=(k1,k2))


plt.plot(t, sde[:, 0], 'b', label='x1')
plt.plot(t, sde[:, 1], 'g', label='x2')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('x1(t), x2(t)')
plt.grid()
plt.show()
