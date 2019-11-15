import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root, fsolve
import os
plt.rc('text',usetex=True)
plt.rcParams['font.weight'] = 700
plt.rcParams['xtick.top']='True'
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.right']='True'
plt.rcParams['ytick.direction']='in'
plt.rcParams['ytick.labelsize']=22
plt.rcParams['xtick.labelsize']=22
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['ytick.minor.visible']=True
plt.rcParams['xtick.major.size']=6
plt.rcParams['xtick.minor.size']=3
plt.rcParams['ytick.major.size']=6
plt.rcParams['ytick.minor.size']=3
plt.rcParams['lines.markersize']=np.sqrt(36) 

def gen():
    for t in tfinal:
        yield K/2.*(t-np.sin(t)), -K/2.*(1.-np.cos(t)), ystraight_line(K/2.*(t-np.sin(t))), t

def func_ani(data):
    x,y,y_st, t = data
    time_text.set_text(r'$t = {:.3f}$'.format(t))
    brach.set_data(x,y)
    stline.set_data(x,y_st)

def calc_time(params):
    K, t = params
    # the corresponding `x` and `y`
    return np.array([K/2*(t - np.sin(t)) - 10, K/2*(1-np.cos(t)) - 2])

def jac_calc_time(params):
    K, t = params
    # calculates the jacobian of the above function
    return np.array([[1./2*(np.sin(t)), K/2.*(1-np.cos(t))],[1./2*(1-np.cos(t)), K/2.*(np.sin(t))]])


sol=root(calc_time,[5,2],method='hybr',jac=jac_calc_time,tol=1.e-15,options={'maxfev':1000000})
K,T = sol.x

tfinal = np.linspace(0.,T,100)
x = K/2*(tfinal - np.sin(tfinal))
y = -K/2.*(1. - np.cos(tfinal))
mslope = (y[-1]-y[0])/(x[-1]-x[0])
gacc = 9.81
Tstraight_line = np.sqrt(2*(1+mslope**2)/-gacc/mslope * x[-1])
print(Tstraight_line,T)
ystraight_line = lambda xf: (y[-1]-y[0])/(x[-1]-x[0])*(xf - x[-1]) + y[-1]

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.plot(x,y,'-r',label='Brachistochrone')
ax.plot(x,ystraight_line(x),'-b',label='Straight-Line')
ax.legend(loc='upper right',fontsize=22,ncol=2,fancybox=False,facecolor='wheat',edgecolor='k',shadow=True)
ax.set_xlabel(r'$x$',fontsize=22)
ax.set_ylabel(r'$y(x)$',fontsize=22)
ax.grid(which='major')
time_text = ax.text(0.45,0.5,'',transform=ax.transAxes, fontsize=24, bbox = dict(boxstyle='round',facecolor='wheat',edgecolor='k'))
brach, = ax.plot([],[],'ro',ms=12)
stline, = ax.plot([],[],'bo',ms=12)


ani = animation.FuncAnimation(fig, func_ani, gen, blit=False, interval=100,repeat=False)
# plt.show()
ani.save(os.path.join(os.getcwd(),'Brachistochrone.mp4'),writer='ffmpeg',fps=30)
plt.show()

# To save the animation, use e.g.
#
# ani.save("movie.mp4")