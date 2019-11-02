import numpy as np
import math 
from copy import deepcopy
import numpy.linalg as nla
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, ode, odeint, solve_ivp
from scipy.optimize import least_squares
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','serif':['lmodern']})
plt.rc('text',usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['font.weight'] = 1000
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
from sympy import lambdify, simplify, Symbol, init_printing, log, solve, Function
init_printing()

# Material properties:
mu = 1.e6
kapp = 20.e6
alph = 2.5
# mu=Symbol('mu')
# kapp=Symbol('kappa')
# alph = Symbol('alpha')
lam1, lam2, lam3 = Symbol('lambda_1'), Symbol('lambda_2'), Symbol('lambda_3')
lam = Symbol('lambda')
R = Symbol('R')
f = Function('f', real=True)
fp = Function('fp', real=True)
Jdet = lam1*lam2*lam3

# define the stored energy function in terms of the eigen-stretches
W = mu/alph*(lam1**alph + lam2**alph + lam3**alph ) - mu*log(Jdet) + kapp/2.*(Jdet - 1)**2
W_asym = simplify(W.subs([(lam1,lam**(-2)), (lam2, lam), (lam3, lam)]))

# compute the symbolic derivatives: 
    # useful in calculation of the eigen stresses
dW_dl1 = lambdify((lam1, lam2, lam3), simplify(W.diff(lam1)), 'numpy')
dW_dl2 = lambdify((lam1, lam2, lam3), simplify(W.diff(lam2)), 'numpy')
dW_dl3 = lambdify((lam1, lam2, lam3), simplify(W.diff(lam3)), 'numpy')

# .... and for the asymptotic solution
dW_asymp_dl = lambdify(lam, simplify(W_asym.diff(lam)), 'numpy')

# t1 = dW_dl1(R*fp+f, f, f)
# t2 = dW_dl2(R*fp+f, f, f)
# t3 = dW_dl3(R*fp+f, f, f)
t1 = simplify(1./(lam2*lam3)*W.diff(lam1)).subs([(lam1,R*fp(R)+f(R)), (lam2, f(R)), (lam3, f(R))])
t2 = simplify(1./(lam1*lam3)*W.diff(lam2)).subs([(lam1,R*fp(R)+f(R)), (lam2, f(R)), (lam3, f(R))])
t3 = simplify(1./(lam1*lam2)*W.diff(lam3)).subs([(lam1,R*fp(R)+f(R)), (lam2, f(R)), (lam3, f(R))])

# create a lambda function in order to aid bc/ic imposition
t1_func = lambdify((lam1, lam2, lam3), simplify(1./(lam2*lam3)*W.diff(lam1)),'numpy')

# Calculate the second derivative f''(R), which will be helpful later on to impose BC and 
# solve the resulting 1st-order system
fpp = simplify(solve(t1.diff(R)+2/R*(t1-t2),fp(R).diff(R))[0])
fpp = simplify(fpp.subs(f(R).diff(R), fp(R)))
# print(fpp)   # simplest way is to copy-paste this in the function below

def ode_sys(r, z, params):
    """
    Returns a system of two first order ODEs from the given second order
    ODE. Let's call the state variable as z = [f, f']
        -- implies z' = [f', f''], where f'' comes from the ode
        -- a simple procedure is to just copy-paste the console output form above;
           a more general way, though, would be to create a callable function using `lambdify`
    """
    mu, kappa, alpha, A, B = params
    f_r, fp_r = z
    fpp_r = (-2.0*r**4*kappa*f_r**4*fp_r**4 - 8.0*r**3*kappa*f_r**5*fp_r**3 - 10.0*r**2*kappa*f_r**6*fp_r**2 + 2.0*r**2*mu*(r*fp_r + f_r)**alpha*fp_r**2 - 2.0*r**2*mu*fp_r**2 - 2*r*alpha*mu*(r*fp_r + f_r)**alpha*f_r*fp_r - 4.0*r*kappa*f_r**7*fp_r + 2.0*r*mu*(r*fp_r + f_r)**alpha*f_r*fp_r - 4.0*r*mu*f_r*fp_r + 2.0*r*mu*f_r**(alpha + 1)*fp_r - 2.0*mu*(r*fp_r + f_r)**alpha*f_r**2 + 2.0*mu*f_r**(alpha + 2))/(r**2*(r**2*kappa*f_r**4*fp_r**2 + 2.0*r*kappa*f_r**5*fp_r + alpha*mu*(r*fp_r + f_r)**alpha + kappa*f_r**6 - mu*(r*fp_r + f_r)**alpha + mu)*f_r)
    return [fp_r, fpp_r]

def integrate_ode_sys(z_at_A, integrator, params, step=1.e-3, silent=True):
    """
    Calls an ODE integrator to integrate the initial value problem over the 
    shell thickness; starting from the outer surface
    """
    mu, kapp, alph, A, B = params
    initial_condition = z_at_A
    integrator.set_initial_value(initial_condition, t=A)
    integrator.set_f_params(params)
    dt =step
    xs, zs = [], []
    while integrator.successful() and integrator.t < B:
        integrator.integrate(integrator.t + dt)
        xs.append(integrator.t)
        zs.append([integrator.y[0], integrator.y[1]])
        if not silent:
            print("Current x and z values: ", integrator.t, integrator.y)
    return xs, zs

def solve_bvp_balloon(f_at_A, params, pvals, step=1.e-3, silent=True):
    """
    Solve the boundary value problem by minimizing the loss function for 
    the mixed (Robin)-like BC at the inner boundary 
    returns the correct value of z = [f, f'] at R = B
    """
    integrator = ode(ode_sys).set_integrator('vode', rtol=1.e-12, method='bdf', order = 5)
    mu, kapp, alph, A, B = params
    # xs, zs = [], []
    def residual_at_A(z_at_A):
        xs, zs = integrate_ode_sys(z_at_A, integrator, params)
        f_A, fp_A = np.array(zs)[0]
        f_B, fp_B = np.array(zs)[-1]
        bc_inside = pvals + t1_func(A*fp_A, f_A, f_A)
        bc_outside = t1_func(B*fp_B, f_B, f_B)
        return [bc_inside, bc_outside]
    
    fp_at_A = lambda fpEval: pvals + t1_func(A*fpEval, f_at_A, f_at_A)
    fp_at_A_guess = least_squares(fp_at_A, 1./(1+pvals**(0.25)), method='trf', loss='soft_l1', max_nfev=3000)
    z_at_A_guess = np.array([f_at_A, fp_at_A_guess.x])
    # z_at_B_guess = np.array([1., 0.01])
    soln = least_squares(residual_at_A, z_at_A_guess, method='trf', loss='soft_l1', max_nfev=3000)
    return soln.x 

pressure_inside = np.linspace(0,0.09*10**6,501)
A, B = 0.95, 1.
parameters = [mu, kapp, alph, A, B]
integrator = ode(ode_sys).set_integrator('vode', rtol=1.e-12, method='bdf', order = 5)
Xvals = []
fvals = []
fpvals = []
fA = 1.
for idx_p, p in enumerate(pressure_inside):   
    z_at_A_soln = solve_bvp_balloon(deepcopy(fA),parameters, p)
    xs, zs = integrate_ode_sys(z_at_A_soln, integrator, parameters)
    zs = np.array(zs)
    if idx_p == 0:
        Xvals.append(xs)
    fA = zs[0,0]
    fvals.append(zs[:,0])
    fpvals.append(zs[:,1])

fvals = np.array(fvals)
fpvals = np.array(fpvals)

lam_asymp = fvals[:,0]
pasymp = (B-A)/A*lam_asymp**(-2)*dW_asymp_dl(lam_asymp)

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.plot(pasymp[1:], lam_asymp[1:], label='Asymptotic')
ax.plot(pressure_inside, lam_asymp, label = 'Numerical')
ax.legend(loc=0,fontsize=26)
ax.grid(which='major',linestyle='--',color='k')
fig.tight_layout()