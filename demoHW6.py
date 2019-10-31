import numpy as np
import math 
import numpy.linalg as nla
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, ode, odeint, solve_ivp
from scipy.optimize import least_squares
plt.rc('text',usetex=True)
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
R = Symbol('R')
f = Function('f', real=True)
fp = Function('fp', real=True)
Jdet = lam1*lam2*lam3

# define the stored energy function in terms of the eigen-stretches
W = mu/alph*(lam1**alph + lam2**alph + lam3**alph ) - mu*log(Jdet) + kapp/2*(Jdet - 1)**2

# compute the symbolic derivatives: 
    # useful in calculation of the eigen stresses
dW_dl1 = lambdify((lam1, lam2, lam3), simplify(W.diff(lam1)), 'numpy')
dW_dl2 = lambdify((lam1, lam2, lam3), simplify(W.diff(lam2)), 'numpy')
dW_dl3 = lambdify((lam1, lam2, lam3), simplify(W.diff(lam3)), 'numpy')

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
fpp = simplify(solve(t1.diff(R)-2/R*(t1-t2),fp(R).diff(R))[0])
fpp = simplify(fpp.subs(f(R).diff(R), fp(R)))
print(fpp)   # simplest way is to copy-paste this in the function below

def ode_sys(R, z, params):
    """
    Returns a system of two first order ODEs from the given second order
    ODE. Let's call the state variable as z = [f, f']
        -- implies z' = [f', f''], where f'' comes from the ode
        -- a simple procedure is to just copy-paste the console output form above;
           a more general way, though, would be to create a callable function using `lambdify`
    """
    mu, kappa, alpha, A, B = params
    f_r, fp_r = z
    fpp_r = (-2.0*R**4*kappa*f_r**4*fp_r**4 - 8.0*R**3*kappa*f_r**5*fp_r**3 - 10.0*R**2*kappa*f_r**6*fp_r**2 + 2.0*R**2*mu*(R*fp_r + f_r)**alpha*fp_r**2 - 2.0*R**2*mu*fp_r**2 - 2*R*alpha*mu*(R*fp_r + f_r)**alpha*f_r*fp_r - 4.0*R*kappa*f_r**7*fp_r + 6.0*R*mu*(R*fp_r + f_r)**alpha*f_r*fp_r - 4.0*R*mu*f_r*fp_r - 2.0*R*mu*f_r**(alpha + 1)*fp_r + 2.0*mu*(R*fp_r + f_r)**alpha*f_r**2 - 2.0*mu*f_r**(alpha + 2))/(R**2*(R**2*kappa*f_r**4*fp_r**2 + 2.0*R*kappa*f_r**5*fp_r + alpha*mu*(R*fp_r + f_r)**alpha + kappa*f_r**6 - mu*(R*fp_r + f_r)**alpha + mu)*f_r)
    return [fp_r, fpp_r]

def integrate_ode_sys(z_at_B, integrator, params, step=1.e-3, silent=True):
    """
    Calls an ODE integrator to integrate the initial value problem over the 
    shell thickness; starting from the outer surface
    """
    mu, kapp, alph, A, B = params
    initial_condition = z_at_B
    integrator.set_initial_value(initial_condition, t=B)
    integrator.set_f_params(params)
    dt =step
    xs, zs = [], []
    while integrator.successful() and integrator.t >= A:
        integrator.integrate(integrator.t - dt)
        xs.append(integrator.t)
        zs.append([integrator.y[0], integrator.y[1]])
        if not silent:
            print("Current x and z values: ", integrator.t, integrator.y)
    return xs, zs

def solve_bvp_balloon(params, pvals, step=1.e-3, silent=True):
    """
    Solve the boundary value problem by minimizing the loss function for 
    the mixed (Robin)-like BC at the inner boundary 
    returns the correct value of z = [f, f'] at R = B
    """
    integrator = ode(ode_sys).set_integrator('vode', rtol=1.e-8, method='bdf')
    mu, kapp, alph, A, B = params
    # xs, zs = [], []
    def residual_at_A(z_at_B):
        xz, zs = integrate_ode_sys(z_at_B, integrator, params)
        f_A, fp_A = np.array(zs)[-1]
        bc_inside = pvals + t1_func(A*fp_A, f_A, f_A)
        return bc_inside
    
    z_at_B_guess = np.array([1+pvals**(1./4), 1./(1. + pvals**(1./4))])
    soln = least_squares(residual_at_A, z_at_B_guess, loss='soft_l1')
    return soln.x 

pressure_inside = np.linspace(0,0.09,101)
A, B = 0.95, 1.
parameters = [mu, kapp, alph, A, B]
integrator = ode(ode_sys).set_integrator('vode', rtol=1.e-8, method='bdf')
Xvals = []
fvals = []
fpvals = []
for idx_p, p in enumerate(pressure_inside):   
    z_at_B_soln = solve_bvp_balloon(parameters, p)
    xs, zs = integrate_ode_sys(z_at_B_soln, integrator, parameters)
    zs = np.array(zs)
    Xvals.append(xs)
    fvals.append(zs[:,0])
    fpvals.append(zs[:,1])
    
    # fvals.append(zs[0])
    

# def ic_z(za):
#     mu, kappa, alpha = 1.e6, 20.e6, 2.5
#     A = 0.95
#     B = 1.
#     f_A, fp_A = za

# #     include the initial condition in terms of the cauchy pressure
#     ic_inside = p + t1_func(A*fp_A+f_A, f_A, f_A)
#     # ic_prime_inside = fp_A - 1./(1+p**(1./4))
#     return [ic_inside]

# def solve_bvpj(p):
#     soln = solve_ivp(ode_sys,[rVals[0],rVals[-1]],ic_z,t_eval=)
#     return 

# pVals = np.linspace(1.e-1,0.09*10**6,101)
# rVals= np.linspace(0.95,1,501)
# for idxp, p in enumerate(pVals):
#     fp_A

