import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, ode, odeint
from scipy.optimize import least_squares
plt.rc('text',usetex=True)
from sympy import lambdify, simplify, Symbol, init_printing, log, solve, Function
init_printing()

# Material properties:
# mu = 1. 
# kapp = 20.
# alph = 2.5
mu=Symbol('mu')
kapp=Symbol('kappa')
alph = Symbol('alpha')
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

# Calculate the second derivative f''(R), which will be helpful later on to impose BC and 
# solve the resulting 1st-order system
fpp = simplify(solve(t1.diff(R)-2/R*(t1-t2),fp(R).diff(R) )[0])
fpp = simplify(fpp.subs(f(R).diff(R), fp(R)))
print(fpp)
def ode_sys(r, z, params):
    """
    Let's call the state variable as z = [f, f']
        -- implies z' = [f', f''], where f'' comes from the ode
    """
    mu, kappa, alpha = params
    f_r, fp_r = z
    fpp_r = (-2.0*R**4*kappa*f_r**4*fp_r**4 - 8.0*R**3*kappa*f_r**5*fp_r**3 - 10.0*R**2*kappa*f_r**6*fp_r**2 + 2.0*R**2*mu*(R*fp_r + f_r)**alpha*fp_r**2 - 2.0*R**2*mu*fp_r**2 - 2*R*alpha*mu*(R*fp_r + f_r)**alpha*f_r*fp_r - 4.0*R*kappa*f_r**7*fp_r + 6.0*R*mu*(R*fp_r + f_r)**alpha*f_r*fp_r - 4.0*R*mu*f_r*fp_r - 2.0*R*mu*f_r**(alpha + 1)*fp_r + 2.0*mu*(R*fp_r + f_r)**alpha*f_r**2 - 2.0*mu*f_r**(alpha + 2))/(R**2*(R**2*kappa*f_r**4*fp_r**2 + 2.0*R*kappa*f_r**5*fp_r + alpha*mu*(R*fp_r + f_r)**alpha + kappa*f_r**6 - mu*(R*fp_r + f_r)**alpha + mu)*f_r)
    return [fp_r, fpp_r]

def bc_z(za, zb, params):
    mu, kappa, alpha = params
    bc_inside = 
