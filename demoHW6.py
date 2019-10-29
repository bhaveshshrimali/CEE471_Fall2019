import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
from sympy import lambdify, simplify, Symbol, init_printing, log, solve, Function
init_printing()

# Material properties:
mu = 1. 
kapp = 20.
alph = 2.

lam1, lam2, lam3 = Symbol('lambda_1'), Symbol('lambda_2'), Symbol('lambda_3')
R = Symbol('R')
f = Function('f', real=True)
Jdet = lam1*lam2*lam3

# define the stored energy function in terms of the eigen-stretches
W = mu/alph*(lam1**alph + lam2**alph + lam3**alph ) - mu*log(Jdet) + kapp/2*(Jdet - 1)**2

# compute the symbolic derivatives: 
    # useful in calculation of the eigen stresses
dW_dl1 = lambdify((lam1, lam2, lam3), simplify(W.diff(lam1)), 'numpy')
dW_dl2 = lambdify((lam1, lam2, lam3), simplify(W.diff(lam2)), 'numpy')
dW_dl3 = lambdify((lam1, lam2, lam3), simplify(W.diff(lam3)), 'numpy')

# t1 = dW_dl1(R*fp+f, f, f)
t1 = simplify(W.diff(lam1)).subs([(lam1,R*f(R).diff(R)+f(R)), (lam2, f(R)), (lam3, f(R))])
t2 = simplify(W.diff(lam2)).subs([(lam1,R*f(R).diff(R)+f(R)), (lam2, f(R)), (lam3, f(R))])
t3 = simplify(W.diff(lam3)).subs([(lam1,R*f(R).diff(R)+f(R)), (lam2, f(R)), (lam3, f(R))])

# Calculate the second derivative f''(R), which will be helpful later on to impose BC and 
# solve the resulting 1st-order system
fpp = simplify(solve(t1.diff(R)-2/R*(t1-t2),f(R).diff(R,2)))
print(fpp)
# t2 = dW_dl2(R*fp+f, f, f)
# t3 = dW_dl3(R*fp+f, f, f)

def ode_sys(r, z):
    """
    Let's call the state variable as z = [f, f']
        -- implies z' = [f', f''], where f'' comes from the ode
    """
    f, fp = z
    



