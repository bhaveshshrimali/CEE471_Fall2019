import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
from sympy import lambdify, simplify, Symbol, init_printing, log

class dW_dLi:
    def __init__(self, mu, alph, kapp):
        lam1, lam2, lam3 = Symbol('lambda_1'), Symbol('lambda_2'), Symbol('lambda_3')
        Jdet = lam1*lam2*lam3
        W = mu/alph*(lam1**alph + lam2**alph + lam3**alph ) - mu*log(Jdet) + kapp/2*(Jdet - 1)**2
        self.dW_dl1 = simplify(W.diff(lam1))
        self.dW_dl2 = simplify(W.diff(lam2))
        self.dW_dl3 = simplify(W.diff(lam3))

def W_free(l1, l2, l3, alph, mu, kapp):
    """
    Define the stored-energy function for the hyperelastic solid
        -- allows to calculate the derivatives of W wrt the arguments
    """
    
    dW_dF = [lambdify((lam1,lam2, lam3),dW_dl1,'numpy'),
             lambdify((lam1,lam2, lam3),dW_dl2,'numpy'),
             lambdify((lam1,lam2, lam3),dW_dl3,'numpy')]

    return dW_dF[0](l1,l2,l3),dW_dF[1](l1,l2,l3), dW_dF[2](l1,l2,l3) 




