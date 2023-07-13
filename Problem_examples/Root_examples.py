import numpy as np
import Solve as s

def NumMethod1(N,zm):
    """ Find the eigenenergys of a quantum well with limited Vo using the
    Newton's method (evaluate df/dx using finite differences)"""

    F = lambda z: np.tan(z) - np.sqrt( (zm/z)**2 - 1 )

    Zvec = np.arange(1e-4,zm,np.pi/2)
    a_list = Zvec[0:2*N:2]
    b_list = Zvec[1:2*N:2]-0.01

    Z = []
    for a,b in zip(a_list,b_list):
        z = s.ZerosNewton(F,a,b,1e-4)
        Z.append(z)
    
    return Z