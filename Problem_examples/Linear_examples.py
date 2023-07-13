import numpy as np
import Solve as ge

# Example of initial problem to SOR_method.
def Problem_init(N: int):
    M = np.diag(np.full(N,2))+np.diag(np.full(N-1,-1),1)+np.diag(np.full(N-1,-1),-1)
    f = lambda x: np.sin(np.pi*x)
    u = lambda x: -np.sin(np.pi*x)/(np.pi**2)
    h = 1/(N+1)

    omega = 2/(1+np.sin(np.pi/(N+1)))
    f_vec = -(h**2)*f(np.linspace(h,1-h,N))
    u_vec = u(np.linspace(h,1-h,N))
    U0 = np.zeros(N)

    return M,omega,f_vec,u_vec,U0
