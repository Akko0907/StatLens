import numpy as np
import matplotlib.pyplot as plt
from Solve import EDO

def Osc_harm():
    
    # Parameters, step and range of interval
    omega = 4.
    dt = 0.05
    T = 4.

    # System to solve
    F_list = [lambda v,x: -omega**2*x, lambda v,x: v]
    
    # Initial conditions
    v0 = 0.
    x0 = 0.2
    init = np.array([v0,x0])

    return omega, F_list, h, T, init

def Epidem():
    
    # Parameters, step and range of interval
    beta = 1/2
    gama = 1/3
    h = 1
    T = 150

    # System to solve
    F_list = [lambda s,j,r: -beta*s*j, lambda s,j,r: beta*s*j-gama*j, lambda s,j,r: gama*j ]

    # Initial conditions
    s = 1
    j = 1.27e-6
    r = 0
    init = np.array([s,j,r])

    return F_list,h,T,init

def Terminal_V():
    
    # Parameters, step and range of interval
    B = 0.01
    g = 9.8
    h = 0.4
    T = 20

    # System to solve
    F_list = [lambda v,x: g-B*v**2/2, lambda v,x: v]

    # Initial conditions
    x0 = 100.
    v0 = 0.
    init = np.array([v0,x0])

    return F_list,h,T,init

def Prob1():

    # Parameters, step and range of interval
    h = 0.05
    T = 1

    # System to solve
    F_list = [lambda y,t: -t*y, lambda y,t: 1]

    # Initial conditions
    y0 = 1.
    t0 = 0.
    init = np.array([y0,t0])

    return F_list,h,T,init


omega,F_list,h,T,init = Osc_harm()
oscilador_harmonico = EDO(F_list,2,0.1)

t,vs,xs = oscilador_harmonico(init,'special')
t,vb,xb = oscilador_harmonico(init,'base')
t,vm,xm = oscilador_harmonico(init,'mod')
t,vr2,xr2 = oscilador_harmonico(init,'rk2')
t,vr4,xr4 = oscilador_harmonico(init,'rk4')

plt.plot(t,xs,label="Special")
plt.plot(t,xb,label="Euler")
plt.plot(t,xm,label="Modified Euler")
plt.plot(t,xr2,label="Runge Kutta 2")
plt.plot(t,xr4,label="Runge Kutta 4")
plt.plot(t,0.2*np.cos(omega*t),label="Exact Solution")
plt.legend()
plt.show()
