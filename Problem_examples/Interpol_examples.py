import matplotlib.pyplot as plt
import numpy as np
import Fitting as interpol

def Problem1_1():
    # Use of LinSpline
    xk = np.linspace(1,3,11)
    yk = 1/xk +np.cos(3*xk)**2

    x,y = interpol.Spline(xk,yk,1,step=0.02)

    x_real = np.linspace(1,3,100)
    y_real = 1/x_real +np.cos(3*x_real)**2

    plt.plot(x,y,c='purple',label="Linear Spline")
    plt.scatter(xk,yk,c='k',label="Data points")
    plt.plot(x_real,y_real,c='r',label="Real function")
    plt.legend()
    plt.show()

def Problem1_2():
    # Use of cubic spline
    xk = np.linspace(1,3,11)
    yk = 1/xk +np.cos(3*xk)**2

    x,y = interpol.Spline(xk,yk,3,step=0.02)

    x_real = np.linspace(1,3,100)
    y_real = 1/x_real +np.cos(3*x_real)**2

    plt.plot(x,y,c='purple',label="Cubic Spline")
    plt.scatter(xk,yk,c='k',label="Data points")
    plt.plot(x_real,y_real,c='r',label="Real function")
    plt.legend()
    plt.show()

def Problem1_3():
    # Use of Lagrange
    xk = np.linspace(1,3,11)
    yk = 1/xk +np.cos(3*xk)**2

    f = interpol.Lagrange(xk,yk)
    x = np.arange(1,3+0.02,0.02)
    y = f.at(x)

    x_real = np.linspace(1,3,100)
    y_real = 1/x_real +np.cos(3*x_real)**2

    plt.plot(x,y,c='purple',label="Lagrange Interpolation")
    plt.scatter(xk,yk,c='k',label="Data points")
    plt.plot(x_real,y_real,c='r',label="Real function")
    plt.legend()
    plt.show()

def Problem1_4():
    # Use of Newton
    xk = np.linspace(1,3,11)
    yk = 1/xk +np.cos(3*xk)**2

    f = interpol.ForwardNewton(xk,yk)
    x = np.arange(1,3+0.02,0.02)
    y = f.at(x)

    x_real = np.linspace(1,3,100)
    y_real = 1/x_real +np.cos(3*x_real)**2

    plt.plot(x,y,c='purple',label="Forward Newton Interpolation")
    plt.scatter(xk,yk,c='k',label="Data points")
    plt.plot(x_real,y_real,c='r',label="Real function")
    plt.legend()
    plt.show()

def GaussPack():
    pack = lambda x: np.exp(-x**2)

    xk = np.linspace(-5,5,14)
    yk = pack(xk)

    f_newton = interpol.ForwardNewton(xk,yk)
    f_lagrange = interpol.Lagrange(xk,yk)

    x = np.arange(-5,5.01,0.01)

    y_newton = f_newton.at(x)
    y_lagrange = f_lagrange.at(x)

    x_spline2,y_spline2 = interpol.Spline(xk,yk,2)
    x_spline,y_spline = interpol.Spline(xk,yk,3)

    plt.scatter(xk,yk,c='g')
    plt.plot(x,y_newton,label='newton',c='k')
    plt.plot(x,y_lagrange,label='lagrange',c='r')
    plt.plot(x_spline,y_spline,label='Cubic spline',c='y')
    plt.plot(x_spline2,y_spline2,label='spline2',c='c')
    plt.legend()
    plt.show()