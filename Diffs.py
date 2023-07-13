import numpy as np
from Fitting import Spline


def CurveDiff(x: np.ndarray, y: np.ndarray, sigy: np.ndarray, sigx: np.ndarray) -> 'tuple[np.ndarray]':
  f = lambda x,y: np.diff(y)/np.diff(x)
  g = lambda x: x[:-1]+np.diff(x)/2
  sigf = 1/np.diff(x) * np.sqrt(sigy[:-1]**2+sigy[1:]**2 + f(x,y)**2 *(sigx[:-1]**2 + sigx[1:]**2) )
  sigdx = sigx[:-1]/np.sqrt(2)

  df = f(x,y)
  dx = g(x)
  return dx, df, sigf, sigdx

#=====================================================================
#=====================================================================

def diff(x_vec: np.ndarray,f: function) -> np.ndarray:
    """Uses finite differences to generate a derivatives table of f"""
    del_f = np.diff(f(x_vec))/np.diff(x_vec)
    return del_f


def Grad(point: 'list|tuple', func: 'function', divs: int=1000) -> np.ndarray:
    '''Returns the Gradient of a function at a specific point.'''
    argc = len(point)

    M = np.array( [np.linspace(var-0.1*var, var+0.1*var, divs) for var in point] )
    h = np.array( [M[i][1]-M[i][0] for i in range(argc)] )
    
    F2 = np.array( [func(*M[:,i]) for i in range(1,divs)] )
    F1 = np.zeros((argc,divs-1))
    for i in range(argc):
        for j in range(1,divs):
            coord = [ M[k][j-1] if k==i else M[k][j] for k in range(argc) ]
            F1[i][j-1] = func(*coord)
        
    df = F2-F1
    diff = df[:,divs//2-1]/h
    
    return diff

#=====================================================================
#=====================================================================

def Simps(x_data: np.ndarray, y_data: np.ndarray, interval: list=None, step: float=0.01, interpol: bool=False) -> float:
    """Integration by Simpsons method inside some interval [a,b]"""

    if interval==None:
        b = x_data[-1]
        a = x_data[0]
    
    if interpol==True:
        x,y = Spline(x_data,y_data,3,step=step)
    else:
        x,y = x_data,y_data
        step = x[1]-x[0]

    s = y[0] + y[-1]
    for i in range(1,len(y)-1):
        if i%2:
            s+=4*y[i]
        else:
            s+=2*y[i]
    I = step*s/3

    print(f"\nIntegration at Interval= [{a},{b}] is:\nI={I}\n")
    return I

#=====================================================================
#=====================================================================

def Trapz(x_data: np.ndarray, y_data: np.ndarray, interval: list=None, step: float=0.01, interpol: bool=False) -> float:
    """Integration by trapzoid method inside some interval [a,b]"""
    
    if interval==None:
        b = x_data[-1]
        a = x_data[0]
    
    if interpol==True:
        x,y = Spline(x_data,y_data,3,step=step)
    else: 
        x,y = x_data,y_data
        step = x[1:]-x[:-1]

    y_mean = (y[1:]+y[:-1])/2

    I = sum(y_mean*step)

    print(f"\nIntegration at Interval= [{a},{b}] is:\nI={I}\n")
    return I

#=====================================================================
#=====================================================================
