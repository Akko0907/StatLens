import scipy.stats as st
import numpy as np

from Diffs import Grad

def Prop_sig(vals: 'tuple', sigs: 'tuple', func: 'function') -> float:
  ''' Propagate Uncertainties. Receives 2 lists/tuples of values representing
    the point/uncertainty, and the function relating the variables.

    Example: 
    
    f(x, y, z) receives the tuple vals=(0.2, 1, 3.75) and sigs=(0.05, 0.1, 0.32) where,

    x = 0.2+-0.05
   
    y = 1+-0.1

    z = 3.75+-0.32

    returns the uncertainty of f(0.2, 1, 3.75).  
  '''
  
  grad = Grad(vals,func)
  sigs = np.array(sigs)
  sigf = np.sqrt( np.sum( grad**2 * sigs**2 ) )

  return sigf

#=====================================================================
#=====================================================================

def Z_score(x: float, mu: float, sigx: float, sigmu: float=0) -> float:
  '''Returns The Z-score of X'''
  sig = (sigx**2 + sigmu**2)**0.5
  Z = abs(x-mu)/sig

  return Z

#=====================================================================
#=====================================================================

def R_sqr(y, yfit) -> float:
    """Returns the R^2 for the fit."""
    r = y- yfit
    ss_res = np.sum(r**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared

#=====================================================================
#=====================================================================

def Full_tests(x, y, y_model, popt, p=0.95) -> 'tuple[float]':
  '''Returns (t, s_err, r_squared, chiq, ci, pi, dof) \n
  t -> t-statistics\n
  s_err -> deviation of error\n
  r_squared -> avaliate goodness of fit\n
  chiq -> avaliate goodness of fit\n
  ci -> confidence interval\n
  pi -> prediction interval\n
  dof -> degrees of freedom
  '''
  
  n = y.size
  dof = n - popt.size                           
  r = y - y_model

  ss_res = np.sum(r**2)
  ss_tot = np.sum((y-np.mean(y))**2)
  c = 1/n + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2)

  t = st.t.ppf(p, dof)               
  chiq = np.sum(r**2/y_model) 
  r_squared = 1 - (ss_res / ss_tot)
  s_err = np.sqrt(ss_res / dof)
  ci = t*s_err*np.sqrt(c)
  pi = t*s_err*np.sqrt(c+1)

  return t, s_err, r_squared, chiq, ci, pi, dof

#=====================================================================
#=====================================================================
