import numpy as np
import matplotlib.pyplot as plt
from time import time
from Diffs import Trapz, Simps

def Problem1_1():
    # Big data_set
    x = np.linspace(1,3,100)
    y = 1/x
    
    print("\nBy Trapz:")
    Trapz(x,y)
    
    print("\nBy Simps:")
    Simps(x,y)

def Problem1_2():
    # Small data_set
    x = np.linspace(1,3,10)
    y = 1/x
    
    print("\nBy Trapz:")
    Trapz(x,y,step=0.02,interpol=True)
    
    print("\nBy Simps:")
    Simps(x,y,step=0.02,interpol=True)

def Problem1_3():
    # Medium data_set
    x = np.linspace(1,3,50)
    y = 1/x
    
    print("\nBy Trapz:")
    Trapz(x,y,step=0.02,interpol=True)

    print("\nBy Simps:")
    Simps(x,y,step=0.02,interpol=True)
        