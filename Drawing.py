import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_widths
from numpy.fft import fft

from Fitting import Fit


def PlotFit(x, y, sigy=None, func: 'function'=None, p0=None,
         xlabel: str='X', ylabel: str='Y', log: bool=False, marker: str='s', 
         markersize: float=12, markeredge: str='k', markerface: str='none'):
    
    popt,pcov,r = Fit(x,y,func,sigy,p0,mute=False)
        
    if sigy is not None:
        fig,ax = plt.subplots(2,figsize=(8,6),sharex=True,gridspec_kw={'height_ratios': [3, 1],'hspace':0.05})
                
        ax[0].errorbar(x, y, sigy, fmt=marker, ecolor=markeredge, markersize=markersize,
                       markerfacecolor=markerface, markeredgecolor=markeredge)
        if len(x)<50:
            start = x[0]
            finish = x[len(x)-1]
            u = np.linspace(start,finish,50)
        else:
            u = x
        ax[0].plot(u,func(u,*popt),c='r')
        ax[1].errorbar(x, r, sigy, fmt=marker, ecolor=markeredge, markersize=markersize,
                       markerfacecolor=markerface, markeredgecolor=markeredge)
        
        # scales types
        if log:
            ax[0].set_xscale('log')
            ax[1].set_xscale('log')
    
        ax[0].grid()
        ax[1].grid()
        ax[0].set_ylabel(ylabel,size=12)
        ax[1].set_ylabel('$Residues$',size=12)

    else:
        fig,ax = plt.subplots(figsize=(8,6))
                
        ax.scatter(x,y,marker=marker,facecolors=markerface,edgecolors=markeredge,s=markersize)
        if len(x)<50:
            start = x[0]
            finish = x[len(x)-1]
            u = np.linspace(start,finish,50)
        else:
            u = x
        ax.plot(u,func(u,*popt),c='r')

        # scales types
        if log:
            ax.set_xscale('log')
    
        ax.grid()
        ax.set_ylabel(ylabel,size=12)
    
    plt.xlabel(xlabel,size=12)
                  
    return (ax,popt,pcov)

#=====================================================================
#=====================================================================

def Plot(x, y, sigy=None, xlabel: str='X', ylabel: str='Y', log: bool=False, 
         marker: str='s', markersize: float=12, markeredge: str='k', markerface: str='none') -> plt.Axes:
    
    if sigy!=None:
        fig,ax = plt.subplots()
        ax.errorbar(x, y, sigy, fmt=marker, ecolor=markeredge,
                       markerfacecolor=markerface, markeredgecolor=markeredge)
        # scales types
        if log:
            ax.set_xscale('log')
          
        ax.grid()
        ax.set_ylabel(ylabel,size=12)
        
    else:
        fig,ax = plt.subplots(figsize=(8,6))
                
        ax.scatter(x,y,marker=marker,facecolors=markerface,edgecolors=markeredge,s=markersize)
        
        # scales types
        if log:
            ax.set_xscale('log')
    
        ax.grid()
        ax.set_ylabel(ylabel,size=12)
    
    plt.xlabel(xlabel,size=12)
                  
    return ax

#=====================================================================
#=====================================================================

def FFTPlot(x, y, ylabel: str="FFT Amplitude", xlabel: str='Frequency',rel_height=1):
    N = len(x)
    Fs = N/(x[N-1]-x[0])
    Yk = fft(y)

    freq = np.arange(0,Fs/2,Fs/N)
    A = abs(Yk[:len(freq)])/(len(freq))

    peaks_index, _ = find_peaks(A, prominence=0.2*max(A))
    fpeaks = np.array(freq[peaks_index])
    Apeaks = np.array(A[peaks_index])

    width,height,left,right = peak_widths(A,peaks_index,rel_height=rel_height)
    right = right*Fs/N
    left = left*Fs/N
    width = right-left
    annots = [f"${round(f,3)}$" for f in fpeaks]
 

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(freq,A,c='k')
    ax.plot(fpeaks, Apeaks+0.015*Apeaks,"vr")
    for i, txt in enumerate(annots):
        ax.annotate(txt, (fpeaks[i], Apeaks[i]+0.03*Apeaks),fontsize=12,ha='center')
    ax.hlines(height,left,right, color="C3")


    ax.grid()
    ax.set_ylabel(ylabel,size=12)
    ax.set_xlabel(xlabel,size=12)

    return ax,fpeaks, [width,height,left,right]

#=====================================================================
#=====================================================================

def GenStatTable(function: str, y, yfit, popt, pcov, sigy=None):
    # FUNCTION TYPE
    txts = [function]
    table = ""

    # FITTED PARAMETERS
    i = 0
    dic = 'abcdefg' 
    while i < len(popt):
        k = popt[i]
        sigma_k = (pcov[i][i]**0.5)
        txts.append(f'${dic[i]} = {k:.4g} \pm {sigma_k:.4g}$')
        i+=1
    
    # R-SQUARED
    r2 = R_squared(y,yfit)
    txts.append(f"$R^2 = {r2:.5g}$")

    # CHI^2
    if sigy is not None:
        chiq = sum(((y-yfit)/sigy)**2)
        txts.append(f"$\chi^2 = {chiq:.5g}$")
        NGL = len(y)-len(popt)
        txts.append(f"$NGL = {NGL:.5g}$")

    for txt in txts:
        table += txt + "\n"
    
    return table[:-1]

#=====================================================================
#=====================================================================

# terminar e testar!!
def TableImg(data,step=1):
    fig = plt.figure(figsize=(4,3), dpi=200)
    ax = plt.subplot(111)

    columns = data.keys()
    ncols = len(columns)
    nrows = len(data)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_axis_off()

    j = 0 
    for col in columns:
        ax.annotate(
            xy=(0.5+j, nrows),
            text=col,
            weight='bold',
            ha='center'
        )
        for y in range(nrows):
            ax.annotate(
                xy=(0.5+j,y),
                text=data[col][y],
                ha='center'
            )
        j+=step
    
    # Add dividing lines
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
    for x in range(1, nrows):
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')


    plt.savefig(
        'figures/a_very_basic_table.png',
        dpi=300,
        transparent=True
    )