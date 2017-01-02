# Version alpha by R. Abel and W. Rath (1.12.2016)

# Estimation of Effective Sample Size (Bretherton et al. (1999))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp
import random
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, betai

# ================================================================================
def correlation_2_arrays(data1, data2, axis = 0):
    '''Description here'''
    r = np.ones(shape=(data1.shape[0]))
    p = np.ones(shape=(data1.shape[0]))
    nt = data1.shape[axis]
    assert data1.shape == data2.shape
    view1 = data1
    view2 = data2

    if axis:
        view1 = np.rollaxis(data1, axis)
        view2 = np.rollaxis(data2, axis)

    data1_norm = (view1 - data1.mean(axis=axis)) / data1.std(axis=axis)
    data2_norm = (view2 - data2.mean(axis=axis)) / data2.std(axis=axis)
    r = np.sum(data1_norm * data2_norm / float(nt), axis=0)
    
    df = nt-2
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    p = betai(0.5*df, 0.5, df / (df + t_squared))
    return r,p

def lag_correlation(x,y,lag):
    '''Y shifted by + lag timesteps'''
    r,p=correlation_2_arrays(x[0:-lag,:,:],y[lag:,:,:],axis=0)
    return(r,p)

def ESS(x,y):
    '''effective sample size (ESS) ratio'''
    '''X has dim (T,X,Y)'''
    assert x.shape == y.shape
    N=len(x)
    r1,p1=lag_correlation(x,x,lag=1)
    r2,p2=lag_correlation(y,y,lag=1)
    N_star= N *(1 - r1*r2)/(1+r1*r2)
    ratio=(1 - r1*r2)/(1+r1*r2)
    return(ratio)

