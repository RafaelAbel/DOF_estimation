# Version alpha by R. Abel and W. Rath (1.12.2016)



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp
from scipy.stats import binom
import random
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, betai
np.random.seed(seed=None) # important, that we are not starting with the same random numbers

# ================================================================================ 
def correlation_matrix_vector(matrix, vector):
    '''Description here'''
    '''Matrix shape T,N'''
    '''Vector shape T  '''
    r = np.ones(shape=(matrix.shape[0]))
    p = np.ones(shape=(matrix.shape[0]))
    nt = matrix.shape[0] #Time dimension
    data1_norm = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    data2_norm = (vector - vector.mean()) / vector.std()
    r = np.sum(np.swapaxes(data1_norm,0,1) * data2_norm / float(nt), axis=1)
    df = nt-2 #DOF
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    p = betai(0.5*df, 0.5, df / (df + t_squared))
    return r,p

def bernoulli_distribution(var,p):
    '''var has shape T,N'''
    '''builds underlying distribution where we draw from'''
    signifi=np.zeros(var.shape[1])*np.nan
    R_t= np.random.normal(0.5,0.1,var.shape[0])
    _ , signifi=correlation_matrix_vector(var,R_t)
    return(signifi <= p)

def bernoulli_trial(distribution,n):
    '''draw n times from distribution'''
    realisation=np.sum(distribution[np.random.randint(low=0,high=distribution.shape[0],size=n)],dtype=int)
    return(realisation)

def find_m(array,p):
    if (array.min() >= p) or (array.max() <= p):
        m=np.nan
    else:
        max = np.sum([(array-p) > 0])-1 #closest positive point to p
        f=interp1d(np.arange(max-1,max+2),array[np.arange(max-1,max+2)],kind='quadratic')
        x = np.linspace(max-1, max+1, num=4*100+1, endpoint=True)
        ynew = f(x)   # use interpolation function returned by `interp1d`
        max = np.sum([(ynew-p) > 0])-1
        m=x[max]
    return m

def get_m_n_from_bernoulli(N):
	p,P_B =  0.05,0.05
	m_n_bernoulli=np.arange(1,N)*np.nan
	for n in np.arange(1,N):
	    x = np.arange(binom.ppf(0.00, n, p),binom.ppf(1.00, n, p))
	    prob = binom.sf(x, n, p)
	    m = find_m(prob,P_B)
	    m_n_bernoulli[n-1]=m*1./n
	return(m_n_bernoulli)

def find_dof(m_n,m_n_bernoulli):
    dof=m_n*np.nan
    for i in np.arange(m_n.shape[0]):
        dof[i] = np.sum([(m_n_bernoulli-m_n[i]) > 0])-1 
    return dof

def calc_m_n(var,p,P_B,S):
        var=var[:,~np.isnan(var.mean(axis=0))] # keep only non-nan values
        n=var.shape[1]
        realisations=np.zeros((S))
        for i in np.arange(S): # S times (Montecarlo simulation)
            distribution=bernoulli_distribution(var,p)
            realisations[i]=bernoulli_trial(distribution,n)
        ranges=np.arange(0,n+1,1)
            #print 'ranges',ranges
        binom_hist, bins, patches =\
        plt.hist(realisations,ranges, normed=1,histtype='step', cumulative=-1) 
        plt.close()
        #print "binom_hist",binom_hist
        #binom_hist, bins = hist_cum(realisations.astype(int),n)
        m = find_m(binom_hist,P_B)
        m_n = m*1./n 
        #print m_n
        return(m_n, m,n)

def B_method(var,S=None,estimates=None,p=None,P_B=None):
    'DOF=B_method(var,S,estimates)'
    'var needs dimensions T,N'
    if (S==None):
	S=1000
    if (p==None):
	p=0.05
    if (P_B==None):
	P_B=0.05
    if (estimates==None):
	estimates=1
    m_n_bernoulli=get_m_n_from_bernoulli(var.shape[1])
    m_n,m=np.arange(estimates)*np.nan,np.arange(estimates)*np.nan
    for i in np.arange(estimates):
        m_n[i],m[i],n=calc_m_n(var,p,P_B,S)
    dof=find_dof(m_n,m_n_bernoulli)
    return dof
# ================================================================================ 
