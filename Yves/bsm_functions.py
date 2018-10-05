
from math import log,sqrt,exp
from scipy import stats
import pandas as pd


"""
Valuation of European call options in BSM model including 
Vega function and implied volatility estimation
"""

#Analytical BSM formula
def bsm_call_value(S0,K,T,r,sigma):
    """
    parameters :
    S0 : float ; initial stock / index level
    K  : float ; strike price
    T  : float ; maturity date(in year fractions)
    r  : float ; constant risk free short rate
    sigma : float ; volatility factor in diffusion term
    
    
    Return :
    value : float ; present value of the European call option
    """
    
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2)*T )/ (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma**2)*T )/ (sigma * sqrt(T))
    
    value = (S0 * stats.norm.cdf(d1,0.0,1.0) - K*exp(-r*T) * stats.norm.cdf(d2,0.0,1.0))
    
    return value

# Vega function
def bsm_vega(S0,K,T,r,sigma):
    """
    parameters :
    S0 : float ; initial stock / index level
    K  : float ; strike price
    T  : float ; maturity date(in year fractions)
    r  : float ; constant risk free short rate
    sigma : float ; volatility factor in diffusion term
    
    Return :
    vega : float ; partial derivative of BSM formula with respect to sigma
    """
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2)*T )/ (sigma * sqrt(T))
    vega = S0*stats.norm.cdf(d1,0.0,1.0)*sqrt(T)
    return vega

# implied volatility function
def bsm_call_imp_vol(S0,K,T,r,C0,sigma_est,it=100):
    """
    parameters :
    S0 : float ; initial stock / index level
    K  : float ; strike price
    T  : float ; maturity date(in year fractions)
    r  : float ; constant risk free short rate
    sigma_est : float ; estimate of implied volatility
    it : integer ; number of iterations
    
    
    Return :
    sigma_est : float ; numerically estimated implied volatility
    """
    
    for i in range(it):
        sigma_est -=((bsm_call_value(S0,K,T,r,sigma_est) - C0) / bsm_vega(S0,K,T,r,sigma_est))
        
    return sigma_est
