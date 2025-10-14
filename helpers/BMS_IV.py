import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, golden

def BMS_with_dividends(
    option_type: str,
    s_hat: float,
    k: float,
    r: float,
    ttm: float,
    sigma: float,
) -> float:
    """
    Calculate the Black-Scholes-Merton price of a European option with discrete dividends.

    Args:
        option_type (str) : Option type: 'C' for call, 'P' for put 
        s_hat (float) : The ex-dividend stock price at valuation date 
        k (float) : The option strike price (K)
        r (float) : Continuously compounded risk-free rate (r)
        ttm (float) : Time to maturity in years (T-t)
        sigma (float) : Annualized volatility 

    Returns:
        price : Theoretical price of the European option under BMS with discrete dividends.
    """

    d1 = (np.log(s_hat / k) + (r + 0.5 * sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))
    d2 = d1 - sigma * np.sqrt(ttm)
    if option_type.upper() == 'C':
        price = s_hat * norm.cdf(d1) - k * np.exp(-r * ttm) * norm.cdf(d2)
    elif option_type.upper() == 'P':
        price = k * np.exp(-r * ttm) * norm.cdf(-d2) - s_hat * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'C' for call or 'P' for put.")
    return price

def implied_vol_BMS(option_type: str, S_hat: float, K: float, r: float, ttm: float, option_price: float, opt_method: str) -> float:
    """Finding the implied volatility using Golden Section Search

    Args:
        option_type (str): Type of the option, 'C' for call, 'P' for put 
        option_price (float): Market price of the option
        ttm (float): Time to maturity in years
        K (float): strike price
        S_hat (float): The ex-dividend stock price at valuation date 
        r (float): risk free rate
        opt_method (str): Optimization method to use ('brentq' or 'golden')

    Returns:
        float: calibrated implied volatility
    """
    if opt_method != "brentq" and opt_method != "golden":
        raise ValueError("opt_method must be 'brentq' or 'golden'.")

    def objective_func(sigma):
        model_price = BMS_with_dividends(option_type,S_hat, K, r, ttm, sigma)
        market_price = option_price
        if opt_method == "brentq":
            return (model_price - market_price)
        elif opt_method == "golden":
            return (model_price - market_price) ** 2
        
    if opt_method == "golden":
        sigma_hat = golden(objective_func, 
                       brack=(1e-8, 50.0),  # interval
                       tol=1e-5,  # tolerance for stopping criterion
                        maxiter=1000,  # maximum number of iterations
                       )
    elif opt_method == "brentq":
        sigma_hat = brentq(objective_func, 
                           1e-7,  # lower bound
                           10.0,  # upper bound
                           maxiter=1000)

    return sigma_hat