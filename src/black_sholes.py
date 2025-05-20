from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np

def black_scholes_price(S, K, t, r, sigma, option_type='call'):
    if t <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    else:  # put
        return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(market_price, S, K, t, r, option_type='call'):
    try:
        return brentq(
            lambda sigma: black_scholes_price(S, K, t, r, sigma, option_type) - market_price,
            a=1e-5, b=5.0, maxiter=500
        )
    except:
        return np.nan
