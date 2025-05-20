import numpy as np
from scipy.optimize import minimize
from numba import njit

@njit
def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """
    Characteristic function of the Heston model.
    """
    a = kappa * theta
    b = kappa + lambd
    rspi = rho * sigma * phi * 1j
    d = np.sqrt((rspi - b)**2 + (phi * 1j + phi**2) * sigma**2)
    g = (b - rspi + d) / (b - rspi - d + 1e-10)  # prevent div-by-zero

    exp1 = np.exp(r * phi * 1j * tau)
    term2 = S0**(phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g + 1e-10))**(-2 * a / sigma**2)
    exp2 = np.exp(
        a * tau * (b - rspi + d) / sigma**2 +
        v0 * (b - rspi + d) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau) + 1e-10)) / sigma**2
    )
    return exp1 * term2 * exp2

@njit
def heston_price_rec_single(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r, umax=100, N=2000):
    """
    Compute Heston price using rectangular integration.
    """
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    P = 0.0
    dphi = umax / N

    for i in range(1, N):
        phi = dphi * (2 * i + 1) / 2
        num = np.exp(r * tau) * heston_charfunc(phi - 1j, *args) - K * heston_charfunc(phi, *args)
        den = 1j * phi * K**(1j * phi)
        P += dphi * num / (den + 1e-10)

    return np.real((S0 - K * np.exp(-r * tau)) / 2 + P / np.pi)

def calibrate_heston_model(S0, K, tau, r, P, x0, bnds, verbose=False):
    """
    Calibrate Heston model parameters to market prices using least squares.
    """
    def SqErr(x):
        v0, kappa, theta, sigma, rho, lambd = x

        # Early reject if parameters are unphysical
        if v0 <= 0 or theta <= 0 or sigma <= 0 or kappa <= 0 or not (-1 < rho < 0.01):
            return 1e10

        prices = np.array([
            heston_price_rec_single(S0, K_i, v0, kappa, theta, sigma, rho, lambd, tau_i, r_i)
            for K_i, tau_i, r_i in zip(K, tau, r)
        ])

        # Mean squared error
        mse = np.mean((P - prices)**2)

        # Penalty for negative prices
        penalty = np.sum(np.maximum(0, -prices)) * 1e3


        if verbose:
            print(f"x = {x}, MSE = {mse:.6f}, Penalty = {penalty:.2f}")

        return mse + penalty

    result = minimize(SqErr, x0, tol=1e-4, method='SLSQP', options={'maxiter': 3000}, bounds=bnds)
    return result

def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """
    High-accuracy Heston price with dense integration (N = 10000).
    """
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    P, umax, N = 0.0, 100, 10000
    dphi = umax / N

    for i in range(1, N):
        phi = dphi * (2 * i + 1) / 2
        num = np.exp(r * tau) * heston_charfunc(phi - 1j, *args) - K * heston_charfunc(phi, *args)
        den = 1j * phi * K**(1j * phi)
        P += dphi * num / (den + 1e-10)

    return np.real((S0 - K * np.exp(-r * tau)) / 2 + P / np.pi)
