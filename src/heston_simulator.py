import numpy as np

def heston_model_sim(S0, v0, rho, kappa, theta, sigma,T, N, M, r):
    """
    Inputs:
     - S0, v0: initial parameters for asset and variance
     - rho   : correlation between asset returns and variance
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - T     : time of simulation
     - N     : number of time steps
     - M     : number of scenarios / simulations

    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # initialise other parameters
    dt = T/N
    mu = np.array([0,0])
    cov = np.array([[1,rho],
                    [rho,1]])

    # arrays for storing prices and variances
    S = np.full(shape=(N+1,M), fill_value=S0)
    v = np.full(shape=(N+1,M), fill_value=v0)

    # sampling correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N,M))

    for i in range(1,N+1):
        S[i] = S[i-1] * np.exp( (r - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0] )
        v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)

    return S, v


def vanilla_option_price_heston(S_paths, K, r, T, option_type="call"):
    """
    Prices a European vanilla call or put option using Heston Monte Carlo paths.

    Parameters:
        S_paths      : Simulated price paths (shape: [N+1, M])
        K            : Strike
        r            : Risk-free rate
        T            : Time to maturity
        option_type  : "call" or "put"

    Returns:
        Discounted expected payoff (float)
    """
    ST = S_paths[-1]
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

def digital_option_price_heston(S_paths, K, r, T, option_type="call", cash_payout=1.0):
    """
    Prices a digital (binary) call or put option using Heston Monte Carlo paths.
    Pays fixed amount (cash_payout) if in-the-money at maturity.

    Parameters:
        S_paths      : Simulated price paths from heston_model_sim (shape: [N+1, M])
        K            : Strike price
        r            : Risk-free rate
        T            : Time to maturity
        option_type  : "call" or "put"
        cash_payout  : Fixed payout if condition is met (default: 1.0)

    Returns:
        Discounted expected payout (float)
    """
    ST = S_paths[-1]
    if option_type == "call":
        payoff = np.where(ST > K, cash_payout, 0)
    else:
        payoff = np.where(ST < K, cash_payout, 0)
    return np.exp(-r * T) * np.mean(payoff)

def barrier_option_price_heston(S_paths, K, r, T, barrier, barrier_type="up-and-out", option_type="call"):
    """
    Prices a barrier option using Heston Monte Carlo.

    Parameters:
        S_paths       : Simulated paths (shape: [N+1, M])
        K             : Strike
        r             : Risk-free rate
        T             : Time to maturity
        barrier       : Barrier level
        barrier_type  : "up-and-out", "down-and-out", "up-and-in", or "down-and-in"
        option_type   : "call" or "put"

    Returns:
        Discounted expected payoff (float)
    """
    ST = S_paths[-1]
    hit_barrier = np.any(S_paths >= barrier, axis=0) if "up" in barrier_type else np.any(S_paths <= barrier, axis=0)

    if "out" in barrier_type:
        valid_paths = ~hit_barrier
    else:
        valid_paths = hit_barrier

    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    return np.exp(-r * T) * np.mean(payoff[valid_paths])


def bull_call_spread_price_heston(S_paths, K1, K2, r, T):
    """
    Bull Call Spread: long call at K1, short call at K2.
    Reuses vanilla call pricing.
    """
    long_call = vanilla_option_price_heston(S_paths, K1, r, T, option_type="call")
    short_call = vanilla_option_price_heston(S_paths, K2, r, T, option_type="call")
    return long_call - short_call


def bear_put_spread_price_heston(S_paths, K1, K2, r, T):
    """
    Bear Put Spread: long put at K2, short put at K1.
    Reuses vanilla put pricing.
    """
    long_put = vanilla_option_price_heston(S_paths, K2, r, T, option_type="put")
    short_put = vanilla_option_price_heston(S_paths, K1, r, T, option_type="put")
    return long_put - short_put


def straddle_price_heston(S_paths, K, r, T):
    """
    Straddle: long call + long put at same strike K.
    Reuses vanilla call and put.
    """
    call = vanilla_option_price_heston(S_paths, K, r, T, option_type="call")
    put = vanilla_option_price_heston(S_paths, K, r, T, option_type="put")
    return call + put


def strangle_price_heston(S_paths, K1, K2, r, T):
    """
    Strangle: long put at K1, long call at K2.
    Reuses vanilla call and put.
    """
    put = vanilla_option_price_heston(S_paths, K1, r, T, option_type="put")
    call = vanilla_option_price_heston(S_paths, K2, r, T, option_type="call")
    return put + call
