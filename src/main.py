import os
import numpy as np
from treasury_curve import get_latest_yield_curve
from data_loader import get_option_data
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
from hestong_calibration_config import param_config
from heston_simulator import *
from heston_model import *

def main():
    ticker = "AAPL"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Retrieve yield curve
    print("📉 Fetching yield curve...")
    maturities, yields = get_latest_yield_curve()
    curve_fit, _ = calibrate_nss_ols(maturities, yields)

    # Step 2: Fetch options
    print(f"📦 Fetching option chain for {ticker}...")
    spot, hist, df, n = get_option_data(ticker, curve_fit, output_dir, min_volume=500)
    print(f"✅ Loaded {n} options | Spot price: {spot:.2f}")
    
    print("⚙️ Calibrating Heston model...")
    df = df[df["type"] == "call"]
    S0 = spot
    r = df["rate"].to_numpy("float")
    K = df["strike"].to_numpy("float")
    tau = df["ttm"].to_numpy("float")
    P = df["price"].to_numpy("float")

    x0 = [param["x0"] for param in param_config.values()]
    bnds = [param["bounds"] for param in param_config.values()]

    result = calibrate_heston_model(S0, K, tau, r, P, x0, bnds)
    print("📈 Calibration result:", result)
    v0, kappa, theta, sigma, rho, lambd = [param for param in result.x]
    heston_prices = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
    df['heston_price'] = heston_prices
    
    # Step 4: Generate Heston prices and compare
    v0, kappa, theta, sigma, rho, lambd = result.x
    heston_prices = np.array([
        heston_price_rec(S0, K_i, v0, kappa, theta, sigma, rho, lambd, tau_i, r_i)
        for K_i, tau_i, r_i in zip(K, tau, r)
    ])
    df['heston_price'] = heston_prices


if __name__ == "__main__":
    main()
