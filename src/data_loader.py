import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import json
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

def get_option_data(ticker_symbol, curve_function, output_dir='output', min_volume=50, min_ttm = 0.02):
    ticker = yf.Ticker(ticker_symbol)

    # Get full historical prices
    spot_hist = ticker.history(period="max")
    if spot_hist.empty:
        raise ValueError(f"No historical price data found for {ticker_symbol}")

    spot = spot_hist["Close"].iloc[-1]

    # Define strike range
    lower_bound = 0.8 * spot
    upper_bound = 1.1 * spot

    # Fetch and filter option chains
    all_data = []
    for expiry in ticker.options:
        try:
            chain = ticker.option_chain(expiry)
            for opt_type, df in [('call', chain.calls), ('put', chain.puts)]:
                df = df[(df["volume"] > min_volume) & 
                        (df["strike"] >= lower_bound) & 
                        (df["strike"] <= upper_bound)].copy()
                if df.empty:
                    continue
                df["maturity"] = expiry
                df["type"] = opt_type
                all_data.append(df[["maturity", "strike", "lastPrice", "type"]])
        except Exception as e:
            print(f"Skipping {expiry}: {e}")
            continue

    if not all_data:
        raise ValueError("No liquid options found in the specified strike range.")

    price_df = pd.concat(all_data, ignore_index=True).rename(columns={"lastPrice": "price"})
    
    # Add rate for each option using the calibrated yield curve function
    today = datetime.today()
    price_df["ttm"] = price_df["maturity"].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d") - today).days / 365.25)
    price_df["ttm"] = price_df["ttm"].astype(float)
    price_df["rate"] = price_df["ttm"].apply(curve_function)
    price_df = price_df[price_df["ttm"] >= min_ttm]
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    base = ticker_symbol.lower()

    # Save options data and spot price
    options_filename = f"{output_dir}/{base}_options.csv"
    spot_filename = f"{output_dir}/{base}_spot.json"
    
    price_df.to_csv(options_filename, index=False)
    with open(spot_filename, "w") as f:
        json.dump({"spot": spot}, f, indent=4)

    return spot, spot_hist, price_df.reset_index(drop=True), len(price_df)
