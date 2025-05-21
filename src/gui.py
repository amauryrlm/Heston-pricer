import streamlit as st
import os
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata
import plotly.graph_objects as go

from treasury_curve import get_latest_yield_curve
from data_loader import get_option_data
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
from heston_model import *
from hestong_calibration_config import param_config
from heston_simulator import (
    vanilla_option_price_heston,
    digital_option_price_heston,
    bull_call_spread_price_heston,
    bear_put_spread_price_heston,
    straddle_price_heston,
    strangle_price_heston,
    heston_model_sim
)

def run_gui():
    st.set_page_config(layout="wide")
    st.title("Heston Model Option Pricer")


    # --- User Input Section ---
    ticker = st.text_input("Enter Ticker", value="AAPL")
    force_download = st.checkbox("Force Download Yield Curve", value=False)
    min_volume = st.number_input("Minimum Volume Filter", min_value=0, value=50, step=10)

    if "spot" in st.session_state:
        st.markdown(f"### 📍 Current Spot Price: **${st.session_state['spot']:.2f}**")

    if st.button("Get Data & Calibrate"):
        try:
            os.makedirs("output", exist_ok=True)
            maturities, yields = get_latest_yield_curve(force_download=force_download)
            curve_fit, _ = calibrate_nss_ols(maturities, yields)

            spot, hist, df, n = get_option_data(ticker, curve_fit, "output", min_volume=min_volume)
            df = df[df["type"] == "call"]

            S0 = spot
            r = df["rate"].to_numpy("float")
            K = df["strike"].to_numpy("float")
            tau = df["ttm"].to_numpy("float")
            P = df["price"].to_numpy("float")

            x0 = [param["x0"] for param in param_config.values()]
            bnds = [param["bounds"] for param in param_config.values()]
            result = calibrate_heston_model(S0, K, tau, r, P, x0, bnds)

            v0, kappa, theta, sigma, rho, lambd = result.x
            heston_prices = np.array([
                heston_price_rec(S0, K_i, v0, kappa, theta, sigma, rho, lambd, tau_i, r_i)
                for K_i, tau_i, r_i in zip(K, tau, r)
            ])
            df["heston_price"] = heston_prices

            st.session_state.update({
                "ticker": ticker,
                "spot": spot,
                "num_options": n,
                "vol_df": df,
                "hist": hist,
                "curve_fit": curve_fit,
                "calibration_result": {
                    "v0": v0, "kappa": kappa, "theta": theta, "sigma": sigma,
                    "rho": rho, "lambd": lambd, "success": result.success,
                    "message": result.message
                }
            })

            st.success(f"✅ Loaded {n} call options and calibrated the Heston model.")

        except Exception as e:
            st.error(f"❌ Error during fetch or calibration: {e}")

    if all(k in st.session_state for k in ["vol_df", "spot", "hist"]):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧮 Option Pricer",
            "📄 Option Data",
            "📉 Spot Price",
            "📊 3D Surface",
            "📌 Calibrated Parameters"
        ])

        # --- Tab 1: Option Pricer ---
        with tab1:
            st.subheader("🧮 Option Pricing")

            df = st.session_state["vol_df"]
            spot = st.session_state["spot"]

            maturity_dates = sorted(df["maturity"].unique())
            selected_expiry = st.selectbox("Select Maturity (Expiry Date)", maturity_dates)
            today = datetime.today()
            selected_maturity = (datetime.strptime(selected_expiry, "%Y-%m-%d") - today).days / 365.25

            filtered_df = df[df["maturity"] == selected_expiry]
            strike_min = float(filtered_df["strike"].min())
            strike_max = float(filtered_df["strike"].max())
            default_strike = max(strike_min, min(strike_max, spot))

            selected_strike = st.number_input(
                "Select Strike", 
                value=default_strike
            )
            strategy = st.selectbox("Strategy Type", [
                "Vanilla", "Digital", "Bull Call Spread", "Bear Put Spread", "Straddle", "Strangle"
            ])

            option_type = None
            if strategy in ["Vanilla", "Digital"]:
                option_type = st.selectbox("Option Type", ["call", "put"], key="option_type_pricer")


            curve_fit = st.session_state["curve_fit"]
            rate = curve_fit(selected_maturity)
            calib = st.session_state["calibration_result"]
            v0, kappa, theta, sigma, rho = calib["v0"], calib["kappa"], calib["theta"], calib["sigma"], calib["rho"]

            M, N = 100000, 250
            S_paths = heston_model_sim(spot, v0, rho, kappa, theta, sigma, selected_maturity, N, M, rate)[0]

            if strategy == "Vanilla":
                price = vanilla_option_price_heston(S_paths, selected_strike, rate, selected_maturity, option_type)

            elif strategy == "Digital":
                payout = st.number_input("Digital Payout", value=1.0)
                price = digital_option_price_heston(S_paths, selected_strike, rate, selected_maturity, option_type, payout)

            elif strategy == "Bull Call Spread":
                K1 = st.number_input("Lower Strike (Long Call)", value=selected_strike - 5.0)
                K2 = st.number_input("Upper Strike (Short Call)", value=selected_strike + 5.0)
                price = bull_call_spread_price_heston(S_paths, K1, K2, rate, selected_maturity)

            elif strategy == "Bear Put Spread":
                K1 = st.number_input("Lower Strike (Short Put)", value=selected_strike - 5.0)
                K2 = st.number_input("Upper Strike (Long Put)", value=selected_strike + 5.0)
                price = bear_put_spread_price_heston(S_paths, K1, K2, rate, selected_maturity)

            elif strategy == "Straddle":
                price = straddle_price_heston(S_paths, selected_strike, rate, selected_maturity)

            elif strategy == "Strangle":
                K1 = st.number_input("Lower Strike (Put)", value=selected_strike - 10.0)
                K2 = st.number_input("Upper Strike (Call)", value=selected_strike + 10.0)
                price = strangle_price_heston(S_paths, K1, K2, rate, selected_maturity)

            st.metric(label=f"{strategy} Strategy Price", value=f"${price:.4f}")

        # --- Tab 2: Market Data ---
        with tab2:
            st.subheader("📄 Market Option Surface")
            st.dataframe(st.session_state["vol_df"])

        # --- Tab 3: Historical Spot ---
        with tab3:
            st.subheader("📉 Historical Spot Price")
            st.line_chart(st.session_state["hist"]["Close"])

        # --- Tab 4: 3D Surface ---
        with tab4:
            st.subheader("📊 Market vs Heston Option Price Surface (3D)")

            selected_type = st.selectbox("Option Type", options=["call", "put"], key="option_type_surface")
            df_filtered = st.session_state["vol_df"]
            df_filtered = df_filtered[df_filtered["type"] == selected_type]

            if df_filtered.empty or "heston_price" not in df_filtered.columns:
                st.warning("No Heston prices available for this type.")
            else:
                x = df_filtered["ttm"]
                y = df_filtered["strike"]
                z_market = df_filtered["price"]
                z_heston = df_filtered["heston_price"]

                fig = go.Figure()
                fig.add_trace(go.Mesh3d(x=x, y=y, z=z_market, opacity=0.5, color='mediumblue', name="Market"))
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z_heston, mode='markers', marker=dict(size=4, color='red'), name="Heston"))

                fig.update_layout(
                    title_text='Market Prices (Mesh) vs Calibrated Heston Prices (Markers)',
                    scene=dict(xaxis_title='TTM (Years)', yaxis_title='Strike', zaxis_title='Price'),
                    height=800, width=1000
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Tab 5: Parameters ---
        with tab5:
            st.subheader("📌 Calibrated Parameters")
            if st.session_state.get("calibration_result"):
                st.json({
                    k: round(v, 6) if isinstance(v, float) else v
                    for k, v in st.session_state["calibration_result"].items()
                })
            else:
                st.warning("Calibration has not been performed yet.")


if __name__ == "__main__":
    run_gui()
