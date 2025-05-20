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

# Strategy pricers
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

    # --- Input Section ---
    ticker = st.text_input("Enter Ticker", value="AAPL")
    force_download = st.checkbox("Force Download Yield Curve", value=False)
    min_volume = st.number_input("Minimum Volume Filter", min_value=0, value=50, step=10)
    if "spot" in st.session_state:
        st.markdown(f"### 📍 Current Spot Price: **${st.session_state['spot']:.2f}**")

    if st.button("Get Data & Calibrate"):
        try:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            maturities, yields = get_latest_yield_curve(force_download=force_download)
            curve_fit, _ = calibrate_nss_ols(maturities, yields)

            spot, hist, df, n = get_option_data(ticker, curve_fit, output_dir, min_volume=min_volume)

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

    # --- GUI Tabs ---
    if all(key in st.session_state for key in ["vol_df", "spot", "hist"]):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧮 Option Pricer",
            "📄 Option Data",
            "📉 Spot Price",
            "📊 3D Surface",
            "📌 Calibrated Parameters"
        ])

        with tab1:
            st.subheader("🧮 Option Pricing")

            if st.session_state.get("calibration_result") is None:
                st.warning("Please calibrate the model first.")
            else:
                df = st.session_state["vol_df"]
                spot = st.session_state["spot"]

                # Show expiries (dates) instead of ttm
                maturity_dates = sorted(df["maturity"].unique())
                selected_expiry = st.selectbox("Select Maturity (Expiry Date)", maturity_dates)

                today = datetime.today()
                selected_maturity = (datetime.strptime(selected_expiry, "%Y-%m-%d") - today).days / 365.25

                filtered_df = df[df["maturity"] == selected_expiry]
                strike_range = filtered_df["strike"].min(), filtered_df["strike"].max()
                selected_strike = st.number_input("Select Strike",
                                                  max_value=float(strike_range[1]), value=spot)

                option_type = st.selectbox("Option Type", ["call", "put"], key="option_type_pricer")
                strategy = st.selectbox("Strategy Type", [
                    "Vanilla", "Digital", "Bull Call Spread", "Bear Put Spread", "Straddle", "Strangle"
                ])

                curve_fit = st.session_state["curve_fit"]
                rate = curve_fit(selected_maturity)
                v0 = st.session_state["calibration_result"]["v0"]
                kappa = st.session_state["calibration_result"]["kappa"]
                theta = st.session_state["calibration_result"]["theta"]
                sigma = st.session_state["calibration_result"]["sigma"]
                rho = st.session_state["calibration_result"]["rho"]

                M = 10000
                N = 250

                if strategy == "Vanilla":
                    args = (selected_strike, rate, selected_maturity, option_type)
                    price = vanilla_option_price_heston(heston_model_sim(spot, v0, rho, kappa, theta, sigma, selected_maturity, N, M, rate)[0], *args)

                elif strategy == "Digital":
                    payout = st.number_input("Digital Payout", value=1.0)
                    args = (selected_strike, rate, selected_maturity, option_type, payout)
                    price = digital_option_price_heston(heston_model_sim(spot, v0, rho, kappa, theta, sigma, selected_maturity, N, M, rate)[0], *args)

                elif strategy == "Bull Call Spread":
                    K1 = st.number_input("Lower Strike (Long Call)", value=selected_strike - 5.0)
                    K2 = st.number_input("Upper Strike (Short Call)", value=selected_strike + 5.0)
                    args = (K1, K2, rate, selected_maturity)
                    price = bull_call_spread_price_heston(heston_model_sim(spot, v0, rho, kappa, theta, sigma, selected_maturity, N, M, rate)[0], *args)

                elif strategy == "Bear Put Spread":
                    K1 = st.number_input("Lower Strike (Short Put)", value=selected_strike - 5.0)
                    K2 = st.number_input("Upper Strike (Long Put)", value=selected_strike + 5.0)
                    args = (K1, K2, rate, selected_maturity)
                    price = bear_put_spread_price_heston(heston_model_sim(spot, v0, rho, kappa, theta, sigma, selected_maturity, N, M, rate)[0], *args)

                elif strategy == "Straddle":
                    args = (selected_strike, rate, selected_maturity)
                    price = straddle_price_heston(
                        heston_model_sim(spot, v0, rho, kappa, theta, sigma, selected_maturity, N, M, rate)[0],
                        *args
                    )

                elif strategy == "Strangle":
                    K1 = st.number_input("Lower Strike (Put)", value=selected_strike - 10.0)
                    K2 = st.number_input("Upper Strike (Call)", value=selected_strike + 10.0)
                    args = (K1, K2, rate, selected_maturity)
                    price = strangle_price_heston(heston_model_sim(spot, v0, rho, kappa, theta, sigma, selected_maturity, N, M, rate)[0], *args)

                st.metric(label=f"{strategy} Strategy Price", value=f"${price:.4f}")

        with tab2:
            st.subheader("📄 Market Option Surface")
            st.dataframe(st.session_state["vol_df"])

        with tab3:
            st.subheader("📉 Historical Spot Price")
            st.line_chart(st.session_state["hist"]["Close"])

        with tab4:
            st.subheader("📊 Market Option Price Surface (3D)")
            df = st.session_state["vol_df"]
            selected_type = st.selectbox("Option Type", options=["call", "put"], key="option_type_surface")
            df_filtered = df[df["type"] == selected_type]

            if df_filtered.empty:
                st.warning(f"No {selected_type} options available.")
            else:
                x, y, z = df_filtered["ttm"], df_filtered["strike"], df_filtered["price"]
                xi = np.linspace(x.min(), x.max(), 50)
                yi = np.linspace(y.min(), y.max(), 50)
                X, Y = np.meshgrid(yi, xi)
                Z = griddata((x, y), z, (X, Y), method='cubic')

                fig = go.Figure()
                fig.add_trace(go.Surface(x=Y, y=X, z=Z, colorscale='Viridis', name='Surface'))
                fig.add_trace(go.Scatter3d(x=y, y=x, z=z, mode='markers',
                                           marker=dict(size=4, color='black'), name="Market"))

                fig.update_layout(
                    scene=dict(xaxis_title='Strike', yaxis_title='TTM', zaxis_title='Price'),
                    height=750, margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

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
