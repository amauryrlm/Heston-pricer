import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def get_latest_yield_curve(month: str = None, force_download: bool = False):
    """
    Retrieves and caches the latest US Treasury yield curve.
    If 'output/yield_curve.csv' was modified within the last day and force_download=False,
    loads from file instead. Otherwise, fetches from the internet.

    Parameters:
        month (str, optional): Date in 'YYYYMM' format. If None, uses current month.
        force_download (bool): If True, always fetch from the internet.

    Returns:
        tuple: (maturities: np.ndarray, yields: np.ndarray)
    """
    path = "output/yield_curve.csv"
    if not force_download and os.path.exists(path):
        last_modified = datetime.fromtimestamp(os.path.getmtime(path))
        if datetime.now() - last_modified < timedelta(days=1):
            print("✅ Using cached yield curve from file")
            df = pd.read_csv(path)
            return df["maturity"].to_numpy(), df["yield"].to_numpy()

    # No recent cache found or force_download is True — fetch from the internet
    if month is None:
        month = datetime.today().strftime("%Y%m")

    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"pages/xmlview?data=daily_treasury_yield_curve&field_tdr_date_value_month={month}"
    )

    response = requests.get(url)
    root = ET.fromstring(response.content)

    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'm': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata',
        'd': 'http://schemas.microsoft.com/ado/2007/08/dataservices'
    }

    labels_years = [
        ("BC_1MONTH", 1 / 12),
        ("BC_2MONTH", 2 / 12),
        ("BC_3MONTH", 3 / 12),
        ("BC_6MONTH", 6 / 12),
        ("BC_1YEAR", 1),
        ("BC_2YEAR", 2),
        ("BC_3YEAR", 3),
        ("BC_5YEAR", 5),
        ("BC_7YEAR", 7),
        ("BC_10YEAR", 10),
        ("BC_20YEAR", 20),
        ("BC_30YEAR", 30),
    ]

    for entry in reversed(root.findall('atom:entry', ns)):
        props = entry.find('atom:content/m:properties', ns)
        try:
            yields = []
            maturities = []
            for label, year in labels_years:
                node = props.find(f'd:{label}', ns)
                if node is not None and node.text is not None:
                    yields.append(float(node.text) / 100)
                    maturities.append(year)

            if len(yields) == len(labels_years):
                print("✅ OK - Yield curve retrieved successfully")

                # Save to output
                df = pd.DataFrame({
                    "maturity": maturities,
                    "yield": yields
                })
                os.makedirs("output", exist_ok=True)
                df.to_csv(path, index=False)
                print(f"📁 Yield curve saved to: {path}")

                return np.array(maturities), np.array(yields)
        except Exception as e:
            print(f"⚠️ Error parsing entry: {e}")
            continue

    raise ValueError(f"No valid yield curve data found for month {month}.")
