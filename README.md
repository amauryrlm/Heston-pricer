
# Heston Option Pricer

An interactive option pricing and calibration interface using Streamlit. This project allows you to fetch option data, calibrate the Heston model, and visualize pricing strategies.

---

## Clone or Download

You can either **clone** the repository or **download it manually**:

### Option 1: Clone via Git

```bash
git clone https://github.com/amauryrlm/Heston-pricer.git
cd Heston-pricer
```

### Option 2: Download via Browser

1. Go to [https://github.com/amauryrlm/Heston-pricer](https://github.com/amauryrlm/Heston-pricer)
2. Click on the green **Code** button
3. Select **Download ZIP**
4. Extract the ZIP and open a terminal into the folder

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv

venv\Scripts\activate    # On Windows
# or
source venv/bin/activate   # On macOS/Linux
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

---

## Usage

To launch the Streamlit GUI:

```bash

python -m streamlit run src/gui.py
```

This will start an interactive interface where you can:

- Input a ticker symbol (e.g., AAPL)  
- Download and filter option chains  
- Calibrate the Heston model to market prices  
- Price various option strategies  
- Visualize the price surface and historical spot data  

---

## Ressources 

- https://www.youtube.com/watch?v=Jy4_AVEyO0w
- https://www.youtube.com/watch?v=o8C6DxZh8dw&t=423s
---

## Requirements

- Python 3.11 (recommended)  
- Streamlit  
- NumPy  
- SciPy  
- pandas  
- yfinance  
- plotly  
- numba  
- nelson-siegel-svensson  

---

## License

This project is licensed under the MIT License.

---

## Author

Amaury Rodriguez-Le Mazou  
Marc Hayek
