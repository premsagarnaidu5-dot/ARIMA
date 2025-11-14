import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import datetime as dt

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

# --------------------
# Utility Functions
# --------------------

def check_stationarity(series):
    """Return ADF p-value and stationarity result."""
    result = adfuller(series.dropna())
    p_value = result[1]
    if p_value < 0.05:
        return p_value, "Stationary"
    else:
        return p_value, "Not Stationary"

def load_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if len(df) == 0:
        return None
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.set_index("Date")
    return df


# --------------------
# Streamlit UI
# --------------------

st.title("📈 Stock Price Forecasting Dashboard")
st.write("Predict stock prices using **ARIMA Time Series Model**")

# Stock selection
stock = st.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS",
     "BHARTIARTL.NS", "LT.NS", "ICICIBANK.NS", "LICI.NS", "IOC.NS"]
)

start_date = st.date_input("Start Date", dt.date(2024,1,1))
end_date = st.date_input("End Date", dt.date.today())

if st.button("Run Forecast"):
    st.subheader(f"📥 Downloading Data for {stock} ...")
    
    df = load_stock_data(stock, start_date, end_date)

    if df is None:
        st.error("Failed to download data. Check stock symbol or date range.")
    else:
        st.success("Data Loaded Successfully!")

        # Keep only Close prices
        df = df[['Close']]

        # Display data
        st.dataframe(df.tail())

        # --------------------
        # Stationarity Test
        # --------------------

        p_value, result = check_stationarity(df['Close'])

        st.subheader("📊 Stationarity Test (ADF)")
        st.write(f"**ADF p-value:** {p_value:.4f}")
        st.write(f"**Conclusion:** {result}")

        # Differencing for stationarity
        df['Close_diff'] = df['Close'].diff()
        df.dropna(inplace=True)

        # --------------------
        # ARIMA Model
        # --------------------

        st.subheader("🔮 ARIMA Model Forecast")

        with st.spinner("Training ARIMA model..."):
            model = ARIMA(df['Close'], order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)

        st.success("Forecast Generated!")

        # Forecast date range
        dates = pd.date_range(start=df.index[-1], periods=11, freq='B')[1:]

        # --------------------
        # Plotting
        # --------------------

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['Close'], label="Actual Price")
        ax.plot(dates, forecast, label="Predicted Price", linestyle="dashed", color="red")
        ax.set_title(f"{stock} Price Forecast")
        ax.legend()

        st.pyplot(fig)

        # --------------------
        # Show forecast table
        # --------------------

        forecast_df = pd.DataFrame({"Date": dates, "Forecast": forecast})
        forecast_df.set_index("Date", inplace=True)

        st.subheader("📅 Forecasted Values (Next 10 Days)")
        st.dataframe(forecast_df)

