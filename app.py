import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="NSE  App", layout="wide")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.title("📈 Stock Forecasting App")

start = st.sidebar.date_input("Start Date", dt.date(2024, 1, 1))
end = st.sidebar.date_input("End Date", dt.date(2025, 11, 14))

stocks = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "L&T": "LT.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "LIC": "LICI.NS",
    "IOC": "IOC.NS"
}

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
symbol = stocks[selected_stock]

# -----------------------------
# Download Data
# -----------------------------
@st.cache_data
def load_data():
    df_list = []
    for name, ticker in stocks.items():
        data = yf.download(ticker, start=start, end=end)
        if not data.empty:
            data.columns = data.columns.get_level_values(0)
            data = data.reset_index()
            data["Stock"] = name
            df_list.append(data)
    return pd.concat(df_list, axis=0)

df = load_data()

st.title("📊 NSE Stock Dashboard")
st.write("View stock analysis, stationarity check, ARIMA forecasting, and creative charts.")

s = df[df["Stock"] == selected_stock].copy()
s.set_index("Date", inplace=True)

# -----------------------------
# Create Returns Column
# -----------------------------
s["Return"] = s["Close"].pct_change()

# -----------------------------
# Candlestick Chart
# -----------------------------
st.subheader(f"📌 Candlestick Chart – {selected_stock}")

fig = go.Figure(data=[
    go.Candlestick(
        x=s.index,
        open=s["Open"],
        high=s["High"],
        low=s["Low"],
        close=s["Close"]
    )
])
fig.update_layout(height=500, width=900)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Line Chart for Close Price
# -----------------------------
st.subheader("📈 Closing Price Trend")
st.line_chart(s["Close"])

# -----------------------------
# Return Plot
# -----------------------------
st.subheader("📉 Daily Returns (%)")
st.line_chart(s["Return"] * 100)

# -----------------------------
# Stationarity Test
# -----------------------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] < 0.05, result[1]

is_stat, p_value = check_stationarity(s["Close"])

st.subheader("🧪 Stationarity Test (ADF)")

if is_stat:
    st.success(f"The series is **stationary** (p-value = {p_value:.5f})")
else:
    st.warning(f"The series is **NOT stationary** (p-value = {p_value:.5f})")

# Differenced series plot
s["Close_diff"] = s["Close"].diff()

st.subheader("📉 Differenced Close Price (1st Order)")
st.line_chart(s["Close_diff"])

# -----------------------------
# ARIMA Forecasting
# -----------------------------
st.subheader("🔮 ARIMA Forecast (Next 10 Days)")

model = ARIMA(s["Close"], order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=10)
future_dates = pd.date_range(start=s.index[-1], periods=11, freq="B")[1:]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=s.index, y=s["Close"], mode="lines", name="Actual"))
fig2.add_trace(go.Scatter(x=future_dates, y=forecast, mode="lines", name="Forecast", line=dict(color="red", dash="dash")))
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Correlation Heatmap of Returns (Creative Graph)
# -----------------------------
st.subheader("🔥 Correlation Heatmap (Returns of All Stocks)")

pivot_returns = df.pivot(index="Date", columns="Stock", values="Close").pct_change()
corr = pivot_returns.corr()

fig3, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig3)

# -----------------------------
# Completion Message
# -----------------------------
st.success("Dashboard Loaded Successfully!")
