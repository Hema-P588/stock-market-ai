import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Load model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.title("📈 Share Market Prediction AI")

symbol = st.text_input("Enter Stock Symbol (Example: TCS.NS)", "TCS.NS")

if st.button("Predict"):
    df = yf.download(symbol, period="60d", progress=False)

    if len(df) < 30:
        st.error("Not enough data")
    else:
        close = df['Close']
        df = yf.download(symbol, period="60d", progress=False)

        close = df['Close'].squeeze()   # 🔥 FIX
        # SAME FEATURE ENGINEERING AS TRAINING
        df['Return'] = close.pct_change()
        df['MA_10'] = close.rolling(10).mean()
        df['MA_20'] = close.rolling(20).mean()
        df['RSI'] = RSIIndicator(close).rsi()
        df['MACD'] = MACD(close).macd()

        df.dropna(inplace=True)

        latest = df.iloc[-1][features].values.reshape(1, -1)

        prediction = model.predict(latest)[0]
        confidence = model.predict_proba(latest)[0][1]

        if prediction == 1:
            st.success(f"✅ BUY Signal (Confidence: {confidence*100:.2f}%)")
        else:
            st.warning(f"❌ NO BUY Signal (Confidence: {confidence*100:.2f}%)")


