import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

companies = {
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS"
}

all_data = []

for name, ticker in companies.items():
    df = yf.download(ticker, start="2018-01-01", progress=False)

    # 🔑 FORCE Close to be 1D Series
    close = df['Close'].squeeze()

    df = df.copy()
    df['Company'] = name

    # Feature engineering
    df['Return'] = close.pct_change()
    df['MA_10'] = close.rolling(10).mean()
    df['MA_20'] = close.rolling(20).mean()

    df['RSI'] = RSIIndicator(close).rsi()
    df['MACD'] = MACD(close).macd()

    # Label creation
    future_close = close.shift(-5)
    df['Buy'] = ((future_close - close) / close > 0.03).astype(int)

    df.dropna(inplace=True)
    all_data.append(df)

# Combine all companies
final_df = pd.concat(all_data)

X = final_df[['Return', 'MA_10', 'MA_20', 'RSI', 'MACD']]
y = final_df['Buy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(X.columns.tolist(), "features.pkl")

joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully")


