import yfinance as yf
import pandas as pd
import numpy as np
import datetime

def get_stock_data(ticker, start="2020-01-01", end=None, filename=None):
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    # Download historical OHLCV data
    stock = yf.download(ticker, start=start, end=end)

    # =============== Fundamentals ===============
    info = yf.Ticker(ticker).info

    fundamentals = {
        "PE_Ratio": info.get("trailingPE"),
        "ROE": info.get("returnOnEquity"),
        "EPS": info.get("trailingEps"),
        "Revenue": info.get("totalRevenue"),
        "NetIncome": info.get("netIncomeToCommon"),
    }

    # Revenue / Net Income ratio
    try:
        fundamentals["Revenue_to_NetIncome"] = fundamentals["Revenue"] / fundamentals["NetIncome"]
    except:
        fundamentals["Revenue_to_NetIncome"] = None

    # Add fundamentals as columns (constant for all rows)
    for k, v in fundamentals.items():
        stock[k] = v

    # =============== Technical Indicators ===============
    # RSI
    delta = stock["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    stock["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = stock["Close"].ewm(span=12, adjust=False).mean()
    ema26 = stock["Close"].ewm(span=26, adjust=False).mean()
    stock["MACD"] = ema12 - ema26
    stock["MACD_Signal"] = stock["MACD"].ewm(span=9, adjust=False).mean()
    stock["MACD_Hist"] = stock["MACD"] - stock["MACD_Signal"]

    # Moving Averages
    stock["SMA_50"] = stock["Close"].rolling(window=50).mean()
    stock["EMA_50"] = stock["Close"].ewm(span=50, adjust=False).mean()

    # Volatility
    stock["StdDev_20"] = stock["Close"].rolling(window=20).std()
    stock["ATR_14"] = (
        (stock["High"] - stock["Low"])
        .combine((stock["High"] - stock["Close"].shift()), np.maximum)
        .combine((stock["Close"].shift() - stock["Low"]), np.maximum)
        .rolling(window=14)
        .mean()
    )

    # Volume already included

    # =============== Calendar Effects ===============
    stock["Month"] = stock.index.month
    stock["Year"] = stock.index.year

    # Quarterly Earnings Dates (upcoming only)
    ticker_obj = yf.Ticker(ticker)
    try:
        earnings = ticker_obj.earnings_dates
        earnings_dates = earnings.index.strftime("%Y-%m-%d").tolist()
        stock["Upcoming_Earnings"] = ", ".join(earnings_dates[:5])  # add first 5 as string
    except:
        stock["Upcoming_Earnings"] = None

    # =============== Save to CSV ===============
    if filename is None:
        filename = f"{ticker}_all_data2.csv"

    stock.to_csv(filename)
    print(f"✅ Saved all data to {filename}")

    return stock

# Example usage
# df = get_stock_data("AAPL", start="2020-01-01")
# print(df.tail())


def get_priority_values(df):
    # Identify ticker (from MultiIndex columns if present)
    if isinstance(df.columns, pd.MultiIndex):
        tickers = [t for t in df.columns.get_level_values(1).unique() if t]
        ticker = tickers[0] if tickers else ''
    else:
        ticker = ''
    
    # Define priority features
    priority_features = [
        ('Close', ticker),
        ('Volume', ticker),
        ('RSI', ''),
        ('MACD', ''),
        ('MACD_Signal', ''),
        ('SMA_50', ''),
        ('ATR_14', ''),
        ('PE_Ratio', ''),
        ('EPS', ''),
        ('Upcoming_Earnings', '')
    ]
    
    last_row = df.iloc[-1]
    values = {}

    for feature, t in priority_features:
        col_name = (feature, t) if isinstance(df.columns, pd.MultiIndex) else feature
        if col_name in df.columns:
            # rename close & volume to generic names
            if feature.lower() == 'close':
                key = 'price'
            elif feature.lower() == 'volume':
                key = 'volume'
            else:
                key = feature.lower()
            values[key] = last_row[col_name]
        else:
            values[feature.lower()] = None  # missing column

    return values
# get_priority_values(df)
# df.head()
# df.columns.tolist()