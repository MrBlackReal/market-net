import yfinance as yf
import pandas as pd
import ccxt
import ta
import numpy as np
import requests
import io
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def fetch_from_yfinance(symbol, start, end):
    """Fetch from Yahoo Finance (No login required)."""
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def fetch_from_stooq(symbol, start, end):
    """Fetch from Stooq manually (No login required, no broken lib)."""
    # symbol example: AAPL.US, ^SPX
    url = f"https://stooq.com/q/d/l/?s={symbol}&f=sd2ohlcv&h&k1=off"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    df = pd.read_csv(io.StringIO(response.text))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Filter by date
    df = df[(df.index >= start) & (df.index <= end)]
    return df

def fetch_from_binance(symbol, start, end):
    """Fetch from Binance Public API (No login/key required)."""
    # symbol example: 'BTC/USDT'
    exchange = ccxt.binance()
    since = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    # Note: fetch_ohlcv has limits, this is a simplified version
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

from datetime import datetime, timedelta

def fractional_diff(series, d, threshold=0.01):
    """
    Apply Fractional Differencing to make data stationary while preserving memory.
    d: differencing order (0 to 1)
    """
    weights = [1.0]
    for k in range(1, 100):
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
        if abs(w) < threshold: break
    
    weights = np.array(weights[::-1]).reshape(-1, 1)
    res = []
    for i in range(len(weights), len(series) + 1):
        res.append(np.dot(series[i-len(weights):i], weights))
    
    # Pad with NaNs to keep original length
    result = np.array(res).flatten()
    padding = np.full(len(series) - len(result), np.nan)
    return np.concatenate([padding, result])

def fetch_data(symbol, start_date, end_date, source="yfinance", include_market=True):
    """
    Unified entry point for fetching data.
    If include_market is True, it also fetches S&P 500 as a correlation feature.
    """
    print(f"Fetching {symbol} from {source}...")
    try:
        if source == "yfinance":
            df = fetch_from_yfinance(symbol, start_date, end_date)
        elif source == "stooq":
            df = fetch_from_stooq(symbol, start_date, end_date)
        elif source == "binance":
            df = fetch_from_binance(symbol, start_date, end_date)
        
        if include_market and symbol != "^SPX":
            print("Fetching S&P 500 for correlation...")
            market_df = fetch_from_yfinance("^GSPC", start_date, end_date)
            if market_df is not None:
                # Rename market columns to avoid collision
                market_df = market_df[['Close']].rename(columns={'Close': 'Market_Close'})
                df = df.join(market_df, how='left').bfill().ffill()
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def add_indicators(df):
    """Add technical indicators and fractional differencing."""
    if df is None or df.empty:
        return None
        
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    
    # Standard Indicators
    df['Sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['Sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['Ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['Ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['Rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['Macd'] = ta.trend.macd(df['Close'])
    df['Macd_signal'] = ta.trend.macd_signal(df['Close'])
    df['Bb_high'] = ta.volatility.bollinger_hband(df['Close'])
    df['Bb_low'] = ta.volatility.bollinger_lband(df['Close'])
    df['Atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['Obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    # Fractional Differencing (d=0.4 is common for price data)
    df['Frac_diff_close'] = fractional_diff(df['Close'].values, d=0.4)
    
    # Log Returns
    df['Log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    if 'Market_close' in df.columns:
        df['Market_return'] = np.log(df['Market_close'] / df['Market_close'].shift(1))
    
    df = df.bfill().ffill()
    return df

def preprocess_data(df):
    """Normalize indicators for neural network input."""
    features = [
        'Close', 'Volume', 'Sma_20', 'Sma_50', 'Ema_12', 'Ema_26', 
        'Rsi', 'Macd', 'Macd_signal', 'Bb_high', 'Bb_low', 'Atr', 'Obv', 
        'Log_return', 'Frac_diff_close'
    ]
    if 'Market_return' in df.columns:
        features.append('Market_return')
    
    data_to_scale = df[features].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    
    return scaled_data, features, scaler

if __name__ == "__main__":
    # Test multiple sources
    # 1. YFinance
    d1 = fetch_data("AAPL", "2023-01-01", "2023-12-31", source="yfinance")
    print(f"YFinance shape: {d1.shape if d1 is not None else 'Failed'}")
    
    # 2. Stooq
    d2 = fetch_data("AAPL.US", "2023-01-01", "2023-12-31", source="stooq")
    print(f"Stooq shape: {d2.shape if d2 is not None else 'Failed'}")
    
    # 3. Binance
    d3 = fetch_data("BTC/USDT", "2023-01-01", "2023-12-31", source="binance")
    print(f"Binance shape: {d3.shape if d3 is not None else 'Failed'}")