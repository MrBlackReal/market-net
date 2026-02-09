import yfinance as yf
import pandas as pd
import ccxt
import ta
import numpy as np
import requests
import io
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from src.dataset import MarketDataset

# Global dataset manager
cache = MarketDataset()

def fetch_from_yfinance(symbol, start, end):
    """Fetch from Yahoo Finance (No login required)."""
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def fetch_from_stooq(symbol, start, end):
    """Fetch from Stooq manually (No login required)."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&f=sd2ohlcv&h&k1=off"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    df = pd.read_csv(io.StringIO(response.text))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[(df.index >= start) & (df.index <= end)]
    return df

def fetch_from_binance(symbol, start, end):
    """Fetch from Binance Public API (No login/key required)."""
    exchange = ccxt.binance()
    since = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

def fetch_data(symbol, start_date, end_date, source="yfinance", include_market=True):
    """
    Unified entry point for fetching data. Checks cache first.
    """
    # 1. Try loading from local batched cache
    df = cache.load_full_range(symbol, start_date, end_date, source)
    if df is not None:
        print(f"Loaded {symbol} from local cache.")
    else:
        # 2. Fetch from internet if cache is missing
        print(f"Fetching {symbol} from {source}...")
        try:
            if source == "yfinance":
                df = fetch_from_yfinance(symbol, start_date, end_date)
            elif source == "stooq":
                df = fetch_from_stooq(symbol, start_date, end_date)
            elif source == "binance":
                df = fetch_from_binance(symbol, start_date, end_date)
            
            # Save new data to cache (partitioned by year)
            if df is not None and not df.empty:
                cache.save_batch(df, symbol, source)
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    # 3. Add Market Correlation
    # Note: Market data (^GSPC) is typically sourced from yfinance or stooq.
    # If using binance, we still fallback to yfinance for the index.
    if include_market and symbol not in ["^GSPC", "^SPX"]:
        market_source = "stooq" if source == "stooq" else "yfinance"
        market_symbol = "^SPX" if market_source == "stooq" else "^GSPC"
        
        market_df = fetch_data(market_symbol, start_date, end_date, source=market_source, include_market=False)
        if market_df is not None:
            market_df = market_df[['Close']].rename(columns={'Close': 'Market_close'})
            df = df.join(market_df, how='left').bfill().ffill()
    
    return df

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

def add_indicators(df):
    """Add technical indicators and Quantum features from Zhang & Huang paper."""
    if df is None or df.empty:
        return None
        
    df = df.copy()
    # Standardize column names to Capitalized
    df.columns = [c.capitalize() for c in df.columns]
    
    # --- Standard Indicators ---
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
    
    # --- Quantum Features (Zhang & Huang) ---
    vol = df['Close'].rolling(window=20).std()
    df['Quantum_mass'] = 1.0 / (vol + 1e-9)
    price_diff = df['Close'].diff(3)
    df['Quantum_trend'] = df['Quantum_mass'] * price_diff
    delta_p = df['Close'].rolling(window=10).std()
    delta_t = df['Quantum_trend'].rolling(window=10).std()
    df['Quantum_uncertainty'] = delta_p * delta_t
    
    # Fractional Differencing
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
        'Log_return', 'Frac_diff_close',
        'Quantum_mass', 'Quantum_trend', 'Quantum_uncertainty'
    ]
    if 'Market_return' in df.columns:
        features.append('Market_return')
    
    data_to_scale = df[features].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    
    return scaled_data, features, scaler

if __name__ == "__main__":
    # Test caching and partitioning
    test_symbol = "AAPL"
    d1 = fetch_data(test_symbol, "2020-01-01", "2023-12-31", source="yfinance")
    print(f"Loaded {test_symbol} shape: {d1.shape}")
    
    # Check if files were created
    import os
    safe_symbol = test_symbol.replace("/", "_").replace("^", "IDX_")
    cache_path = os.path.join("data_cache", safe_symbol, "yfinance")
    if os.path.exists(cache_path):
        print(f"Cache files created in {cache_path}: {os.listdir(cache_path)}")