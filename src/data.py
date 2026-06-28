import yfinance as yf
import pandas as pd
import ccxt
import ta
import numpy as np
import requests
import io
import os
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
    """Fetch from Binance Public API with pagination."""
    exchange = ccxt.binance()
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
    
    all_ohlcv = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=current_ts, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        # Move pointer to the last candle + 1 day (roughly)
        current_ts = ohlcv[-1][0] + 86400000 
        if len(ohlcv) < 1000: # No more data
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df[(df.index >= start) & (df.index <= end)]

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
    
    # --- Stationary / scale-invariant transforms ---------------------------
    # Absolute price levels (Close/SMA/EMA/Bollinger/MACD/Volume) are non-
    # stationary: they trend across years and differ across symbols, so a
    # scaler fit on the past maps recent values to extreme z-scores the model
    # never trained on. We convert everything to ratios / relative measures so
    # 2016 and 2026 (and AAPL vs the index) look statistically comparable.
    eps = 1e-9
    ret = df['Close'].pct_change()
    df['Close_sma20_ratio'] = df['Close'] / (df['Sma_20'] + eps) - 1.0
    df['Sma_ratio'] = df['Sma_20'] / (df['Sma_50'] + eps) - 1.0
    df['Ema_ratio'] = df['Ema_12'] / (df['Ema_26'] + eps) - 1.0
    df['Bb_position'] = (df['Close'] - df['Bb_low']) / (df['Bb_high'] - df['Bb_low'] + eps)
    df['Macd_hist_rel'] = (df['Macd'] - df['Macd_signal']) / (df['Close'] + eps)
    df['Atr_rel'] = df['Atr'] / (df['Close'] + eps)
    df['Volume_rel'] = df['Volume'] / (df['Volume'].rolling(20).mean() + eps)
    df['Obv_z'] = (df['Obv'] - df['Obv'].rolling(20).mean()) / (df['Obv'].rolling(20).std() + eps)

    # --- Quantum Features (Zhang & Huang), made scale-invariant via returns ---
    rvol = ret.rolling(window=20).std().clip(lower=1e-3)   # floor avoids 1/0 blow-up
    df['Quantum_mass'] = 1.0 / rvol
    ret3 = df['Close'].pct_change(3)
    df['Quantum_trend'] = df['Quantum_mass'] * ret3
    delta_p = ret.rolling(window=10).std()
    delta_t = df['Quantum_trend'].rolling(window=10).std()
    df['Quantum_uncertainty'] = delta_p * delta_t

    # Fractional Differencing on LOG price (linear in price -> log makes it
    # scale-invariant, unlike differencing the raw price).
    df['Frac_diff_close'] = fractional_diff(np.log(df['Close'].values), d=0.4)

    # Log Returns  (log1p(pct_change) == log(C_t / C_{t-1}))
    df['Log_return'] = np.log1p(ret)

    df = df.bfill().ffill()
    return df

def get_feature_list(df):
    """Canonical (stationary) feature order used for model input.

    Fixed across symbols so one model/scaler works on any ticker. Deliberately
    excludes raw price-level columns and market-context to keep the feature
    count identical for indices and single stocks.
    """
    return [
        'Log_return', 'Rsi', 'Macd_hist_rel',
        'Close_sma20_ratio', 'Sma_ratio', 'Ema_ratio', 'Bb_position',
        'Atr_rel', 'Volume_rel', 'Obv_z', 'Frac_diff_close',
        'Quantum_mass', 'Quantum_trend', 'Quantum_uncertainty',
    ]

def scale_features(scaler, X, clip=5.0):
    """Apply a fitted scaler and clip to +/-`clip` std to tame heavy tails."""
    return np.clip(scaler.transform(X), -clip, clip).astype(np.float32)

def preprocess_data(df):
    """Fit a scaler on this (training) data and normalize it for NN input."""
    features = get_feature_list(df)
    scaler = StandardScaler().fit(df[features].values)
    return scale_features(scaler, df[features].values), features, scaler

def export_processed_data(symbol, start_date, end_date, source="yfinance"):
    """Fetch, process, and save the dataset to a single CSV for portability."""
    print(f"Exporting processed data for {symbol}...")
    df = fetch_data(symbol, start_date, end_date, source=source)
    if df is None or df.empty:
        print("No data found to export.")
        return None
        
    df = add_indicators(df)
    
    # Sanitize filename
    safe_symbol = symbol.replace("/", "_").replace("^", "IDX_")
    os.makedirs("exports", exist_ok=True)
    filename = f"exports/{safe_symbol}_processed_data.csv"
    
    df.to_csv(filename)
    print(f"Successfully saved {len(df)} rows to {filename}")
    return filename

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