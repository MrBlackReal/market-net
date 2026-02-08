# Market-Net: Neural Network Trading Agent

A Deep Reinforcement Learning (DQN) agent that learns to trade assets (Buy/Sell/Hold) using historical market data and technical indicators.

## Features
- **RL Agent:** Uses an LSTM-based Deep Q-Network to handle temporal market data.
- **Custom Environment:** A `gymnasium` environment that simulates trading logic, fees, and portfolio management.
- **Multiple Data Sources (Anonymous):**
  - **Yahoo Finance**: Global stocks and crypto.
  - **Binance**: Direct crypto data (via CCXT).
  - **Stooq**: Global indices and stock data.
- **Technical Indicators:** RSI, MACD, Bollinger Bands, SMA, EMA, ATR, and OBV.

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install ccxt ta requests
   ```

## Usage

The project is controlled via `main.py`.

### 1. Training the Agent
Train the agent on a specific symbol and data source.

```bash
# Train on Apple using Yahoo Finance
python main.py --mode train --symbol AAPL --source yfinance --episodes 50

# Train on Bitcoin using Binance
python main.py --mode train --symbol "BTC/USDT" --source binance --episodes 100

# Train on S&P 500 using Stooq
python main.py --mode train --symbol "^SPX" --source stooq --episodes 50
```

### 2. Backtesting
Evaluate a trained model on unseen data. This will generate a `backtest_result.png`.

```bash
python main.py --mode test --symbol AAPL --model model_AAPL.pth
```

## Configuration
- **State Space:** Includes OHLCV data, 11 technical indicators, and portfolio status (balance, shares held).
- **Action Space:** 0 (Hold), 1 (Buy), 2 (Sell).
- **Hyperparameters:** Can be adjusted in `src/model.py` and `src/train.py`.

## Project Structure
- `main.py`: CLI entry point.
- `src/data.py`: Data fetching and indicator logic.
- `src/env.py`: Trading environment.
- `src/model.py`: DQN Agent & LSTM architecture.
- `src/train.py`: Training loop.
