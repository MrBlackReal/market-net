# Market-Net: Quantum-Enhanced RL Trading Agent

A professional-grade Deep Reinforcement Learning (DRL) research platform that trains autonomous agents to trade assets using a blend of **Quantum Finance**, **Physics-inspired Feature Engineering**, and **Advanced RL Architectures**.

## 🚀 Advanced Features

### 🧠 Neural Architecture
- **Dueling Double DQN (D3QN):** Uses a Dueling architecture (splitting State Value and Action Advantage) combined with Double DQN logic to prevent Q-value overestimation and stabilize learning.
- **LSTM Backbone:** Captures long-term temporal dependencies in market data, allowing the agent to "remember" market cycles.
- **Optuna Integration:** Built-in Bayesian optimization for hyperparameter tuning.

### ⚛️ Quantum & Physics Features
- **Quantum Market Model:** Implements features derived from the **Zhang & Huang (2010)** paper, treating stock price as a wave function:
    - **Quantum Mass:** Proxy for market inertia/market cap.
    - **Quantum Trend:** Momentum weighted by inertia.
    - **Quantum Uncertainty:** Real-time product of price and trend volatility to detect equilibrium shifts.
- **Fractional Differencing:** Uses non-integer differencing (e.g., $d=0.4$) to make data stationary while preserving maximum historical memory.
- **Correlation Signals:** Automatically fetches and correlates S&P 500 data for all stock symbols.

### ⚖️ Realistic Environment
- **Transaction Fees:** 0.1% fee per trade to penalize over-trading.
- **Slippage Simulation:** 0.05% price slippage to model liquidity constraints.
- **Risk-Adjusted Rewards:** Reward function penalizes drawdowns to encourage capital preservation.

## 🛠 Setup

1. **Initialize Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install ccxt ta requests tensorboard optuna
   ```

## 📈 Usage

### 1. Hyperparameter Optimization
Find the best settings for your specific asset before training:
```bash
python optuna_search.py
```

### 2. Training
Train the agent with a 10-year historical window and real-time monitoring.
```bash
# Train on Apple (Default: yfinance)
python main.py --mode train --symbol AAPL --episodes 100

# Train on Bitcoin (Binance source)
python main.py --mode train --symbol "BTC/USDT" --source binance --episodes 200
```

### 3. Monitoring
Track rewards, loss, and agent "brain" metrics in real-time:
```bash
tensorboard --logdir runs
```

### 4. Backtesting
Evaluate performance on the most recent 2-year window (out-of-sample).
```bash
python main.py --mode test --symbol AAPL --model model_AAPL_best.pth
```

## 📂 Project Structure
- `main.py`: Entry point for Train/Test modes.
- `optuna_search.py`: Hyperparameter search script.
- `src/data.py`: Advanced feature engineering (Quantum, FracDiff, Indicators).
- `src/env.py`: Realistic trading environment with fees/slippage.
- `src/model.py`: Dueling LSTM D3QN Agent.
- `src/train.py`: Training loop with Val-split, TensorBoard, and Early Stopping.