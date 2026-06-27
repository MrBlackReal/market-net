# Market-Net: Quantum-Enhanced RL Trading Agent

A professional-grade Deep Reinforcement Learning (DRL) research platform that trains autonomous agents to trade assets using a blend of **Predictive Coding**, **Quantum Finance**, and **D3QN Architectures**.

## 🚀 Advanced Features

### 🧠 Neural Architecture
- **Predictive Coding (PC) Hybrid:** Implements a generative predictive head alongside the DQN. The agent learns to minimize "Market Surprise" (prediction error), allowing it to detect regime changes and anomalies faster than standard models.
- **Dueling Double DQN (D3QN):** Splitting State Value and Action Advantage to prevent Q-value overestimation.
- **LSTM Backbone:** Captures long-term temporal dependencies in market data.
- **Optuna Integration:** Built-in Bayesian optimization to find the best `lr`, `gamma`, `hidden_dim`, `batch_size`, `epsilon_decay`, and `pred_alpha`.

### ⚛️ Quantum & Physics Features
- **Quantum Market Model:** Derived from **Zhang & Huang (2010)**, treating price as a wave function:
    - **Quantum Mass:** Proxy for market inertia.
    - **Quantum Trend:** Momentum weighted by mass.
    - **Quantum Uncertainty:** Product of price and trend volatility.
- **Fractional Differencing:** Uses $d=0.4$ to make data stationary while preserving historical memory.
- **Correlation Signals:** Integration of S&P 500 as a global market feature, automatically sourced from the primary data provider.

### 🗄️ Batched Data Caching
- **Persistent Storage:** Automatically saves downloaded market data locally, partitioned by year and symbol.
- **Low-RAM Friendly:** Designed to manage large datasets without loading everything into memory at once.
- **Efficient Updates:** Only fetches missing data, reducing redundant downloads.

### ⚖️ Realistic Environment
- **Transaction Fees:** 0.1% fee per trade.
- **Slippage Simulation:** 0.05% price slippage.
- **Loss-Averse Rewards:** Negative step rewards are amplified by 1.2× to discourage capital erosion.

## 🛠 Setup

1. **Initialize Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install requests
   ```

## 📈 Usage

### 1. Hyperparameter Optimization
Find the best settings for a specific asset and architecture:
```bash
python optuna_search.py --symbol AAPL --source yfinance --model_type pc --trials 30
```

### 2. Auto-Pipeline (Recommended)
Runs Optuna search → full training → backtest in one shot:
```bash
python auto_train.py --symbol AAPL --source yfinance --model_type pc --trials 30 --episodes 200
```

### 3. Training
```bash
# Train using the Predictive Coding Agent on Bitcoin from Binance
python main.py --mode train --symbol "BTC/USDT" --source binance --model_type pc --episodes 100
```

Data sources: `yfinance`, `binance`, `stooq`.

### 4. Monitoring
Track Reward, NetWorth, DQN Loss, Epsilon, and **Predictive Surprise (PredLoss)**:
```bash
tensorboard --logdir runs
```

### 5. Backtesting
```bash
python main.py --mode test --symbol AAPL --model_type pc --model models/model_AAPL_best.pth
```

### 6. Export Processed Data
```bash
python main.py --mode export --symbol AAPL --source yfinance
```

## 📊 Model Outputs

| Name | Description |
|------|-------------|
| `models/model_*.pth` | Trained PyTorch weights for the agent. |
| `backtest_*.png` | Visual comparison of Agent vs. Buy & Hold strategy. |
| `exports/` | Processed CSV exports from `--mode export`. |
| `runs/` | TensorBoard event files for real-time training analytics. |
| `data_cache/` | Local batched storage of market data partitioned by year. |

## 🧪 Testing
The project follows modular design patterns. You can run verification scripts or add unit tests for indicators:
```bash
# Verify data pipeline and caching
python src/data.py
```

## 📂 Project Structure
- `main.py`: CLI entry point for training, testing, and data export.
- `auto_train.py`: Full 3-phase pipeline — Optuna search → training → backtest.
- `optuna_search.py`: Modular hyperparameter optimization.
- `src/train.py`: Core training loop and backtest logic.
- `src/data.py`: Feature engineering and unified data pipeline.
- `src/dataset.py`: Batched caching and low-RAM dataset management.
- `src/env.py`: Realistic trading environment.
- `src/model_pc.py`: Predictive Coding Hybrid Agent.
- `src/model.py`: Standard D3QN Agent.
- `paper.txt`: Theoretical context on Predictive Coding.
