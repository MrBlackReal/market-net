# Market-Net: Quantum-Enhanced RL Trading Agent

A professional-grade Deep Reinforcement Learning (DRL) research platform that trains autonomous agents to trade assets using a blend of **Predictive Coding**, **Quantum Finance**, and **D3QN Architectures**.

## 🚀 Advanced Features

### 🧠 Neural Architecture
- **Predictive Coding (PC) Hybrid:** (New) Implements a generative predictive head alongside the DQN. The agent learns to minimize "Market Surprise" (prediction error), allowing it to detect regime changes and anomalies faster than standard models.
- **Dueling Double DQN (D3QN):** Splitting State Value and Action Advantage to prevent Q-value overestimation.
- **LSTM Backbone:** Captures long-term temporal dependencies in market data.
- **Optuna Integration:** Built-in Bayesian optimization to find the best `lr`, `gamma`, and `model_type`.

### ⚛️ Quantum & Physics Features
- **Quantum Market Model:** Derived from **Zhang & Huang (2010)**, treating price as a wave function:
    - **Quantum Mass:** Proxy for market inertia.
    - **Quantum Trend:** Momentum weighted by mass.
    - **Quantum Uncertainty:** Product of price and trend volatility.
- **Fractional Differencing:** Uses $d=0.4$ to make data stationary while preserving historical memory.
- **Correlation Signals:** Integration of S&P 500 (`^GSPC`) as a global market feature.

### ⚖️ Realistic Environment
- **Transaction Fees:** 0.1% fee per trade.
- **Slippage Simulation:** 0.05% price slippage.
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
Evaluate both **Standard** and **PC** models across 30+ trials:
```bash
python optuna_search.py
```

### 2. Training
```bash
# Train using the Predictive Coding Agent
python main.py --mode train --symbol AAPL --model_type pc --episodes 100
```

### 3. Monitoring
Track Reward, DQN Loss, and **Predictive Surprise (PredLoss)**:
```bash
tensorboard --logdir runs
```

### 4. Backtesting
```bash
python main.py --mode test --symbol AAPL --model_type pc --model model_AAPL_best.pth
```

## 📊 Model Outputs

| Name | Description |
|------|-------------|
| `model_*.pth` | Trained PyTorch weights for the agent. |
| `backtest_*.png` | Visual comparison of Agent vs. Buy & Hold strategy. |
| `runs/` | TensorBoard event files for real-time training analytics. |
| `best_params` | Optimal hyperparameters found via Optuna (printed to console). |

## 🧪 Testing
The project follows modular design patterns. You can run verification scripts or add unit tests for indicators:
```bash
# Verify data pipeline
python src/data.py
```

## 📂 Project Structure
- `main.py`: CLI entry point.
- `optuna_search.py`: Hyperparameter optimization.
- `src/data.py`: Feature engineering (Quantum, FracDiff, Indicators).
- `src/env.py`: Realistic trading environment.
- `src/model_pc.py`: Predictive Coding Hybrid Agent.
- `src/model.py`: Standard D3QN Agent.
- `paper.txt`: Theoretical context on Predictive Coding.
