# Market-Net

A research platform for systematic trading on equities and crypto. It contains
two complementary approaches:

1. **A reinforcement-learning agent** (Dueling Double-DQN with an LSTM, plus an
   optional Predictive-Coding variant) that learns a buy/hold/sell policy.
2. **Transparent rules-based strategies** (diversification + cross-sectional
   momentum) that, in honest backtests, have a far more reliable edge than the
   RL agent.

> **Reality check.** On out-of-sample data the RL agent trades at roughly chance
> versus buy-and-hold. The systematic momentum/diversification strategy is the
> one with a documented, validated edge. See **[Performance](#-honest-performance)**
> before trusting any of these numbers. Nothing here is financial advice.

---

## ⚡ Setup

CPU-only; no GPU required (oneDNN/MKLDNN acceleration is enabled automatically).

```bash
python3 -m venv venv
source venv/bin/activate        # bash/zsh   (fish: source venv/bin/activate.fish)
pip install -r requirements.txt
```

All commands below assume the venv's Python (`./venv/bin/python` or an activated venv).

---

## 📈 Usage

### Systematic strategy (the part with a real edge)

```bash
# What to hold today: top momentum names, equal-weighted
python main.py --mode allocate --capital 10000

# Backtest momentum vs equal-weight vs S&P 500 (writes strategy_comparison.png)
python main.py --mode strategy
```

### Reinforcement-learning agent

```bash
# Train on a basket (pass a comma list) and validate/test on one symbol.
# (Programmatic: train_agent(symbol=[...], val_symbol="^GSPC"))
python main.py --mode train --symbol "^GSPC" --episodes 100

# Out-of-sample backtest: evaluates strictly AFTER the training cutoff,
# using the scaler saved during training (no leakage).
python main.py --mode test --symbol "^GSPC" --model models/model_IDX_GSPC_best.pth

# Paper-trading signal (simulated book in paper_portfolio.json — no real orders)
python main.py --mode signal --symbol "AAPL,MSFT,GOOGL" --model models/model_IDX_GSPC_best.pth
```

### Hyperparameter search / full pipeline

```bash
python optuna_search.py --symbol AAPL --model_type pc --trials 30
python auto_train.py    --symbol AAPL --model_type pc --trials 30 --episodes 200
```

### Other

```bash
python main.py --mode export --symbol AAPL --source yfinance   # processed CSV -> exports/
tensorboard --logdir runs                                      # training curves
```

Data sources: `yfinance` (default), `stooq`, `binance`.

---

## 🧠 How it works

### RL agent
- **Dueling Double-DQN (D3QN)** with an **LSTM** backbone over a 30-day window.
- **Predictive-Coding variant** (`--model_type pc`): adds a generative head that
  also minimizes market "surprise" (prediction error).
- **Differential Sharpe ratio reward** — optimizes risk-adjusted return online,
  not just raw P&L.
- **Stationary, scale-invariant features** so 2016 and 2026 (and AAPL vs an
  index) look statistically comparable — see below.
- **Multi-symbol, random-window training** to discourage memorizing one path.

### Features (`src/data.py: get_feature_list`)
14 stationary features: log returns, RSI, MACD histogram (price-relative), price
vs SMA/EMA ratios, Bollinger position, ATR/price, relative volume, OBV z-score,
fractional differencing of **log** price, and three return-based "quantum"
features (mass/trend/uncertainty, after Zhang & Huang 2010). All scaled and
clipped to ±5σ. **No raw price levels** are fed to the model (that was the main
cause of poor generalization).

### Environment (`src/env.py`)
- Actions: Hold / Buy (all-in) / Sell (all-out). Fees 0.1%, slippage 0.05%.
- Stationary portfolio state: cash fraction, position fraction, unrealized P&L.
- Random-window episodes via `reset(start_step=..., episode_length=...)`.

### Systematic strategies (`src/strategy.py`)
- **Diversification**: equal-weighting a basket roughly doubles the index's
  risk-adjusted return (Sharpe ~1.1 vs ~0.7).
- **Cross-sectional momentum**: monthly-rebalanced equal weight on the top-N
  positive 6-month performers (Jegadeesh & Titman).

---

## 🗄️ Data caching
Market data is cached under `data_cache/<symbol>/<source>/<year>.csv`. The cache
auto-refreshes when it doesn't reach within ~5 days of the requested end date, so
the perpetually-incomplete current year stays up to date.

---

## 📊 Honest performance

Out-of-sample (Feb–Jun 2026, 19 symbols), the RL agent beat buy-and-hold **53%**
of the time — essentially a coin flip — though it tends to reduce drawdowns.

Systematic strategies, 2014–2026 ($10k start, momentum net of costs):

| Strategy | CAGR | Sharpe | Max DD | Grew to |
|---|---:|---:|---:|---:|
| S&P 500 buy & hold | 11.8% | 0.73 | −34% | $40k |
| Equal-weight basket | 20.0% | 1.14 | −32% | $97k |
| Momentum top-5 | 24.5% | 1.10 | −33% | $153k |

**Caveats that matter:**
- The default universe is hand-picked survivors → **survivorship bias** inflates
  absolute returns. The *relative* ranking (momentum > equal-weight > index > RL)
  is the trustworthy signal.
- Momentum has periodic crashes (it was flat 2018–2021 here).
- Past performance ≠ future returns. **Paper-trade before risking capital.**

---

## 📂 Project structure

| Path | Purpose |
|---|---|
| `main.py` | CLI entry point (`train`, `test`, `export`, `signal`, `allocate`, `strategy`) |
| `src/strategy.py` | Momentum/diversification strategy: backtest + live allocation |
| `src/live.py` | Paper-trading bot (simulated book in `paper_portfolio.json`) |
| `src/train.py` | RL training loop + leakage-free out-of-sample backtest |
| `src/model.py` | Standard D3QN agent |
| `src/model_pc.py` | Predictive-Coding hybrid agent |
| `src/env.py` | Trading environment (DSR reward, random windows) |
| `src/data.py` | Data fetching + stationary feature engineering |
| `src/dataset.py` | Year/symbol-partitioned cache with staleness refresh |
| `optuna_search.py` / `auto_train.py` | Hyperparameter search / full pipeline |
| `1009.4843.pdf` | Zhang & Huang quantum-finance paper (feature inspiration) |

### Outputs
`models/model_*.pth` + `models/scaler_*.pkl` (paired) · `backtest_*.png` ·
`strategy_comparison.png` · `exports/` · `runs/` · `data_cache/` ·
`paper_portfolio.json`
</content>
