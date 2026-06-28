# CLAUDE.md

Guidance for working in this repo. Read this before changing training, data, or
evaluation code.

## What this is
A trading research platform with two tracks:
- **RL agent** — Dueling Double-DQN + LSTM (`src/model.py`), plus a
  Predictive-Coding variant (`src/model_pc.py`). Trades buy/hold/sell.
- **Systematic strategies** — diversification + cross-sectional momentum
  (`src/strategy.py`). This is the track with an actual, validated edge; the RL
  agent trades near chance out-of-sample. Be honest about that in any reporting.

## Environment
- **CPU-only** (Intel i7-10610U, no usable GPU). `torch` is the `+cpu` build.
- Use the venv interpreter: `./venv/bin/python` (Python 3.14).
- oneDNN/**MKLDNN is enabled on purpose** in `model.py`/`model_pc.py` (~3.4× faster
  LSTM). Do **not** disable it. Thread count is 4 (= physical cores); don't raise it.
- Long runs: launch in background and use `python -u` (stdout is block-buffered
  otherwise, so logs look empty mid-run). Foreground `sleep` is blocked by the harness.

## Commands
```bash
./venv/bin/python main.py --mode allocate --capital 10000     # today's momentum portfolio
./venv/bin/python main.py --mode strategy                     # strategy backtest + plot
./venv/bin/python main.py --mode train  --symbol "^GSPC"      # train RL agent
./venv/bin/python main.py --mode test   --symbol "^GSPC" --model models/model_IDX_GSPC_best.pth
./venv/bin/python main.py --mode signal --symbol "AAPL,MSFT"  # RL paper-trade (no real orders)
./venv/bin/python main.py --mode paper  --capital 10000       # momentum paper-bot: one daily tick
./venv/bin/python main.py --mode paper_report                 # momentum paper-bot: realised track record
./venv/bin/python main.py --mode paper_backtest               # replay the bot's own execution path vs S&P B&H
```
The momentum paper-bot (`src/paper_bot.py`) is the autonomous, self-driving book
built on the *validated* strategy track. State: `paper_book.json` /
`paper_book_history.csv` (separate from the RL agent's `paper_portfolio.json`).
It is idempotent per bar (safe to run repeatedly), rebalances monthly to the
top-N positive-6mo-momentum names, pays `ONE_WAY_COST` on turnover, and always
reports vs the S&P 500 buy-and-hold baseline. `run_paper_bot.sh` is a cron-ready
wrapper if daily automation is wanted.

**MT5 (CFDs):** `src/mt5_broker.py` + `--mode mt5` rebalance an MT5 demo account
to the momentum targets (credentials via `MT5_LOGIN/PASSWORD/SERVER/SUFFIX` env;
dry-run unless `MT5_EXECUTE=1`). See `MT5_SETUP.md`. **Caveat:** the universe
trades as stock CFDs whose overnight swap (~6-8%/yr) erases the momentum edge in
backtest — model it with `replay_history(swap_annual=...)`. Real-share brokers
avoid this. The `MetaTrader5` package is Windows-only (needs Wine/VM on Linux);
the import is guarded so the rest of the repo is unaffected.

Multi-symbol RL training is programmatic: `train_agent(symbol=[...], val_symbol="^GSPC")`.
There is **no test suite**; validate changes by running the relevant mode.

## Critical conventions (don't regress these)
- **No data leakage.** `preprocess_data` / a pooled `StandardScaler` is *fit on
  training data only*. Everything else (validation, backtest, live) must
  `scale_features(scaler, ...)` (transform, never refit). Training persists the
  scaler to `models/scaler_<safe_symbol>.pkl`; backtest/live load it.
- **Out-of-sample = strictly after the cutoff.** Training uses data ≤
  `datetime(2026, 2, 8)` (hardcoded "today" in `train.py`). `backtest()` only
  scores bars after `train_cutoff`, with an unscored lookback/indicator lead-in.
- **Features must stay stationary / scale-invariant** (`get_feature_list`). Never
  feed raw price levels (Close/SMA/EMA/Bollinger/MACD/volume) to the model — that
  was the main generalization bug. Add new features as ratios/returns/z-scores.
- **Symbol → filename**: `safe = symbol.replace("/","_").replace("^","IDX_")`
  (so `^GSPC` → `IDX_GSPC`). Models and scalers are paired and keyed by the
  validation/test symbol.

## Data layer (`src/data.py`, `src/dataset.py`)
- `fetch_data(symbol, start, end, source)` — checks cache, else downloads and
  merges. Sources: `yfinance` (default), `stooq`, `binance`.
- Cache: `data_cache/<safe_symbol>/<source>/<year>.csv`. `load_full_range`
  returns `None` (forcing a refetch) when the cache doesn't reach within 5 days
  of the requested end — this keeps the current (incomplete) year fresh.
- `add_indicators` computes raw indicators *then* stationary transforms; it joins
  `^GSPC` as market context for non-index symbols, but `get_feature_list`
  deliberately excludes market columns so feature count is identical across symbols.

## RL specifics
- Reward = **differential Sharpe ratio** of net-worth returns (`env._dsr_reward`),
  clipped to ±1, with a warmup guard. Not raw P&L.
- Episodes use **random windows**: `env.reset(start_step=..., episode_length=...)`.
- Portfolio state in the observation is stationary: cash fraction, position
  fraction, unrealized P&L (no absolute price).
- `replay()` runs every step (the main cost). A `train_freq` would be the obvious
  speedup if asked.

## Reporting expectations
The user values calibrated honesty over optimistic framing (there is a
session goal hook that rejects overclaiming). When presenting results: include
the buy-and-hold baseline, note survivorship bias in the default universe, and
don't describe a coin-flip result as an edge.
</content>
