# Running the momentum bot on MetaTrader 5 (demo)

This connects the momentum bot (`src/mt5_broker.py`) to an MT5 account and
rebalances it to the bot's target weights. **Start on a demo account.**

## ⚠️ Read this first — does MT5 even make sense here?
On MT5 the universe (AAPL, MSFT, …) trades as **stock CFDs**, which charge
**overnight swap/financing** on positions you hold. This strategy holds for
weeks, so financing is a major cost. In backtest:

| trading vehicle          | bot CAGR | Sharpe | edge vs S&P B&H |
|--------------------------|---------:|-------:|----------------:|
| real shares (0% swap)    |   28.4%  |  1.36  | **+23.4%**      |
| CFD @ 6%/yr swap         |   21.0%  |  1.05  | +0.5% (gone)    |
| CFD @ 8%/yr swap         |   18.6%  |  0.95  | **−6.2%**       |

Reproduce: `./venv/bin/python -c "from src.paper_bot import replay_history; replay_history(swap_annual=0.06)"`

Realistic MT5 stock-CFD swap today is ~6–8%/yr (benchmark rate + broker markup),
which erases the edge. **Check your broker's actual swap on the demo before
trusting any of this.** A real-share broker (e.g. Alpaca) avoids this entirely.

## 1. Open a demo account
Install MetaTrader 5 from your broker (one that offers **stock CFDs** covering
the universe) and open a **demo** account. Note: login (number), password,
and server name (e.g. `YourBroker-Demo`).

## 2. Linux note
The `MetaTrader5` Python package talks to a running MT5 **terminal** over Windows
IPC — it is effectively Windows-only. On this Linux machine you have two options:
- Run the MT5 terminal (and Python) under **Wine**, or
- Run them in a **Windows VM**.
The Python process and the terminal must be on the same Windows environment.

## 3. Install the package (in that Windows/Wine Python)
```
pip install MetaTrader5
```
(It will not install usefully on native Linux — that's expected; the adapter
guards the import so the rest of the repo still works.)

## 4. Match your broker's symbol names
Brokers name stock CFDs differently: `AAPL`, `AAPL.US`, `#AAPL`, `AAPL.NAS`, …
Check Market Watch, then set either a suffix or explicit overrides:
```python
MT5Bot(suffix=".US")                      # AAPL -> AAPL.US
MT5Bot(symbol_map={"AAPL": "#AAPL"})      # explicit per-symbol
```

## 5. Dry run first (prints intended orders, sends nothing)
```bash
export MT5_LOGIN=12345 MT5_PASSWORD='...' MT5_SERVER='YourBroker-Demo' MT5_SUFFIX='.US'
./venv/bin/python main.py --mode mt5
```
Inspect the BUY/SELL lots. When you're satisfied, execute on the **demo**:
```bash
MT5_EXECUTE=1 ./venv/bin/python main.py --mode mt5
```

## 6. Run it monthly
The strategy rebalances monthly. Run `--mode mt5` once near the start of each
month (or daily — it only trades when the target differs from current holdings).

## Honesty checklist before risking real money
- [ ] Confirmed the demo's **actual swap** and re-ran `replay_history(swap_annual=...)` with it.
- [ ] Verified symbol mapping covers the whole universe.
- [ ] Watched a full rebalance cycle on demo (fills, slippage, margin).
- [ ] Accepted momentum's regime risk (it underperformed the index over the last 2 years).
