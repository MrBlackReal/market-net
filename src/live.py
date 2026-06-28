"""Paper-trading runner: applies the trained agent to the latest market data.

This simulates trades against a persisted paper portfolio (paper_portfolio.json)
— it does NOT place real orders. Run it on a schedule (e.g. once per trading
day) to let the agent manage the simulated book and track performance honestly
before risking any real capital.
"""
import json
import os
import pickle
from datetime import datetime, timedelta

import numpy as np

from src.data import fetch_data, add_indicators, get_feature_list, scale_features
from src.model import DQNAgent

PORTFOLIO_FILE = "paper_portfolio.json"
ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}


def _load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {}


def _save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)


def paper_trade(symbol, model_path="models/model_IDX_GSPC_best.pth",
                scaler_symbol="^GSPC", hidden_dim=128, lookback=30,
                initial_balance=10000.0, fee_rate=0.001, slippage_rate=0.0005,
                execute=True):
    """Decide and (paper-)apply one action for `symbol` on the latest bar."""
    safe = scaler_symbol.replace("/", "_").replace("^", "IDX_")
    scaler_path = f"models/scaler_{safe}.pkl"
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print(f"ERROR: missing model/scaler ({model_path} / {scaler_path}). Train first.")
        return None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    df = fetch_data(symbol, start, end)
    if df is None or len(df) < lookback + 60:
        print(f"ERROR: not enough recent data for {symbol}.")
        return None
    df = add_indicators(df)
    feats = get_feature_list(df)
    scaled = scale_features(scaler, df[feats].values)
    price = float(df['Close'].iloc[-1])
    date = str(df.index[-1])[:10]

    # Current paper position for this symbol.
    port = _load_portfolio()
    st = port.get(symbol, {"cash": initial_balance, "shares": 0.0, "entry": 0.0})
    net = st["cash"] + st["shares"] * price
    cash_frac = st["cash"] / (net + 1e-9)
    pos_frac = st["shares"] * price / (net + 1e-9)
    unreal = (price / st["entry"] - 1.0) if st["shares"] > 0 and st["entry"] > 0 else 0.0

    # Build the latest observation (market window + portfolio state).
    obs = np.zeros((lookback, len(feats) + 3), dtype=np.float32)
    obs[:, :len(feats)] = scaled[-lookback:]
    obs[:, -3:] = [cash_frac, pos_frac, unreal]

    agent = DQNAgent(len(feats) + 3, 3, hidden_dim=hidden_dim)
    agent.load(model_path)
    agent.epsilon = 0.0
    action = agent.act(obs)

    note = "no change"
    if action == 1 and st["shares"] == 0:          # Buy (only if flat)
        exec_p = price * (1 + slippage_rate)
        shares = st["cash"] // exec_p
        cost = shares * exec_p
        fee = cost * fee_rate
        if shares > 0 and st["cash"] >= cost + fee:
            st["shares"] += shares
            st["cash"] -= cost + fee
            st["entry"] = exec_p
            note = f"BOUGHT {shares:.0f} @ {exec_p:.2f}"
    elif action == 2 and st["shares"] > 0:          # Sell
        exec_p = price * (1 - slippage_rate)
        revenue = st["shares"] * exec_p
        fee = revenue * fee_rate
        note = f"SOLD {st['shares']:.0f} @ {exec_p:.2f}"
        st["cash"] += revenue - fee
        st["shares"] = 0.0
        st["entry"] = 0.0

    net = st["cash"] + st["shares"] * price
    if execute:
        port[symbol] = st
        _save_portfolio(port)

    print(f"[{date}] {symbol:6}  price={price:9.2f}  SIGNAL={ACTIONS[action]:4}  {note}")
    print(f"         paper book: cash={st['cash']:.2f}  shares={st['shares']:.0f}  "
          f"net_worth={net:.2f}  (PnL {net / initial_balance - 1:+.2%})")
    return {"symbol": symbol, "date": date, "price": price,
            "action": ACTIONS[action], "net_worth": net}
