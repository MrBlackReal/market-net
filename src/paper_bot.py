"""Autonomous momentum paper-trading bot.

A self-driving paper book built on the *validated* track (cross-sectional
momentum, see src/strategy.py) — NOT the RL agent, which trades near chance
out-of-sample. Each `run_paper_step` call is one daily tick:

  1. Fetch the latest closes for the universe + ^GSPC.
  2. Mark the existing book to market (net worth = cash + shares * price).
  3. On a month boundary, rebalance to the top-N positive-6mo-momentum names
     (equal weight), paying ONE_WAY_COST on the traded notional. Otherwise just
     hold and mark to market.
  4. Append a dated row to the history log and persist state.

It is idempotent per day: running twice on the same bar does not double-trade
or duplicate history. Run it once per trading day (see main.py --mode paper, or
schedule it) and it will manage the simulated book on its own. State lives in
paper_book.json / paper_book_history.csv — separate from the RL agent's
paper_portfolio.json so the two tracks never clobber each other.

Honesty: every report prints the buy-and-hold S&P 500 baseline over the same
window, and the default universe is survivorship-biased large caps (the
momentum-vs-equal-weight *relative* edge is the trustworthy signal).
"""
import csv
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data import fetch_data
from src.strategy import (DEFAULT_UNIVERSE, ONE_WAY_COST, TRADING_DAYS,
                          _load_closes, _metrics, _vol_scaled_weights)

BOOK_FILE = "paper_book.json"
HISTORY_FILE = "paper_book_history.csv"


def _new_book(capital, asof):
    return {
        "initial_capital": capital,
        "inception": asof,
        "cash": capital,
        "positions": {},          # symbol -> shares (float)
        "last_rebalance": None,    # "YYYY-MM"
        "last_bar": None,          # "YYYY-MM-DD" of last processed bar
        "spx_inception_close": None,
    }


def _load_book():
    if os.path.exists(BOOK_FILE):
        with open(BOOK_FILE) as f:
            return json.load(f)
    return None


def _save_book(book):
    with open(BOOK_FILE, "w") as f:
        json.dump(book, f, indent=2)


def _target_weights(px, top_n, lookback, skip=0, vol_scale=False,
                    trend_filter=False, spx_px=None):
    """Weights on the top-N names with positive trailing momentum.

    Default (skip=21, lookback=252, vol_scale=True, trend_filter=True) uses
    the enhanced signal: 12-1 month momentum, vol-scaled sizing, trend filter.

    Returns dict {symbol: weight}; weights sum to <= 1.0 (remainder = cash).
    """
    n = len(px)
    if n < lookback + max(skip, 1):
        return {}

    # Trend filter: if SPX < 200-day MA, go to cash
    if trend_filter and spx_px is not None:
        spx_valid = spx_px.dropna()
        if len(spx_valid) >= 200:
            ma200 = float(spx_valid.iloc[-200:].mean())
            if float(spx_valid.iloc[-1]) < ma200:
                return {}

    if skip > 0:
        mom = (px.iloc[-skip] / px.iloc[-lookback] - 1.0).dropna()
    else:
        mom = (px.iloc[-1] / px.iloc[-lookback] - 1.0).dropna()
    top = mom[mom > 0].nlargest(top_n)

    if len(top) == 0:
        return {}

    if vol_scale:
        return _vol_scaled_weights(px, top.index)
    w = 1.0 / len(top)
    return {sym: w for sym in top.index}


def _target_shares(net, prices, win, top_n, lookback, one_way_cost, positions,
                   skip=21, vol_scale=True, trend_filter=True, spx_px=None):
    """Target share counts for a rebalance with fee headroom.

    Two-pass estimate: size to full `net`, measure turnover, re-size against
    `net - estimated_fees` so the book never goes cash-negative.
    """
    tw = _target_weights(win, top_n, lookback, skip=skip, vol_scale=vol_scale,
                         trend_filter=trend_filter, spx_px=spx_px)

    def shares_for(investable):
        out = {}
        for s, w in tw.items():
            p = float(prices[s]) if s in prices.index else 0.0
            if p > 0:
                out[s] = (investable * w) / p
        return out

    def turnover(ts):
        notional = 0.0
        for s in set(ts) | set(positions):
            p = float(prices[s]) if s in prices.index else 0.0
            if p > 0:
                notional += abs(ts.get(s, 0.0) - positions.get(s, 0.0)) * p
        return notional

    ts = shares_for(net)
    fees = turnover(ts) * one_way_cost
    return shares_for(max(0.0, net - fees))


def _append_history(asof, net_worth, cash, n_pos, spx_close):
    new = not os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["date", "net_worth", "cash", "n_positions", "spx_close"])
        wr.writerow([asof, f"{net_worth:.2f}", f"{cash:.2f}", n_pos, f"{spx_close:.2f}"])


def run_paper_step(universe=None, top_n=5, lookback=126, skip=0,
                   vol_scale=False, trend_filter=False,
                   capital=10000.0, one_way_cost=ONE_WAY_COST, verbose=True):
    """Run one daily tick of the momentum paper bot. Returns a summary dict."""
    universe = universe or DEFAULT_UNIVERSE
    end = datetime.now().strftime("%Y-%m-%d")
    buf_days = (lookback + skip) * 2 + 30
    start = (datetime.now() - timedelta(days=buf_days)).strftime("%Y-%m-%d")

    px = _load_closes(universe, start, end)
    if px is None or len(px) < lookback + skip + 2:
        print("ERROR: not enough recent data to run the bot.")
        return None
    prices = px.iloc[-1]
    asof = str(px.index[-1])[:10]

    spx = fetch_data("^GSPC", start, end)
    # The current (forming) bar can be NaN until the index settles; use the last
    # valid close so the buy-and-hold baseline is never NaN.
    spx_close = float("nan")
    spx_px = None
    if spx is not None:
        spx_px = spx['Close'].reindex(px.index).ffill()
        valid = spx['Close'].dropna()
        if len(valid):
            spx_close = float(valid.iloc[-1])

    book = _load_book() or _new_book(capital, asof)
    if book.get("spx_inception_close") is None and not np.isnan(spx_close):
        book["spx_inception_close"] = spx_close

    def mark_to_market():
        val = book["cash"]
        for sym, sh in book["positions"].items():
            if sym in prices.index:
                val += sh * float(prices[sym])
        return val

    # Idempotency: if we've already processed this bar, just report — no trades,
    # no duplicate history row.
    already_done = book.get("last_bar") == asof
    cur_month = asof[:7]
    is_rebalance = (book.get("last_rebalance") != cur_month) and not already_done

    trades = []
    if is_rebalance:
        net = mark_to_market()
        target_shares = _target_shares(net, prices, px, top_n, lookback,
                                       one_way_cost, book["positions"],
                                       skip=skip, vol_scale=vol_scale,
                                       trend_filter=trend_filter, spx_px=spx_px)

        # Sells / trims first (free up cash), then buys.
        all_syms = set(book["positions"]) | set(target_shares)
        for sym in sorted(all_syms):
            cur = book["positions"].get(sym, 0.0)
            tgt = target_shares.get(sym, 0.0)
            delta = tgt - cur
            if abs(delta) < 1e-9:
                continue
            p = float(prices[sym]) if sym in prices.index else 0.0
            if p <= 0:
                continue
            notional = abs(delta) * p
            cost = notional * one_way_cost
            book["cash"] -= delta * p + cost          # delta>0 buy: cash down
            book["positions"][sym] = tgt
            trades.append((sym, delta, p))
        book["positions"] = {s: sh for s, sh in book["positions"].items() if abs(sh) > 1e-9}
        book["last_rebalance"] = cur_month

    net = mark_to_market()

    if not already_done:
        book["last_bar"] = asof
        _append_history(asof, net, book["cash"], len(book["positions"]), spx_close)
        _save_book(book)

    pnl = net / book["initial_capital"] - 1.0
    spx0 = book.get("spx_inception_close")
    spx_pnl = (spx_close / spx0 - 1.0) if spx0 else float("nan")

    if verbose:
        tag = "REBALANCED" if is_rebalance else ("already processed" if already_done else "HOLD")
        print(f"\n=== Paper bot tick {asof} ({tag}) ===")
        if trades:
            print("Trades:")
            for sym, d, p in trades:
                side = "BUY " if d > 0 else "SELL"
                print(f"  {side} {abs(d):8.2f} {sym:6} @ {p:10.2f}")
        print(f"{'symbol':8}{'shares':>10}{'price':>11}{'value':>12}{'weight':>9}")
        for sym in sorted(book["positions"]):
            sh = book["positions"][sym]
            p = float(prices[sym]) if sym in prices.index else 0.0
            v = sh * p
            print(f"{sym:8}{sh:>10.2f}{p:>11.2f}{v:>12,.0f}{v/net*100:>8.1f}%")
        print(f"{'CASH':8}{'':>10}{'':>11}{book['cash']:>12,.0f}{book['cash']/net*100:>8.1f}%")
        print(f"\nNet worth: ${net:,.2f}   PnL since inception ({book['inception']}): {pnl:+.2%}")
        if not np.isnan(spx_pnl):
            print(f"S&P 500 buy & hold over same window:                  {spx_pnl:+.2%}")
            print(f"Excess vs S&P (alpha):                                {pnl - spx_pnl:+.2%}")

    return {"date": asof, "net_worth": net, "pnl": pnl, "spx_pnl": spx_pnl,
            "rebalanced": is_rebalance, "n_positions": len(book["positions"])}


def replay_history(universe=None, top_n=5, lookback=126, skip=0,
                   vol_scale=False, trend_filter=False,
                   capital=10000.0, start="2023-01-01", end=None,
                   one_way_cost=ONE_WAY_COST, swap_annual=0.0, verbose=True):
    """Replay the bot's exact tick/rebalance logic day-by-day over history.

    This drives the *same* book mechanics as run_paper_step (cash, fractional
    shares, monthly momentum rebalance, ONE_WAY_COST on traded notional) bar by
    bar over [start, end], producing a realised net-worth curve. It demonstrates
    what the bot's own execution path would have earned out-of-sample, net of
    costs, against a buy-and-hold S&P 500 baseline over the identical window.

    `swap_annual`: annual financing/swap rate charged daily on held long
    notional. Set this to a non-zero value (e.g. ~0.06-0.08) to model trading
    the universe as **stock CFDs** (e.g. on MT5), where you pay overnight
    financing on positions you hold. Leave 0.0 for real (owned) shares.

    Survivorship caveat applies (the default universe is hand-picked survivors);
    the trustworthy read is the momentum book vs the B&H baseline, not the
    absolute return.
    """
    universe = universe or DEFAULT_UNIVERSE
    end = end or datetime.now().strftime("%Y-%m-%d")
    buf_days = (lookback + skip) * 2 + 30
    buf_start = (pd.Timestamp(start) - pd.Timedelta(days=buf_days)).strftime("%Y-%m-%d")
    px = _load_closes(universe, buf_start, end)
    if px is None or len(px) < lookback + skip + 2:
        print("ERROR: not enough data for replay.")
        return None

    spx = fetch_data("^GSPC", buf_start, end)
    spx_close_series = spx['Close'].reindex(px.index).ffill() if spx is not None else None

    bars = px.index[px.index >= pd.Timestamp(start)]
    bars = bars[px.index.get_indexer(bars) >= lookback + skip]
    if len(bars) < 2:
        print("ERROR: replay window too short after lookback lead-in.")
        return None

    cash = capital
    positions = {}            # symbol -> shares
    last_rebal_month = None
    rows = []                 # (date, net_worth, spx_close)

    for d in bars:
        prices = px.loc[d]
        win = px.loc[:d]
        month = str(d)[:7]
        spx_px_slice = spx_close_series.loc[:d] if spx_close_series is not None else None
        if month != last_rebal_month:
            net = cash + sum(sh * float(prices[s]) for s, sh in positions.items()
                             if s in prices.index)
            target_shares = _target_shares(net, prices, win, top_n, lookback,
                                           one_way_cost, positions,
                                           skip=skip, vol_scale=vol_scale,
                                           trend_filter=trend_filter,
                                           spx_px=spx_px_slice)
            for s in sorted(set(positions) | set(target_shares)):
                cur = positions.get(s, 0.0)
                tgt = target_shares.get(s, 0.0)
                delta = tgt - cur
                p = float(prices[s]) if s in prices.index else 0.0
                if p <= 0 or abs(delta) < 1e-9:
                    continue
                cash -= delta * p + abs(delta) * p * one_way_cost
                positions[s] = tgt
            positions = {s: sh for s, sh in positions.items() if abs(sh) > 1e-9}
            last_rebal_month = month
        # CFD financing: pay swap on held long notional each day held.
        if swap_annual:
            held = sum(sh * float(prices[s]) for s, sh in positions.items()
                       if s in prices.index)
            cash -= held * (swap_annual / TRADING_DAYS)
        net = cash + sum(sh * float(prices[s]) for s, sh in positions.items()
                         if s in prices.index)
        sp = float(spx_close_series.loc[d]) if spx_close_series is not None and not np.isnan(spx_close_series.loc[d]) else np.nan
        rows.append((str(d)[:10], net, sp))

    eq = np.array([r[1] for r in rows], float)
    bot = _metrics(eq)
    out = {"bot": bot}
    if verbose:
        print(f"\n=== Bot replay (realised, net of costs) {rows[0][0]} -> {rows[-1][0]} "
              f"({len(rows)} bars) ===")
        print(f"{'series':14}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>8}{'growth':>9}{'final$':>12}")
        print(f"{'momentum bot':14}{bot['cagr']:>8.1f}{bot['sharpe']:>8.2f}{bot['mdd']:>8.1f}"
              f"{bot['mult']:>8.2f}x{eq[-1]:>12,.0f}")
    sp_series = np.array([r[2] for r in rows], float)
    if np.isfinite(sp_series).sum() >= 2:
        sp_series = sp_series[np.isfinite(sp_series)]
        sp = _metrics(sp_series)
        out["sp500"] = sp
        sp_final = capital * sp["mult"]
        if verbose:
            print(f"{'S&P 500 B&H':14}{sp['cagr']:>8.1f}{sp['sharpe']:>8.2f}{sp['mdd']:>8.1f}"
                  f"{sp['mult']:>8.2f}x{sp_final:>12,.0f}")
            print(f"\nRealised excess vs S&P (alpha): {(bot['mult'] - sp['mult']) / sp['mult'] * 100:+.1f}% of B&H growth")
    return out


def report():
    """Summarise the paper book's realised track record from history."""
    if not os.path.exists(HISTORY_FILE):
        print("No history yet — run the bot at least once (--mode paper).")
        return None
    h = pd.read_csv(HISTORY_FILE)
    if len(h) < 2:
        print(f"Only {len(h)} tick(s) logged; need >=2 for metrics.")
        return None
    bot = _metrics(h["net_worth"].values)
    out = {"bot": bot}
    print(f"\n=== Paper bot track record {h['date'].iloc[0]} -> {h['date'].iloc[-1]} "
          f"({len(h)} ticks) ===")
    print(f"{'series':14}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>8}{'growth':>9}")
    print(f"{'bot':14}{bot['cagr']:>8.1f}{bot['sharpe']:>8.2f}{bot['mdd']:>8.1f}{bot['mult']:>8.2f}x")
    if h["spx_close"].notna().any():
        spx = _metrics(h["spx_close"].dropna().values)
        out["sp500"] = spx
        print(f"{'sp500 B&H':14}{spx['cagr']:>8.1f}{spx['sharpe']:>8.2f}{spx['mdd']:>8.1f}{spx['mult']:>8.2f}x")
    return out
