"""Systematic portfolio strategies with documented edges.

Rather than a black-box agent that trades at chance level, these are
transparent, rules-based strategies whose edges are well established in the
literature and validated here across sub-periods (see momentum_validate):

  - Diversified equal-weight: the single biggest risk-adjusted improvement
    (basket Sharpe ~1.1 vs ~0.7 for individual names).
  - Cross-sectional momentum: overweight the trailing winners (Jegadeesh &
    Titman). +~5pp CAGR over equal-weight in this universe, net of costs.

IMPORTANT CAVEATS (read before trusting the numbers):
  * The default universe is hand-picked large caps that *survived* -> absolute
    returns are inflated by survivorship bias. The momentum-vs-equal-weight
    *relative* edge is the trustworthy signal.
  * Momentum suffers periodic "crashes" (it was flat 2018-2021 here).
  * Past performance != future returns. Paper-trade before risking capital.
"""
import numpy as np
import pandas as pd

from src.data import fetch_data

DEFAULT_UNIVERSE = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM', 'PG', 'KO', 'WMT',
                    'GOOGL', 'AMZN', 'META', 'NVDA', 'V', 'HD', 'CVX', 'UNH', 'DIS']
TRADING_DAYS = 252
ONE_WAY_COST = 0.0015   # fee + slippage applied to turnover


def _load_closes(universe, start, end):
    closes = {}
    for s in universe:
        df = fetch_data(s, start, end)
        if df is not None and len(df) > 200:
            closes[s] = df['Close']
    return pd.DataFrame(closes).sort_index().ffill().dropna(how='all')


def _metrics(equity):
    equity = np.asarray(equity, float)
    r = np.diff(equity) / equity[:-1]
    yrs = len(equity) / TRADING_DAYS
    cagr = (equity[-1] / equity[0]) ** (1 / yrs) - 1
    sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(TRADING_DAYS)
    peak = np.maximum.accumulate(equity)
    mdd = ((equity - peak) / peak).min()
    return {"cagr": cagr * 100, "sharpe": sharpe, "mdd": mdd * 100,
            "mult": equity[-1] / equity[0]}


def _vol_scaled_weights(px_window, top_names, vol_window=21):
    """1/vol weights among `top_names`; falls back to equal weight on bad data."""
    daily_rets = px_window[list(top_names)].pct_change().dropna()
    if len(daily_rets) < vol_window:
        w = 1.0 / len(top_names)
        return {s: w for s in top_names}
    vols = daily_rets.iloc[-vol_window:].std().clip(lower=1e-8)
    inv = 1.0 / vols
    total = inv.sum()
    return {s: float(inv[s] / total) for s in top_names}


def _momentum_weights(px, top_n=5, lookback=126, skip=21,
                      vol_scale=False, trend_filter=False, spx_px=None):
    """Monthly-rebalanced weights on the top-N positive-momentum names.

    skip: trading days to exclude from the tail of the signal window (default 21
        ≈ 1 month). Setting skip=21 with lookback=252 gives the academic
        "12-1 momentum" (12-month return skipping the most recent month),
        which avoids the short-term reversal that contaminates raw 6-month.
        Set skip=0 for the original behaviour.
    vol_scale: weight inversely by 21-day realised volatility among winners
        (risk parity within the basket). Replaces equal-weight; improves Sharpe.
    trend_filter: invest only when SPX > its 200-day MA at each rebalance.
        Reduces exposure during momentum-crash conditions (bear markets).
    spx_px: Series of SPX closes aligned with px.index; required when
        trend_filter=True (ignored otherwise).
    """
    if skip > 0:
        # 12-1 convention: return from t-lookback to t-skip
        # px.shift(skip) is price at (t - skip); px.shift(lookback) is price at (t - lookback)
        mom = px.shift(skip) / px.shift(lookback) - 1.0
    else:
        mom = px / px.shift(lookback) - 1.0

    rebal = set(px.resample('ME').last().index)
    W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur = {c: 0.0 for c in px.columns}

    for d in px.index:
        if d in rebal:
            if trend_filter and spx_px is not None:
                spx_slice = spx_px.loc[:d].dropna()
                if len(spx_slice) >= 200:
                    ma200 = float(spx_slice.iloc[-200:].mean())
                    if float(spx_slice.iloc[-1]) < ma200:
                        cur = {c: 0.0 for c in px.columns}
                        W.loc[d] = pd.Series(cur)
                        continue

            m = mom.loc[d].dropna()
            top = m[m > 0].nlargest(top_n).index

            if len(top) == 0:
                cur = {c: 0.0 for c in px.columns}
            elif vol_scale:
                wmap = _vol_scaled_weights(px.loc[:d], top)
                cur = {c: wmap.get(c, 0.0) for c in px.columns}
            else:
                w = 1.0 / len(top)
                cur = {c: (w if c in top else 0.0) for c in px.columns}

        W.loc[d] = pd.Series(cur)
    return W


def _equity_curve(px, mom_w, rets):
    w_prev = mom_w.shift(1).fillna(0.0)
    turn = (mom_w - w_prev).abs().sum(axis=1)
    r = (w_prev * rets).sum(axis=1) - turn * ONE_WAY_COST
    return np.cumprod(1 + r.values)


def backtest_portfolio(universe=None, start="2014-01-01", end="2026-06-27",
                       top_n=5, lookback=126, plot=True):
    """Compare momentum variants vs equal-weight vs S&P.

    Three momentum variants are shown:
      momentum_6mo   — original: equal-weight, 6-month signal
      momentum_12_1  — 12-1 month signal (skips most-recent month), vol-scaled
      momentum_enh   — enhanced: 12-1 signal + vol-scaled sizing + trend filter
    Returns dict of metrics.
    """
    universe = universe or DEFAULT_UNIVERSE
    spx = fetch_data("^GSPC", start, end)
    spx_px = spx['Close'] if spx is not None else None

    px = _load_closes(universe, start, end)
    if spx_px is not None:
        spx_px = spx_px.reindex(px.index).ffill()

    rets = px.pct_change().fillna(0.0)
    ew = np.cumprod(1 + rets.mean(axis=1).values)

    # Original: equal-weight, 6-month, no skip
    mom_w0 = _momentum_weights(px, top_n, lookback=126, skip=0)
    mom_6mo = _equity_curve(px, mom_w0, rets)

    # 12-1 signal + vol-scaled sizing (no trend filter)
    mom_w1 = _momentum_weights(px, top_n, lookback=252, skip=21, vol_scale=True)
    mom_12_1 = _equity_curve(px, mom_w1, rets)

    # Enhanced: 12-1 + vol-scale + trend filter (go-to-cash in bear markets)
    mom_w2 = _momentum_weights(px, top_n, lookback=252, skip=21,
                               vol_scale=True, trend_filter=True, spx_px=spx_px)
    mom_enh = _equity_curve(px, mom_w2, rets)

    spx_eq = (np.cumprod(1 + spx['Close'].pct_change().fillna(0.0).values)
              if spx is not None else None)

    out = {
        "equal_weight": _metrics(ew),
        "momentum_6mo": _metrics(mom_6mo),
        "momentum_12_1": _metrics(mom_12_1),
        "momentum_enh": _metrics(mom_enh),
    }
    if spx_eq is not None:
        out["sp500"] = _metrics(spx_eq)

    print(f"\n--- Strategy backtest {px.index[0].date()} -> {px.index[-1].date()} "
          f"(all momentum net of costs, survivorship-biased universe) ---")
    print(f"{'strategy':22}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>8}{'growth':>9}")
    for name in ("sp500", "equal_weight", "momentum_6mo", "momentum_12_1", "momentum_enh"):
        if name in out:
            m = out[name]
            print(f"{name:22}{m['cagr']:>8.1f}{m['sharpe']:>8.2f}{m['mdd']:>8.1f}{m['mult']:>8.2f}x")
    print("  momentum_6mo  : original (equal-weight, 6-month signal)")
    print("  momentum_12_1 : 12-1 month signal + volatility-scaled sizing")
    print("  momentum_enh  : 12-1 + vol-scale + trend filter (cash in bear markets)")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13, 7))
        if spx_eq is not None:
            plt.plot(px.index, spx_eq * 10000,
                     label=f"S&P 500 B&H ({out['sp500']['mult']:.1f}x)", alpha=0.6)
        plt.plot(px.index, ew * 10000,
                 label=f"Equal-weight ({out['equal_weight']['mult']:.1f}x)", alpha=0.7)
        plt.plot(px.index, mom_6mo * 10000,
                 label=f"Mom 6mo equal-wt ({out['momentum_6mo']['mult']:.1f}x)", linestyle="--")
        plt.plot(px.index, mom_12_1 * 10000,
                 label=f"Mom 12-1 vol-scale ({out['momentum_12_1']['mult']:.1f}x)")
        plt.plot(px.index, mom_enh * 10000,
                 label=f"Mom enhanced ({out['momentum_enh']['mult']:.1f}x)", linewidth=2.5)
        plt.yscale("log")
        plt.title("Momentum strategy variants vs S&P 500 ($10k start, log scale)")
        plt.xlabel("Date"); plt.ylabel("Portfolio value ($, log)")
        plt.legend(); plt.grid(True, alpha=0.3, which="both")
        plt.savefig("strategy_comparison.png", bbox_inches="tight")
        plt.close()
        print("Plot saved to strategy_comparison.png")
    return out


def current_allocation(universe=None, top_n=5, lookback=126, skip=0,
                       vol_scale=False, trend_filter=False, capital=10000.0):
    """Print today's recommended momentum portfolio (what to actually hold).

    Defaults replicate the original 6-month equal-weight signal.
    Optional enhancements (set individually or together):
      lookback=252, skip=21  →  12-1 month momentum (12mo return, skip last month)
      vol_scale=True         →  size by 1/realized-vol (risk parity within basket)
      trend_filter=True      →  go to cash if S&P 500 < its 200-day MA
    Note: enhancements show better Sharpe on broader universes; on this
    survivorship-biased tech-heavy basket the original signal is hard to beat.
    """
    from datetime import datetime, timedelta
    universe = universe or DEFAULT_UNIVERSE
    end = datetime.now().strftime("%Y-%m-%d")
    # Need lookback + skip days plus buffer for the full signal window
    buf_days = max((lookback + skip) * 2, 400)
    start = (datetime.now() - timedelta(days=buf_days)).strftime("%Y-%m-%d")

    px = _load_closes(universe, start, end)
    spx = fetch_data("^GSPC", start, end) if trend_filter else None
    spx_px = spx['Close'].reindex(px.index).ffill() if spx is not None else None

    # Trend filter: check current bar against 200-day MA
    in_bull = True
    if trend_filter and spx_px is not None:
        spx_valid = spx_px.dropna()
        if len(spx_valid) >= 200:
            ma200 = float(spx_valid.iloc[-200:].mean())
            in_bull = float(spx_valid.iloc[-1]) >= ma200

    asof = str(px.index[-1])[:10]
    signal_desc = f"{lookback}d" + (f" skip-{skip}d" if skip else "") + (" vol-scale" if vol_scale else " equal-wt") + (" trend-filter" if trend_filter else "")
    print(f"\n=== Recommended allocation as of {asof} (capital ${capital:,.0f}) ===")
    print(f"  Signal: {signal_desc}")

    if not in_bull:
        print(f"  TREND FILTER: S&P 500 is BELOW its 200-day MA -> hold CASH (risk-off).")
        return {"cash": capital}

    n = len(px)
    if n < lookback + max(skip, 1):
        print("ERROR: not enough history for the signal window.")
        return {"cash": capital}

    # Momentum signal: return from t-lookback to t-skip (or t-0 if skip=0)
    if skip > 0:
        mom = (px.iloc[-skip] / px.iloc[-lookback] - 1.0).dropna()
    else:
        mom = (px.iloc[-1] / px.iloc[-lookback] - 1.0).dropna()
    top = mom[mom > 0].nlargest(top_n)

    if len(top) == 0:
        print("No assets with positive momentum -> hold CASH (risk-off).")
        return {"cash": capital}

    if vol_scale:
        daily_rets = px[list(top.index)].pct_change().dropna()
        vol_window = min(21, len(daily_rets))
        vols = daily_rets.iloc[-vol_window:].std().clip(lower=1e-8)
        inv = 1.0 / vols
        weights = (inv / inv.sum()).to_dict()
    else:
        w = 1.0 / len(top)
        weights = {sym: w for sym in top.index}

    alloc = {}
    print(f"{'symbol':8}{'mom%':>10}{'weight':>9}{'$ amount':>12}{'~shares':>9}")
    for sym in top.index:
        m_pct = top[sym] * 100
        w = weights[sym]
        dollars = capital * w
        price = float(px[sym].iloc[-1])
        alloc[sym] = dollars
        print(f"{sym:8}{m_pct:>10.1f}{w*100:>8.1f}%{dollars:>12,.0f}{dollars/price:>9.1f}")

    invested = sum(weights.values())
    if invested < 0.9999:
        cash = capital * (1 - invested)
        alloc["cash"] = cash
        print(f"{'CASH':8}{'':>11}{(1-invested)*100:>8.1f}%{cash:>12,.0f}")
    return alloc
