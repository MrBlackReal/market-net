"""MetaTrader 5 broker adapter for the momentum bot.

Translates the bot's target momentum weights into MT5 orders against a (demo or
live) account. Designed for a DEMO/CFD account first.

READ THIS BEFORE USING IT FOR ANYTHING REAL:
  * The default universe trades on MT5 as **stock CFDs**, which charge overnight
    swap/financing on positions you hold. This is a hold-for-weeks strategy, so
    financing is large: in backtest, a realistic 6-8%/yr swap erases the entire
    momentum edge vs simply holding the S&P (see replay_history(swap_annual=...)).
    Validate on demo with your broker's *actual* swap before trusting it.
  * The official `MetaTrader5` Python package talks to a running MT5 *terminal*
    over Windows IPC — it is effectively Windows-only. On Linux you need the
    terminal under Wine or in a Windows VM. See MT5_SETUP.md.
  * Broker symbol names vary ("AAPL" vs "AAPL.US" vs "#AAPL"). Set `suffix`
    and/or `symbol_map` to match your broker's Market Watch.
  * This adapter has NOT been run against a live terminal in this repo. Start in
    dry_run=True (the default) and inspect the intended orders before sending.

Usage:
    from src.mt5_broker import MT5Bot
    bot = MT5Bot(login=12345, password="...", server="Broker-Demo",
                 suffix=".US")            # match your broker
    bot.connect()
    bot.rebalance(dry_run=True)           # print intended orders only
    # bot.rebalance(dry_run=False)        # actually send (demo first!)
    bot.shutdown()
"""
from datetime import datetime, timedelta

from src.strategy import DEFAULT_UNIVERSE
from src.paper_bot import _target_weights, _load_closes

try:
    import MetaTrader5 as mt5
    _HAVE_MT5 = True
except Exception:                          # not installed / not on Windows
    mt5 = None
    _HAVE_MT5 = False


class MT5Bot:
    def __init__(self, login=None, password=None, server=None, path=None,
                 universe=None, top_n=5, lookback=126, suffix="",
                 symbol_map=None, deviation=20, magic=770077):
        self.login = login
        self.password = password
        self.server = server
        self.path = path                   # path to terminal64.exe (optional)
        self.universe = universe or DEFAULT_UNIVERSE
        self.top_n = top_n
        self.lookback = lookback
        self.suffix = suffix               # appended to each ticker, e.g. ".US"
        self.symbol_map = symbol_map or {} # explicit overrides {"AAPL": "#AAPL"}
        self.deviation = deviation
        self.magic = magic

    # --- symbol helpers -----------------------------------------------------
    def broker_symbol(self, ticker):
        return self.symbol_map.get(ticker, f"{ticker}{self.suffix}")

    # --- connection ---------------------------------------------------------
    def connect(self):
        if not _HAVE_MT5:
            raise RuntimeError(
                "MetaTrader5 package not available. It requires a running MT5 "
                "terminal (Windows, or Wine/VM on Linux). See MT5_SETUP.md.")
        kwargs = {}
        if self.path:
            kwargs["path"] = self.path
        if self.login:
            kwargs.update(login=int(self.login), password=self.password,
                          server=self.server)
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info failed: {mt5.last_error()}")
        print(f"Connected to MT5: login={info.login} server={info.server} "
              f"equity={info.equity:.2f} {info.currency}")
        return info

    def shutdown(self):
        if _HAVE_MT5:
            mt5.shutdown()

    # --- target computation (reuses the validated momentum logic) -----------
    def target_weights(self):
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        px = _load_closes(self.universe, start, end)
        if px is None or len(px) < self.lookback + 1:
            raise RuntimeError("not enough price history to compute targets.")
        return _target_weights(px, self.top_n, self.lookback)

    # --- current positions --------------------------------------------------
    def _current_volume(self, bsym):
        """Net long volume (lots) currently held for a broker symbol."""
        positions = mt5.positions_get(symbol=bsym) or []
        vol = 0.0
        for p in positions:
            vol += p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume
        return vol

    @staticmethod
    def _round_step(volume, step):
        if step <= 0:
            return volume
        return round(round(volume / step) * step, 8)

    # --- rebalance ----------------------------------------------------------
    def rebalance(self, dry_run=True):
        """Move the account toward the momentum target weights (long-only).

        dry_run=True prints the intended orders without sending them.
        """
        if not _HAVE_MT5:
            raise RuntimeError("MetaTrader5 not available; cannot rebalance.")
        equity = mt5.account_info().equity
        weights = self.target_weights()
        print(f"\nTarget weights: "
              f"{ {k: round(v, 3) for k, v in weights.items()} or 'CASH (risk-off)'}")

        # Union of names we hold and names we want, so we can also exit losers.
        held = {p.symbol for p in (mt5.positions_get() or [])}
        wanted = {self.broker_symbol(t): t for t in weights}
        all_bsyms = held | set(wanted)

        plan = []
        for bsym in sorted(all_bsyms):
            si = mt5.symbol_info(bsym)
            if si is None:
                print(f"  ! {bsym}: not found in Market Watch; skipping "
                      f"(check suffix/symbol_map).")
                continue
            if not si.visible:
                mt5.symbol_select(bsym, True)
                si = mt5.symbol_info(bsym)
            tick = mt5.symbol_info_tick(bsym)
            if tick is None or tick.ask <= 0:
                print(f"  ! {bsym}: no tick; skipping.")
                continue

            ticker = wanted.get(bsym)
            w = weights.get(ticker, 0.0) if ticker else 0.0
            target_value = equity * w
            contract = si.trade_contract_size or 1.0
            target_vol = self._round_step(target_value / (tick.ask * contract),
                                          si.volume_step)
            if 0 < target_vol < si.volume_min:
                target_vol = si.volume_min  # can't take a position below min lot
            cur_vol = self._current_volume(bsym)
            delta = self._round_step(target_vol - cur_vol, si.volume_step)
            if abs(delta) < (si.volume_step or 1e-9):
                continue
            plan.append((bsym, delta, tick))

        if not plan:
            print("Already at target — no orders needed.")
            return []

        sent = []
        # Sells/reductions first to free margin, then buys.
        for bsym, delta, tick in sorted(plan, key=lambda x: x[1]):
            is_buy = delta > 0
            price = tick.ask if is_buy else tick.bid
            order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
            side = "BUY " if is_buy else "SELL"
            print(f"  {side} {abs(delta):8.2f} lots {bsym:10} @ {price:.2f}")
            if dry_run:
                continue
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": bsym,
                "volume": abs(delta),
                "type": order_type,
                "price": price,
                "deviation": self.deviation,
                "magic": self.magic,
                "comment": "momentum-bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_mode(bsym),
            }
            res = mt5.order_send(req)
            ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
            print(f"     -> {'OK' if ok else 'FAILED'} "
                  f"retcode={getattr(res, 'retcode', None)} "
                  f"{getattr(res, 'comment', '')}")
            sent.append((bsym, delta, ok))
        if dry_run:
            print("\n(dry run — no orders sent. Pass dry_run=False to execute.)")
        return sent

    def _filling_mode(self, bsym):
        """Pick a filling mode the symbol supports (brokers differ)."""
        si = mt5.symbol_info(bsym)
        modes = getattr(si, "filling_mode", 0)
        if modes & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        if modes & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN
