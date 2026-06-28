import argparse
import torch
from src.train import train_agent, backtest

if __name__ == "__main__":
    # Optimize CPU usage
    if not torch.cuda.is_available():
        import os
        num_threads = os.cpu_count() // 2 if os.cpu_count() > 4 else os.cpu_count()
        torch.set_num_threads(num_threads)
        print(f"CPU Optimization: Using {num_threads} threads.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "export", "signal", "allocate", "strategy", "paper", "paper_report", "paper_backtest", "mt5"], default="train")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--source", choices=["yfinance", "stooq", "binance"], default="yfinance")
    parser.add_argument("--model_type", choices=["standard", "pc"], default="standard")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--model", default="models/model_AAPL_best.pth")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_agent(
            symbol=args.symbol, 
            episodes=args.episodes, 
            batch_size=args.batch_size,
            source=args.source, 
            model_type=args.model_type,
            hidden_dim=args.hidden_dim
        )
    elif args.mode == "test":
        backtest(args.symbol, args.model, model_type=args.model_type, hidden_dim=args.hidden_dim)
    elif args.mode == "signal":
        # Paper-trading signal for one or more comma-separated symbols.
        from src.live import paper_trade
        for sym in args.symbol.split(","):
            paper_trade(sym.strip(), model_path=args.model, scaler_symbol="^GSPC",
                        hidden_dim=args.hidden_dim)
    elif args.mode == "allocate":
        # Recommended momentum portfolio for today.
        from src.strategy import current_allocation
        current_allocation(capital=args.capital)
    elif args.mode == "strategy":
        # Backtest the systematic strategies vs the S&P 500.
        from src.strategy import backtest_portfolio
        backtest_portfolio()
    elif args.mode == "paper":
        # One daily tick of the autonomous momentum paper bot.
        from src.paper_bot import run_paper_step
        run_paper_step(capital=args.capital)
    elif args.mode == "paper_report":
        # Realised track record of the paper bot so far.
        from src.paper_bot import report
        report()
    elif args.mode == "paper_backtest":
        # Replay the bot's own execution path over history vs S&P buy-and-hold.
        from src.paper_bot import replay_history
        replay_history(capital=args.capital)
    elif args.mode == "mt5":
        # Rebalance an MT5 (demo) account to the momentum targets.
        # Credentials from env: MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / MT5_SUFFIX.
        # Dry-run unless MT5_EXECUTE=1 (start on a DEMO account; CFD swap can
        # erase the edge — see replay_history(swap_annual=...) and MT5_SETUP.md).
        import os
        from src.mt5_broker import MT5Bot
        bot = MT5Bot(login=os.environ.get("MT5_LOGIN"),
                     password=os.environ.get("MT5_PASSWORD"),
                     server=os.environ.get("MT5_SERVER"),
                     suffix=os.environ.get("MT5_SUFFIX", ""))
        bot.connect()
        try:
            bot.rebalance(dry_run=os.environ.get("MT5_EXECUTE") != "1")
        finally:
            bot.shutdown()
    elif args.mode == "export":
        from src.data import export_processed_data
        from datetime import datetime, timedelta
        today = datetime(2026, 2, 8)
        start_date = (today - timedelta(days=365*10)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        export_processed_data(args.symbol, start_date, end_date, source=args.source)