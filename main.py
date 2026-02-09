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
    parser.add_argument("--mode", choices=["train", "test", "export"], default="train")
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
    elif args.mode == "export":
        from src.data import export_processed_data
        from datetime import datetime, timedelta
        today = datetime(2026, 2, 8)
        start_date = (today - timedelta(days=365*10)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        export_processed_data(args.symbol, start_date, end_date, source=args.source)