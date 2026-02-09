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
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--source", choices=["yfinance", "stooq", "binance"], default="yfinance")
    parser.add_argument("--model_type", choices=["standard", "pc"], default="standard")
        parser.add_argument("--episodes", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--model", default="model_AAPL_best.pth")
        
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
        else:
            backtest(args.symbol, args.model, model_type=args.model_type, hidden_dim=args.hidden_dim)
    