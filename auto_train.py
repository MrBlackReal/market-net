import argparse
import optuna
import torch
from src.train import train_agent, backtest
from optuna_search import objective

def run_auto_pipeline(args):
    print("\n" + "="*50)
    print(f"🚀 STARTING AUTO-PIPELINE FOR {args.symbol} ({args.model_type})")
    print("="*50)

    # 1. OPTUNA SEARCH
    print(f"\n[PHASE 1] Running Hyperparameter Optimization ({args.trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    
    best_params = study.best_params
    print("\n🏆 PHASE 1 COMPLETE")
    print(f"Best Net Worth Found: {study.best_value:.2f}")
    print(f"Optimal Params: {best_params}")

    # 2. FULL TRAINING
    print(f"\n[PHASE 2] Starting Full Training ({args.episodes} episodes)...")
    train_agent(
        symbol=args.symbol,
        source=args.source,
        model_type=args.model_type,
        episodes=args.episodes,
        is_search=False,
        **best_params
    )
    print("\n✅ PHASE 2 COMPLETE: Model Saved.")

    # 3. BACKTEST
    print(f"\n[PHASE 3] Running Final Backtest...")
    model_path = f"model_{args.symbol}_best.pth"
    backtest(args.symbol, model_path, model_type=args.model_type, hidden_dim=best_params.get("hidden_dim", 128))
    print("\n🏁 AUTO-PIPELINE FINISHED.")

if __name__ == "__main__":
    # CPU Optimization
    if not torch.cuda.is_available():
        import os
        num_threads = os.cpu_count() // 2 if os.cpu_count() > 4 else os.cpu_count()
        torch.set_num_threads(num_threads)

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL", help="Asset symbol (e.g. AAPL, BTC/USDT)")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "binance", "stooq"])
    parser.add_argument("--model_type", default="pc", choices=["standard", "pc"], help="Agent architecture")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--episodes", type=int, default=200, help="Final training episodes")
    
    args = parser.parse_args()
    run_auto_pipeline(args)
