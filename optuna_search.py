import optuna
import argparse
from src.train import train_agent

def objective(trial, args):
    # Suggest hyperparameters
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epsilon_decay": trial.suggest_float("epsilon_decay", 0.99, 0.999),
    }
    
    if args.model_type == "pc":
        params["pred_alpha"] = trial.suggest_float("pred_alpha", 0.1, 1.0)

    # Call the UNIFIED training function
    # Search is faster: 5 episodes, 5 year range
    val_worth = train_agent(
        symbol=args.symbol,
        source=args.source,
        model_type=args.model_type,
        episodes=5,
        start_years=5,
        val_years=1,
        is_search=True,
        **params
    )
    
    return val_worth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--source", choices=["yfinance", "stooq", "binance"], default="yfinance")
    parser.add_argument("--model_type", choices=["standard", "pc"], default="standard")
    parser.add_argument("--trials", type=int, default=20)
    
    args = parser.parse_args()

    print(f"Starting Optuna Search for {args.symbol} using {args.model_type} agent...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)

    print("\n" + "="*30)
    print(f"Optimization Finished for {args.symbol} ({args.model_type})")
    print(f"Best Trial Score (Net Worth): {study.best_value:.2f}")
    print("Best hyperparameters:", study.best_params)
    print("="*30)