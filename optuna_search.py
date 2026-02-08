import optuna
import torch
from src.data import fetch_data, add_indicators, preprocess_data
from src.env import TradingEnv
from src.model import DQNAgent, QNetwork
from src.train import evaluate_agent
from datetime import datetime, timedelta
import numpy as np

def objective(trial):
    # Hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.999)

    # Simplified training for optimization (shorter window, fewer episodes)
    symbol = "AAPL"
    today = datetime(2026, 2, 8)
    start_date = (today - timedelta(days=365*5)).strftime("%Y-%m-%d")
    val_split_date = (today - timedelta(days=365*1)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    full_df = fetch_data(symbol, start_date, end_date, include_market=False)
    full_df = add_indicators(full_df)
    
    train_df = full_df[full_df.index < val_split_date]
    val_df = full_df[full_df.index >= val_split_date]
    
    train_scaled, features, scaler = preprocess_data(train_df)
    val_scaled = scaler.transform(val_df[features].values)
    
    train_env = TradingEnv(train_df, train_scaled)
    val_env = TradingEnv(val_df, val_scaled)
    
    state_size = train_scaled.shape[1] + 3
    action_size = train_env.action_space.n
    
    # Custom agent with trial params
    agent = DQNAgent(state_size, action_size, lr=lr, gamma=gamma, epsilon_decay=epsilon_decay)
    # Override hidden_dim
    agent.model = QNetwork(state_size, action_size, hidden_dim=hidden_dim).to(agent.device)
    agent.target_model = QNetwork(state_size, action_size, hidden_dim=hidden_dim).to(agent.device)
    agent.update_target_network()
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)

    # Train for 5 episodes
    for e in range(5):
        state, _ = train_env.reset()
        for _ in range(len(train_df) - 31):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if done: break
        agent.update_target_network()

    # Evaluate
    val_worth = evaluate_agent(val_env, agent)
    return val_worth

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
