from src.data import fetch_data, add_indicators, preprocess_data
from src.env import TradingEnv
from src.model import DQNAgent, QNetwork
from src.model_pc import DQNAgentPC, QNetworkPC
from datetime import datetime, timedelta
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

def evaluate_agent(env, agent):
    state, _ = env.reset()
    done = False
    old_eps = agent.epsilon
    agent.epsilon = 0
    
    while not done:
        action = agent.act(state)
        state, reward, done, truncated, info = env.step(action)
        
    agent.epsilon = old_eps
    return info['net_worth']

def train_agent(symbol="AAPL", episodes=50, batch_size=32, source="yfinance", model_type="standard", 
                lr=1e-3, gamma=0.99, hidden_dim=128, epsilon_decay=0.995, pred_alpha=0.5,
                start_years=10, val_years=2, is_search=False):
    
    # 1. Fetch and process data
    today = datetime(2026, 2, 8)
    start_date = (today - timedelta(days=365*start_years)).strftime("%Y-%m-%d")
    val_split_date = (today - timedelta(days=365*val_years)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    if not is_search:
        print(f"Fetching data for {symbol} ({start_date} to {end_date})...")
    
    full_df = fetch_data(symbol, start_date, end_date, source=source)
    if full_df is None or len(full_df) < 100: 
        return 0 if is_search else (None, None)
    full_df = add_indicators(full_df)
    
    # Split into Train and Val sets (80% Train, 20% Val)
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]
    val_df = full_df.iloc[split_idx:]
    
    train_scaled, features, scaler = preprocess_data(train_df)
    val_scaled = scaler.transform(val_df[features].values)
    
    # 2. Setup environment and agent
    train_env = TradingEnv(train_df, train_scaled, fee_rate=0.001, slippage_rate=0.0005)
    val_env = TradingEnv(val_df, val_scaled, fee_rate=0.001, slippage_rate=0.0005)
    
    state_size = train_scaled.shape[1] + 3
    action_size = train_env.action_space.n
    
    if model_type == "pc":
        agent = DQNAgentPC(state_size, action_size, lr=lr, gamma=gamma, epsilon_decay=epsilon_decay, hidden_dim=hidden_dim)
        agent.pred_alpha = pred_alpha
    else:
        agent = DQNAgent(state_size, action_size, lr=lr, gamma=gamma, epsilon_decay=epsilon_decay, hidden_dim=hidden_dim)
    
    agent.update_target_network()

    # TensorBoard setup (skip if searching)
    writer = None
    if not is_search:
        print(f"Using {model_type} Agent. Hidden Dim: {hidden_dim}")
        writer = SummaryWriter(f"runs/{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    best_val_worth = 0
    patience = 15
    no_improve_count = 0
    
    # 3. Training Loop
    if not is_search:
        print(f"Starting training: {len(train_df)} train days, {len(val_df)} val days.")
        
    for e in range(episodes):
        state, _ = train_env.reset()
        total_reward = 0
        losses = []
        pred_losses = []
        
        for time in range(len(train_df) - 31):
            action = agent.act(state)
            next_state, reward, done, truncated, info = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if agent.mem_cnt > batch_size:
                losses_batch = agent.replay(batch_size)
                if isinstance(losses_batch, tuple):
                    dqn_l, pred_l = losses_batch
                    losses.append(dqn_l)
                    pred_losses.append(pred_l)
                else:
                    losses.append(losses_batch)
                
            if done: break
        
        avg_loss = np.mean(losses) if losses else 0
        avg_pred_loss = np.mean(pred_losses) if pred_losses else 0
        agent.update_target_network()
        
        if writer:
            writer.add_scalar("Train/Reward", total_reward, e)
            writer.add_scalar("Train/NetWorth", info['net_worth'], e)
            writer.add_scalar("Train/Loss", avg_loss, e)
            if pred_losses: writer.add_scalar("Train/PredLoss", avg_pred_loss, e)
            writer.add_scalar("Train/Epsilon", agent.epsilon, e)
        
        if not is_search:
            loss_str = f"Loss: {avg_loss:.4f}"
            if pred_losses: loss_str += f" | PredLoss: {avg_pred_loss:.4f}"
            print(f"Ep {e+1}/{episodes} | Worth: {info['net_worth']:.2f} | {loss_str} | Eps: {agent.epsilon:.2f}")

        # Validation
        if (e + 1) % 5 == 0 or e == episodes - 1:
            val_worth = evaluate_agent(val_env, agent)
            if writer: writer.add_scalar("Val/NetWorth", val_worth, e)
            
            if not is_search:
                print(f"--- VALIDATION: Net Worth: {val_worth:.2f} ---")
            
            if val_worth > best_val_worth:
                best_val_worth = val_worth
                if not is_search: 
                    os.makedirs("models", exist_ok=True)
                    safe_symbol = symbol.replace("/", "_").replace("^", "IDX_")
                    agent.save(f"models/model_{safe_symbol}_best.pth")
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if not is_search and no_improve_count >= patience:
                print("Early stopping triggered.")
                break
                
    if writer: writer.close()
    
    if is_search:
        return best_val_worth
    return agent, scaler

def backtest(symbol, agent_path, model_type="standard", hidden_dim=128):
    import matplotlib.pyplot as plt
    print(f"\n--- BACKTESTING: {symbol} using {agent_path} ({model_type}) ---")
    today = datetime(2026, 2, 8)
    start_date = (today - timedelta(days=365*2)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    df = fetch_data(symbol, start_date, end_date) 
    if df is None: return
    df = add_indicators(df)
    scaled_data, features, scaler = preprocess_data(df)
    
    env = TradingEnv(df, scaled_data, fee_rate=0.001, slippage_rate=0.0005)
    state_size = scaled_data.shape[1] + 3
    action_size = env.action_space.n
    
    if model_type == "pc":
        agent = DQNAgentPC(state_size, action_size, hidden_dim=hidden_dim)
    else:
        agent = DQNAgent(state_size, action_size, hidden_dim=hidden_dim)
    
    agent.load(agent_path)
    agent.epsilon = 0.0 
    
    state, _ = env.reset()
    net_worths = [env.initial_balance]
    
    for _ in range(len(df) - 31):
        action = agent.act(state)
        state, reward, done, truncated, info = env.step(action)
        net_worths.append(info['net_worth'])
        if done: break
            
    plt.figure(figsize=(12, 6))
    plt.plot(net_worths, label=f'Agent ({model_type}) Net Worth')
    bh_prices = df['Close'].values[30:]
    bh_normalized = (bh_prices / bh_prices[0]) * env.initial_balance
    plt.plot(bh_normalized, label='Buy and Hold (Normalized)', alpha=0.7)
    
    plt.title(f"Backtest Results: {symbol} ({model_type})")
    plt.xlabel("Days")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = f"backtest_{symbol}_{model_type}.png"
    plt.savefig(save_path)
    print(f"Backtest finished. Final Net Worth: {net_worths[-1]:.2f}")
    print(f"Plot saved to {save_path}")