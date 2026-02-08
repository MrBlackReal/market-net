import numpy as np
import torch
from src.data import fetch_data, add_indicators, preprocess_data
from src.env import TradingEnv
from src.model import DQNAgent
from src.model_pc import DQNAgentPC
from datetime import datetime, timedelta
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

def train_agent(symbol="AAPL", episodes=50, batch_size=32, source="yfinance", model_type="standard"):
    # 1. Fetch and process data (10 year window)
    today = datetime(2026, 2, 8)
    start_date = (today - timedelta(days=365*10)).strftime("%Y-%m-%d")
    val_split_date = (today - timedelta(days=365*2)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    print(f"Fetching data for {symbol} ({start_date} to {end_date})...")
    full_df = fetch_data(symbol, start_date, end_date, source=source)
    if full_df is None: return None, None
    full_df = add_indicators(full_df)
    
    # Split into Train and Val sets
    train_df = full_df[full_df.index < val_split_date]
    val_df = full_df[full_df.index >= val_split_date]
    
    train_scaled, features, scaler = preprocess_data(train_df)
    val_scaled = scaler.transform(val_df[features].values)
    
    # 2. Setup environment and agent
    train_env = TradingEnv(train_df, train_scaled, fee_rate=0.001, slippage_rate=0.0005)
    val_env = TradingEnv(val_df, val_scaled, fee_rate=0.001, slippage_rate=0.0005)
    
    state_size = train_scaled.shape[1] + 3
    action_size = train_env.action_space.n
    
    if model_type == "pc":
        print("Using Predictive Coding (PC) Agent.")
        agent = DQNAgentPC(state_size, action_size)
    else:
        print("Using Standard D3QN Agent.")
        agent = DQNAgent(state_size, action_size)
    
    # TensorBoard setup
    writer = SummaryWriter(f"runs/{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    
    best_val_worth = 0
    patience = 10
    no_improve_count = 0
    
    # 3. Training Loop
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
        
        # Log training metrics
        writer.add_scalar("Train/Reward", total_reward, e)
        writer.add_scalar("Train/NetWorth", info['net_worth'], e)
        writer.add_scalar("Train/Loss", avg_loss, e)
        if pred_losses:
            writer.add_scalar("Train/PredLoss", avg_pred_loss, e)
        writer.add_scalar("Train/Epsilon", agent.epsilon, e)
        
        loss_str = f"Loss: {avg_loss:.4f}"
        if pred_losses:
            loss_str += f" | PredLoss: {avg_pred_loss:.4f}"
            
        print(f"Ep {e+1}/{episodes} | Worth: {info['net_worth']:.2f} | {loss_str} | Eps: {agent.epsilon:.2f}")

        # Validation every 5 episodes
        if (e + 1) % 5 == 0:
            val_worth = evaluate_agent(val_env, agent)
            writer.add_scalar("Val/NetWorth", val_worth, e)
            print(f"--- VALIDATION: Net Worth: {val_worth:.2f} ---")
            
            # Early Stopping check
            if val_worth > best_val_worth:
                best_val_worth = val_worth
                agent.save(f"model_{symbol}_best.pth")
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                print("Early stopping triggered.")
                break
                
    writer.close()
    return agent, scaler

def evaluate_agent(env, agent):
    state, _ = env.reset()
    done = False
    # Use epsilon=0 for evaluation
    old_eps = agent.epsilon
    agent.epsilon = 0
    
    while not done:
        action = agent.act(state)
        state, reward, done, truncated, info = env.step(action)
        
    agent.epsilon = old_eps # Restore epsilon
    return info['net_worth']

if __name__ == "__main__":
    train_agent()
