import argparse
import matplotlib.pyplot as plt
from src.train import train_agent
from src.data import fetch_data, add_indicators, preprocess_data
from src.env import TradingEnv
from src.model import DQNAgent
import torch
from datetime import datetime, timedelta
import numpy as np

def backtest(symbol, agent_path):
    print(f"Backtesting {symbol} using {agent_path}...")
    today = datetime(2026, 2, 8)
    # Test on the last 2 years (Out of sample if we trained on the full 10y, 
    # but our train script splits it. This is a final evaluation.)
    start_date = (today - timedelta(days=365*2)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    df = fetch_data(symbol, start_date, end_date) 
    if df is None: return
    df = add_indicators(df)
    scaled_data, features, scaler = preprocess_data(df)
    
    # Use realistic settings for backtest too
    env = TradingEnv(df, scaled_data, fee_rate=0.001, slippage_rate=0.0005)
    state_size = scaled_data.shape[1] + 3
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    try:
        agent.load(agent_path)
    except FileNotFoundError:
        print(f"Error: Model file {agent_path} not found.")
        return
        
    agent.epsilon = 0.0 
    
    state, _ = env.reset()
    net_worths = [env.initial_balance]
    
    for _ in range(len(df) - 31):
        action = agent.act(state)
        state, reward, done, truncated, info = env.step(action)
        net_worths.append(info['net_worth'])
        if done:
            break
            
    plt.figure(figsize=(12, 6))
    plt.plot(net_worths, label='Agent Net Worth')
    
    # Normalize Buy & Hold to start at the same initial balance
    bh_prices = df['Close'].values[30:]
    bh_normalized = (bh_prices / bh_prices[0]) * env.initial_balance
    plt.plot(bh_normalized, label='Buy and Hold (Normalized)', alpha=0.7)
    
    plt.title(f"Backtest Results: {symbol} (Realistic Fees/Slippage)")
    plt.xlabel("Days")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("backtest_result.png")
    print(f"Backtest finished. Final Net Worth: {net_worths[-1]:.2f}")
    print("Backtest results saved to backtest_result.png")

if __name__ == "__main__":
    # Optimize CPU usage
    if not torch.cuda.is_available():
        import os
        # Set threads to match physical cores for best performance in RL
        num_threads = os.cpu_count() // 2 if os.cpu_count() > 4 else os.cpu_count()
        torch.set_num_threads(num_threads)
        print(f"CPU Optimization: Using {num_threads} threads.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--source", choices=["yfinance", "stooq", "binance"], default="yfinance")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model", default="model_AAPL.pth")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_agent(symbol=args.symbol, episodes=args.episodes, source=args.source)
    else:
        # For backtest, we'll use yfinance by default but could also make it an arg
        backtest(args.symbol, args.model)
