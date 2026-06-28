from src.data import fetch_data, add_indicators, preprocess_data, get_feature_list, scale_features
from src.env import TradingEnv
from src.model import DQNAgent, QNetwork
from src.model_pc import DQNAgentPC, QNetworkPC
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import random
import numpy as np
import pandas as pd

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
                lr=1e-3, gamma=0.99, hidden_dim=128, epsilon_decay=0.9999, pred_alpha=0.5,
                start_years=10, val_years=2, is_search=False, val_symbol=None, episode_length=350):

    # Accept either a single symbol or a basket. Training on several tickers
    # forces the policy to learn generalizable structure, not one stock's path.
    symbols = list(symbol) if isinstance(symbol, (list, tuple)) else [symbol]
    val_sym = val_symbol or symbols[0]

    # 1. Fetch and process data
    today = datetime(2026, 2, 8)
    start_date = (today - timedelta(days=365*start_years)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    if not is_search:
        print(f"Fetching data for {symbols} ({start_date} to {end_date})...")

    # Build the 80% training split for every symbol.
    train_dfs = []
    for sym in symbols:
        full_df = fetch_data(sym, start_date, end_date, source=source)
        if full_df is None or len(full_df) < 200:
            if not is_search:
                print(f"  Skipping {sym}: insufficient data.")
            continue
        full_df = add_indicators(full_df)
        split_idx = int(len(full_df) * 0.8)
        train_dfs.append(full_df.iloc[:split_idx])
    if not train_dfs:
        return 0 if is_search else (None, None)

    features = get_feature_list(train_dfs[0])

    # One scaler fit on the POOLED training data (stationary features are now
    # comparable across tickers), then per-symbol scaled training environments.
    pooled = np.concatenate([tdf[features].values for tdf in train_dfs], axis=0)
    scaler = StandardScaler().fit(pooled)
    train_envs = [TradingEnv(tdf, scale_features(scaler, tdf[features].values),
                             fee_rate=0.001, slippage_rate=0.0005) for tdf in train_dfs]

    # Validation: held-out (last 20%) split of the val symbol, same scaler.
    vfull = add_indicators(fetch_data(val_sym, start_date, end_date, source=source))
    val_df = vfull.iloc[int(len(vfull) * 0.8):]
    val_env = TradingEnv(val_df, scale_features(scaler, val_df[features].values),
                         fee_rate=0.001, slippage_rate=0.0005)

    state_size = len(features) + 3
    action_size = train_envs[0].action_space.n

    if model_type == "pc":
        agent = DQNAgentPC(state_size, action_size, lr=lr, gamma=gamma, epsilon_decay=epsilon_decay, hidden_dim=hidden_dim)
        agent.pred_alpha = pred_alpha
    else:
        agent = DQNAgent(state_size, action_size, lr=lr, gamma=gamma, epsilon_decay=epsilon_decay, hidden_dim=hidden_dim)

    agent.update_target_network()

    # TensorBoard setup (skip if searching)
    writer = None
    if not is_search:
        print(f"Using {model_type} Agent. Hidden Dim: {hidden_dim} | Train symbols: {symbols} | Val: {val_sym}")
        writer = SummaryWriter(f"runs/{val_sym}_{model_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    safe_symbol = val_sym.replace("/", "_").replace("^", "IDX_")
    best_val_worth = 0
    patience = 20
    no_improve_count = 0

    # 3. Training Loop
    if not is_search:
        total_days = sum(len(e.df) for e in train_envs)
        print(f"Starting training: {len(train_envs)} symbol(s), {total_days} train days, {len(val_df)} val days.")

    for e in range(episodes):
        # Sample a random symbol and a random window within it.
        env = random.choice(train_envs)
        max_start = len(env.df) - 2 - episode_length
        if max_start > env.lookback_window:
            start_step = random.randint(env.lookback_window, max_start)
        else:
            start_step = env.lookback_window
        state, _ = env.reset(start_step=start_step, episode_length=episode_length)
        total_reward = 0
        losses = []
        pred_losses = []

        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
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
            print(f"Ep {e+1}/{episodes} | Worth: {info['net_worth']:.2f} | {loss_str} | Eps: {agent.epsilon:.3f}")

        # Validation
        if (e + 1) % 5 == 0 or e == episodes - 1:
            val_worth = evaluate_agent(val_env, agent)
            if writer: writer.add_scalar("Val/NetWorth", val_worth, e)

            if not is_search:
                print(f"--- VALIDATION ({val_sym}): Net Worth: {val_worth:.2f} ---")

            if val_worth > best_val_worth:
                best_val_worth = val_worth
                if not is_search:
                    os.makedirs("models", exist_ok=True)
                    agent.save(f"models/model_{safe_symbol}_best.pth")
                    # Persist the TRAINING scaler so backtests transform (not refit)
                    # with the same statistics -> no test-set leakage.
                    with open(f"models/scaler_{safe_symbol}.pkl", "wb") as f:
                        pickle.dump(scaler, f)
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

def backtest(symbol, agent_path, model_type="standard", hidden_dim=128,
             train_cutoff=datetime(2026, 2, 8), end_date=None, lookback=30,
             scaler_symbol=None, plot=True, verbose=True):
    """Out-of-sample backtest. Returns a metrics dict (or None on failure).

    Evaluates strictly on data AFTER `train_cutoff` (data the model never saw)
    and normalizes it with the scaler fitted during training (transform, not
    refit) -> no test-set leakage. A `lookback`-day lead-in before the cutoff is
    fetched only to warm up indicators and the first observation window; no
    pre-cutoff step is scored. `scaler_symbol` lets one model/scaler be applied
    to any ticker (the features are symbol-agnostic).
    """
    import matplotlib.pyplot as plt
    if verbose:
        print(f"\n--- BACKTESTING (out-of-sample): {symbol} using {agent_path} ({model_type}) ---")

    safe_scaler = (scaler_symbol or symbol).replace("/", "_").replace("^", "IDX_")
    scaler_path = f"models/scaler_{safe_scaler}.pkl"
    safe_symbol = symbol.replace("/", "_").replace("^", "IDX_")
    if not os.path.exists(scaler_path):
        print(f"ERROR: training scaler not found at {scaler_path}. Train the model first.")
        return None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Fetch [cutoff - warmup, end]; warmup covers the longest indicator window
    # (Sma_50) plus the lookback so cutoff-day features/observations are real.
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    fetch_start = (train_cutoff - timedelta(days=120 + lookback)).strftime("%Y-%m-%d")

    df = fetch_data(symbol, fetch_start, end_date)
    if df is None:
        return None
    df = add_indicators(df)
    df.index = pd.to_datetime(df.index)

    # First row strictly after the cutoff, then back up `lookback` for the lead-in.
    cutoff_pos = int(np.searchsorted(df.index.values,
                                     np.datetime64(train_cutoff) + np.timedelta64(1, "D")))
    start_pos = cutoff_pos - lookback
    if start_pos < 0 or len(df) - start_pos < lookback + 5:
        print(f"ERROR: not enough OOS data for {symbol}.")
        return None
    oos = df.iloc[start_pos:].copy()

    features = get_feature_list(oos)
    scaled_data = scale_features(scaler, oos[features].values)

    env = TradingEnv(oos, scaled_data, fee_rate=0.001, slippage_rate=0.0005,
                     lookback_window=lookback)
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
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, truncated, info = env.step(action)
        net_worths.append(info['net_worth'])

    nw = np.array(net_worths)
    prices = oos['Close'].values[lookback:]
    bh = prices / prices[0] * env.initial_balance

    def _stats(s):
        rets = np.diff(s) / s[:-1]
        total = (s[-1] / s[0] - 1) * 100
        sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if rets.std() > 0 else 0.0
        peak = np.maximum.accumulate(s)
        mdd = ((s - peak) / peak).min() * 100
        return total, sharpe, mdd

    a_tot, a_sh, a_dd = _stats(nw)
    b_tot, b_sh, b_dd = _stats(bh)
    if verbose:
        print(f"OOS window: {oos.index[lookback].date()} -> {oos.index[-1].date()} "
              f"({len(oos) - lookback} trading days)")
        print(f"{'':18}{'Agent':>11}{'Buy&Hold':>11}")
        print(f"{'Total return %':18}{a_tot:>11.2f}{b_tot:>11.2f}")
        print(f"{'Sharpe (ann.)':18}{a_sh:>11.2f}{b_sh:>11.2f}")
        print(f"{'Max drawdown %':18}{a_dd:>11.2f}{b_dd:>11.2f}")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(oos.index[lookback:], nw, label=f'Agent ({model_type}) {a_tot:+.1f}%')
        plt.plot(oos.index[lookback:], bh, label=f'Buy & Hold {b_tot:+.1f}%', alpha=0.7)
        plt.title(f"Out-of-Sample Backtest: {symbol} ({model_type})")
        plt.xlabel("Date")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = f"backtest_{safe_symbol}_{model_type}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"Backtest finished. Final Net Worth: {nw[-1]:.2f}\nPlot saved to {save_path}")

    return {"symbol": symbol, "days": len(oos) - lookback,
            "agent_return": a_tot, "bh_return": b_tot, "excess": a_tot - b_tot,
            "agent_sharpe": a_sh, "bh_sharpe": b_sh,
            "agent_mdd": a_dd, "bh_mdd": b_dd, "beat_bh": a_tot > b_tot}
