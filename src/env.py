import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom trading environment for Reinforcement Learning.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """
    def __init__(self, df, scaled_data, initial_balance=10000, lookback_window=30, fee_rate=0.001, slippage_rate=0.0005):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.scaled_data = scaled_data
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.fee_rate = fee_rate # e.g., 0.1% fee
        self.slippage_rate = slippage_rate # e.g., 0.05% slippage
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: market data window + portfolio state
        # Portfolio state: [balance, position, current_price]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window, scaled_data.shape[1] + 3), 
            dtype=np.float32
        )
        
        # Pre-allocate observation buffer to avoid expensive hstack in the loop
        self.obs_buffer = np.zeros((lookback_window, scaled_data.shape[1] + 3), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None, start_step=None, episode_length=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.entry_price = 0.0          # cost basis of the current position

        # Random-window support: start anywhere (after warmup) and run a fixed
        # number of steps. Replaying the same full series every episode invites
        # memorization; random windows force the policy to generalize.
        if start_step is None:
            self.current_step = self.lookback_window
        else:
            self.current_step = int(np.clip(start_step, self.lookback_window, len(self.df) - 2))
        if episode_length is None:
            self.end_step = len(self.df) - 1
        else:
            self.end_step = min(len(self.df) - 1, self.current_step + int(episode_length))

        # Differential Sharpe ratio (DSR) running moments.
        self.dsr_A = 0.0
        self.dsr_B = 0.0
        self.dsr_n = 0

        self.history = []
        return self._get_observation(), {}

    def _get_observation(self):
        # Fast slice of market data
        self.obs_buffer[:, :self.scaled_data.shape[1]] = self.scaled_data[self.current_step - self.lookback_window : self.current_step]

        # Portfolio state (all stationary: fractions + relative PnL, normalized
        # by current net worth rather than the absolute starting price).
        current_price = self.df['Close'].iloc[self.current_step]
        net = self.balance + self.shares_held * current_price
        cash_frac = self.balance / (net + 1e-9)
        position_frac = (self.shares_held * current_price) / (net + 1e-9)
        unrealized_pnl = (current_price / self.entry_price - 1.0) if (self.shares_held > 0 and self.entry_price > 0) else 0.0

        # Vectorized fill for portfolio state columns
        self.obs_buffer[:, -3:] = [cash_frac, position_frac, unrealized_pnl]

        return self.obs_buffer.copy()

    def _dsr_reward(self, R, eta=0.02):
        """Differential Sharpe ratio: rewards return per unit of risk online.

        Encourages high, low-variance returns instead of just riding the trend.
        Guarded against the early-warmup zero-variance singularity and clipped.
        """
        self.dsr_n += 1
        dA = R - self.dsr_A
        dB = R * R - self.dsr_B
        var = self.dsr_B - self.dsr_A * self.dsr_A
        if self.dsr_n <= 5 or var <= 1e-8:
            reward = 0.0                       # warmup / degenerate variance
        else:
            reward = (self.dsr_B * dA - 0.5 * self.dsr_A * dB) / (var ** 1.5)
        # Update running moments (EMA)
        self.dsr_A += eta * dA
        self.dsr_B += eta * dB
        return float(np.clip(reward, -1.0, 1.0))

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        prev_net_worth = self.net_worth
        
        # Execute trade
        if action == 1: # Buy
            # Apply slippage (buy at a slightly higher price)
            execution_price = current_price * (1 + self.slippage_rate)
            if self.balance > execution_price:
                shares_to_buy = self.balance // execution_price
                cost = shares_to_buy * execution_price
                fee = cost * self.fee_rate

                if self.balance >= (cost + fee) and shares_to_buy > 0:
                    self.shares_held += shares_to_buy
                    self.balance -= (cost + fee)
                    self.entry_price = execution_price   # record cost basis

        elif action == 2: # Sell
            # Apply slippage (sell at a slightly lower price)
            execution_price = current_price * (1 - self.slippage_rate)
            if self.shares_held > 0:
                revenue = self.shares_held * execution_price
                fee = revenue * self.fee_rate
                self.balance += (revenue - fee)
                self.shares_held = 0
                self.entry_price = 0.0

        self.current_step += 1
        self.net_worth = self.balance + self.shares_held * current_price

        # Risk-adjusted reward: differential Sharpe ratio of net-worth returns.
        pct_change = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-9)
        reward = self._dsr_reward(pct_change)

        done = self.current_step >= self.end_step
        truncated = False
        
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, truncated, {"net_worth": self.net_worth}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held}")
