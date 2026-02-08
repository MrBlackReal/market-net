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
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.current_step = self.lookback_window
        
        self.history = []
        return self._get_observation(), {}

    def _get_observation(self):
        # Window of market data
        window = self.scaled_data[self.current_step - self.lookback_window : self.current_step]
        
        # Portfolio state (normalized roughly)
        current_price = self.df['Close'].iloc[self.current_step]
        portfolio_state = np.array([
            [self.balance / self.initial_balance, 
             self.shares_held * current_price / self.initial_balance,
             current_price / self.df['Close'].iloc[0]]
        ] * self.lookback_window) # Repeat for each step in window
        
        # Concatenate market data and portfolio state
        obs = np.hstack((window, portfolio_state))
        return obs.astype(np.float32)

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
                
                if self.balance >= (cost + fee):
                    self.shares_held += shares_to_buy
                    self.balance -= (cost + fee)
        
        elif action == 2: # Sell
            # Apply slippage (sell at a slightly lower price)
            execution_price = current_price * (1 - self.slippage_rate)
            if self.shares_held > 0:
                revenue = self.shares_held * execution_price
                fee = revenue * self.fee_rate
                self.balance += (revenue - fee)
                self.shares_held = 0
                
        self.current_step += 1
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Reward calculation:
        # 1. Base reward: percentage change in net worth
        pct_change = (self.net_worth - prev_net_worth) / prev_net_worth
        
        # 2. Risk penalty: Penalize if net worth is significantly below recent peaks (drawdown)
        # For simplicity, we'll use a small penalty for any negative step
        reward = pct_change
        if pct_change < 0:
            reward *= 1.2 # Slightly amplify the pain of losses to encourage caution
            
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, truncated, {"net_worth": self.net_worth}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held}")
