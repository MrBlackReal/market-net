import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    """Brain of the agent (Dueling LSTM DQN)."""
    def __init__(self, input_dim, action_size, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Advantage stream
        self.adv_fc1 = nn.Linear(hidden_dim, 64)
        self.adv_fc2 = nn.Linear(64, action_size)
        
        # Value stream
        self.val_fc1 = nn.Linear(hidden_dim, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, lookback, input_dim)
        _, (hn, _) = self.lstm(x)
        x = hn[-1] # Take last hidden state
        
        adv = F.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)
        
        val = F.relu(self.val_fc1(x))
        val = self.val_fc2(val)
        
        # Combine Value and Advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return val + (adv - adv.mean(dim=1, keepdim=True))

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=5000, hidden_dim=128):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # High-performance NumPy Buffer
        self.memory_size = memory_size
        self.mem_ptr = 0
        self.mem_cnt = 0
        
        # Pre-allocate memory for speed (assuming lookback=30)
        lookback = 30
        self.state_mem = np.zeros((memory_size, lookback, state_size), dtype=np.float32)
        self.next_state_mem = np.zeros((memory_size, lookback, state_size), dtype=np.float32)
        self.action_mem = np.zeros(memory_size, dtype=np.int64)
        self.reward_mem = np.zeros(memory_size, dtype=np.float32)
        self.done_mem = np.zeros(memory_size, dtype=np.bool_)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size, hidden_dim=hidden_dim).to(self.device)
        self.target_model = QNetwork(state_size, action_size, hidden_dim=hidden_dim).to(self.device)
        
        # Enable oneDNN/MKLDNN: ~3.4x faster LSTM forward/backward on this CPU.
        torch.backends.mkldnn.enabled = True

        # weight_decay (L2) regularizes the net to curb overfitting -> better OOS.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        idx = self.mem_ptr % self.memory_size
        self.state_mem[idx] = state
        self.next_state_mem[idx] = next_state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.done_mem[idx] = done
        self.mem_ptr += 1
        self.mem_cnt = min(self.mem_cnt + 1, self.memory_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            # Standard precision for stability
            action_values = self.model(state_t)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if self.mem_cnt < batch_size:
            return
        
        # Fast sampling using NumPy
        batch_indices = np.random.choice(self.mem_cnt, batch_size, replace=False)
        
        states = torch.from_numpy(self.state_mem[batch_indices]).to(self.device)
        next_states = torch.from_numpy(self.next_state_mem[batch_indices]).to(self.device)
        actions = torch.from_numpy(self.action_mem[batch_indices]).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(self.reward_mem[batch_indices]).unsqueeze(1).to(self.device)
        dones = torch.from_numpy(self.done_mem[batch_indices]).unsqueeze(1).to(self.device).float()
        
        # Standard precision training
        current_q = self.model(states).gather(1, actions)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_network()