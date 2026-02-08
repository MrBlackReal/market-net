import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class QNetworkPC(nn.Module):
    """Brain of the agent (Dueling LSTM DQN + Predictive Head)."""
    def __init__(self, input_dim, action_size, hidden_dim=128):
        super(QNetworkPC, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Advantage stream
        self.adv_fc1 = nn.Linear(hidden_dim, 64)
        self.adv_fc2 = nn.Linear(64, action_size)
        
        # Value stream
        self.val_fc1 = nn.Linear(hidden_dim, 64)
        self.val_fc2 = nn.Linear(64, 1)

        # Predictive Head: Predicts the next market state (input_dim)
        # This implements the "Predictive Coding" principle by building a generative model of the input.
        self.pred_fc1 = nn.Linear(hidden_dim, 64)
        self.pred_fc2 = nn.Linear(64, input_dim)

    def forward(self, x):
        # x shape: (batch, lookback, input_dim)
        _, (hn, _) = self.lstm(x)
        x_latent = hn[-1] # Take last hidden state
        
        # 1. DQN Outputs (Dueling)
        adv = F.relu(self.adv_fc1(x_latent))
        adv = self.adv_fc2(adv)
        val = F.relu(self.val_fc1(x_latent))
        val = self.val_fc2(val)
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        
        # 2. Predictive Output (Next State Prediction)
        pred_next = F.relu(self.pred_fc1(x_latent))
        pred_next = self.pred_fc2(pred_next)
        
        return q_values, pred_next

class DQNAgentPC:
    """Predictive Coding Hybrid Agent."""
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=5000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.pred_alpha = 0.5 # Weight for the predictive loss (Surprise minimization)
        
        # High-performance NumPy Buffer
        self.memory_size = memory_size
        self.mem_ptr = 0
        self.mem_cnt = 0
        
        lookback = 30
        self.state_mem = np.zeros((memory_size, lookback, state_size), dtype=np.float32)
        self.next_state_mem = np.zeros((memory_size, lookback, state_size), dtype=np.float32)
        self.action_mem = np.zeros(memory_size, dtype=np.int64)
        self.reward_mem = np.zeros(memory_size, dtype=np.float32)
        self.done_mem = np.zeros(memory_size, dtype=np.bool_)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetworkPC(state_size, action_size).to(self.device)
        self.target_model = QNetworkPC(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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
            q_values, _ = self.model(state_t)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if self.mem_cnt < batch_size:
            return 0, 0
        
        batch_indices = np.random.choice(self.mem_cnt, batch_size, replace=False)
        
        states = torch.from_numpy(self.state_mem[batch_indices]).to(self.device)
        next_states = torch.from_numpy(self.next_state_mem[batch_indices]).to(self.device)
        actions = torch.from_numpy(self.action_mem[batch_indices]).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(self.reward_mem[batch_indices]).unsqueeze(1).to(self.device)
        dones = torch.from_numpy(self.done_mem[batch_indices]).unsqueeze(1).to(self.device).float()
        
        # 1. DQN Loss
        current_q, pred_state = self.model(states)
        current_q = current_q.gather(1, actions)
        
        with torch.no_grad():
            next_q, _ = self.model(next_states)
            next_actions = next_q.argmax(1).unsqueeze(1)
            target_q_all, _ = self.target_model(next_states)
            target_q = target_q_all.gather(1, next_actions)
            expected_q = rewards + (self.gamma * target_q * (1 - dones))
        
        dqn_loss = F.mse_loss(current_q, expected_q)
        
        # 2. Predictive Loss (Internal Model of Market Physics)
        # Target is the LAST step of the next state sequence
        true_next_step = next_states[:, -1, :] 
        pred_loss = F.mse_loss(pred_state, true_next_step)
        
        # 3. Total Loss: Combined reward maximization and surprise minimization
        total_loss = dqn_loss + (self.pred_alpha * pred_loss)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return dqn_loss.item(), pred_loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_network()
