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
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q = self.model(states).gather(1, actions)
        
        # Double DQN: Use model to pick action, target_model to get Q-value
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
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
