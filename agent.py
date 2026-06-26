import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
from model import DQN

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.uint8)
        )
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim=168, action_dim=9, lr=3e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=20000, buffer_capacity=100000, batch_size=128, tau=0.001, min_replay_size=2000):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.min_replay_size = min_replay_size
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Device selection (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Online and Target networks
        self.online_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() # Target net is in evaluation mode
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss() # Huber loss
        self.memory = ReplayBuffer(buffer_capacity)
        
    def select_action(self, state, evaluate=False):
        """
        Select action using epsilon-greedy strategy.
        If evaluate=True, always select the greedy action.
        """
        # Update epsilon
        if not evaluate:
            self.steps_done += 1
            # Exponential decay
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           np.exp(-1.0 * self.steps_done / self.epsilon_decay)
            
        if evaluate or random.random() > self.epsilon:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
                return q_values.argmax(dim=1).item()
        else:
            return random.randint(0, self.action_dim - 1)
            
    def update(self):
        """
        Performs one gradient descent update step.
        """
        if len(self.memory) < self.min_replay_size:
            return None
            
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)
        
        # Current Q-values: online network calculates Q(s, a)
        current_q = self.online_net(states_t).gather(1, actions_t)
        
        # Target Q-values: Double DQN (DDQN) to prevent overestimation bias
        with torch.no_grad():
            next_state_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_state_actions)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

            
        # Compute Loss
        loss = self.loss_fn(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft update of target network weights
        # target = tau * online + (1 - tau) * target
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
            
        return loss.item()
        
    def save(self, filepath, metadata=None):
        checkpoint = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }
        if metadata is not None:
            checkpoint['metadata'] = metadata
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        
        if 'target_net_state_dict' in checkpoint:
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if 'steps_done' in checkpoint:
            self.steps_done = checkpoint['steps_done']
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           np.exp(-1.0 * self.steps_done / self.epsilon_decay)
            print(f"Model loaded from {filepath}. Epsilon set to {self.epsilon:.3f}")
        else:
            self.steps_done = 0
            self.epsilon = 0.0
            print(f"Model loaded from {filepath} (raw weights). Epsilon set to {self.epsilon:.3f}")

        return checkpoint.get('metadata', None)
