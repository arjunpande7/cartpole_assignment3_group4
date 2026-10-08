import random
import torch
import torch.nn as nn
import torch.optim as optim

from src.network import QNetwork
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, config):
        # from  config file
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.grad_clip = config["grad_clip"]
        self.action_dim = 2
        self.q_network = QNetwork(state_dim=4, action_dim=2)
        self.target_network = QNetwork(state_dim=4, action_dim=2)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config["learning_rate"])
        self.loss_function = nn.SmoothL1Loss()  
        self.buffer = ReplayBuffer(config["buffer_size"])

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def learn(self):
        states, actions, rewards, next_states, terminateds = self.buffer.sample(self.batch_size)
        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * max_next_q * (1 - terminateds)

        loss = self.loss_function(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)  
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)