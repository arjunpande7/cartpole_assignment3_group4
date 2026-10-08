import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        # stores the last `capacity` transitions, the oldest ones are removed
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        terminated_s = []

        for state, action, reward, next_state, terminated in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminated_s.append(terminated)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        terminated_s = torch.tensor(terminated_s, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, terminated_s

    def __len__(self):
        return len(self.memory)