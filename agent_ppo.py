import random
import torch
from collections import deque
from network import ActorNetwork
from network import CriticNetwork


class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma= 0.98, clip_ratio=0.2, action_scale=2, batch_size=64, memory_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=memory_size)
        self.action_scale = action_scale
        self.actor_model = ActorNetwork(self.state_dim, self.action_dim)
        self.critic_model = CriticNetwork(self.state_dim, self.action_dim)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=critic_lr)

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action = self.actor_model(state)

        return action

    def fit(self):
        pass




