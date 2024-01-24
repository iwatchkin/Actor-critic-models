import random
import torch
from collections import deque
from copy import deepcopy
from network import ActorNetwork
from network import CriticNetwork


class DDPGAgent:
    def __init__(self, state_dim, action_dim, std=1, actor_lr=1e-4, critic_lr=1e-3, gamma=0.98, tau=0.02, action_scale=2, batch_size=64, memory_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.std = std
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=memory_size)
        self.action_scale = action_scale
        self.actor_model = ActorNetwork(self.state_dim, self.action_dim)
        self.critic_model = CriticNetwork(self.state_dim, self.action_dim)
        self.actor_target = deepcopy(self.actor_model)
        self.critic_target = deepcopy(self.critic_model)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=critic_lr)

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            pred_action = self.actor_model(state)
            action = torch.clip(self.action_scale * pred_action + torch.normal(mean=0, std=self.std, size=(self.action_dim,)), min=-self.action_scale,
                            max=self.action_scale)

        return action

    def __update_target_parameters(self, model, target_model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def fit(self, state, action, reward, done, next_state):
        self.replay_buffer.append((state, action, reward, done, next_state))

        if len(self.replay_buffer) >= self.batch_size:
            mini_batch = random.sample(self.replay_buffer, self.batch_size)

            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*mini_batch))

            actions = actions.reshape((self.batch_size, 1))
            rewards = rewards.reshape((self.batch_size, 1))
            dones = dones.reshape((self.batch_size, 1))

            next_actions = self.action_scale * self.actor_target(next_states).reshape((self.batch_size, 1))
            next_states_and_actions = torch.cat((next_states, next_actions), dim=1)

            target = rewards + self.gamma * (1 - dones) * self.critic_target(next_states_and_actions)

            states_and_actions = torch.cat((states, actions), dim=1)
            critic_loss = torch.mean((self.critic_model(states_and_actions).reshape((self.batch_size, 1)) - target) ** 2)
            self.__update_target_parameters(self.critic_model, self.critic_target, self.critic_optimizer, critic_loss)

            pred_actions = self.action_scale * self.actor_model(states)
            states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
            actor_loss = -torch.mean(self.critic_model(states_and_pred_actions))
            self.__update_target_parameters(self.actor_model, self.actor_target, self.actor_optimizer, actor_loss)





