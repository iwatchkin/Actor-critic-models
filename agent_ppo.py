import torch
import torch.distributions
from copy import deepcopy
from network import ActorNetwork
from network import ValueFunctionNetwork


class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.98,
                 clip_ratio=0.2, action_scale=2, batch_size=64):

        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.__gamma = gamma
        self.__clip_ratio = clip_ratio
        self.__batch_size = batch_size
        self.__action_scale = action_scale
        self.__trajectories = []
        self.__actor_model = ActorNetwork(self.__state_dim, self.__action_dim)
        self.__critic_model = ValueFunctionNetwork(self.__state_dim)
        self.__actor_model_copy = deepcopy(self.__actor_model)
        self.__actor_lr = actor_lr
        self.__critic_lr = critic_lr
        self.__actor_optimizer_copy = torch.optim.Adam(self.__actor_model_copy.parameters(), lr=actor_lr)
        self.__critic_optimizer = torch.optim.SGD(self.__critic_model.parameters(), lr=critic_lr)

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            pred_action = self.__actor_model(state)
            action = self.__action_scale * pred_action

        return action

    def collect_trajectories(self, state, action, reward, done, next_state):
        self.__trajectories.append((state, action, reward, done, next_state))

    def fit(self):
        assert len(self.__trajectories) >= self.__batch_size, \
            "The length of the trajectory must be greater than the size of the mini-batch"

        num_batch = 0
        while True:
            mini_batch = self.__trajectories[:self.__batch_size]
            self.__trajectories = self.__trajectories[self.__batch_size:]

            if len(mini_batch) == 0:
                break

            states, actions, rewards, _, next_states = map(torch.FloatTensor, zip(*mini_batch))
            rewards = rewards.reshape((self.__batch_size, 1))

            old_policy = self.__actor_model(states)
            new_policy = self.__actor_model_copy(states)
            advantage_estimates = (rewards + self.__gamma * self.__critic_model(next_states)
                                   - self.__critic_model(states))
            ratio_policy = new_policy / old_policy

            actor_loss = -torch.mean(torch.min(ratio_policy * advantage_estimates, torch.clip(ratio_policy, 1 -
                                                    self.__clip_ratio, 1 + self.__clip_ratio) * advantage_estimates))
            self.__actor_optimizer_copy.zero_grad()
            actor_loss.backward()
            self.__actor_optimizer_copy.step()

            k = num_batch * self.__batch_size
            discount_rates = (torch.FloatTensor([self.__gamma ** k_t for k_t in range(k, k+self.__batch_size)])
                              .reshape((self.__batch_size, 1)))
            rewards_to_go = torch.cumsum(discount_rates * rewards, dim=1)
            critic_loss = torch.mean((self.__critic_model(states - rewards_to_go) ** 2))
            self.__critic_optimizer.zero_grad()
            critic_loss.backward()
            self.__critic_optimizer.step()

            num_batch += 1

        self.__actor_model.load_state_dict(self.__actor_model_copy.state_dict())
