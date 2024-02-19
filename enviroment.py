import gym
import warnings
from agent_ppo import PPOAgent
from plot_history_rewards import plot_history_rewards
import os


class Environment:
    def __init__(self, env):
        self.env = env

    def get_initial_state(self):
        return self.env.reset()[0]

    def step_in_environment(self, action):
        return self.env.step(action)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

env = gym.make('Pendulum-v1', g=9.81)
environment = Environment(env)

episodes, batch_size, num_batch = 100, 200, 10
trajectory_len = batch_size * num_batch

agent = PPOAgent(state_dim=3, action_dim=1, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
                 clip_ratio=0.2, action_scale=2, batch_size=batch_size)
history_rewards_ppo = []

print("\nPPO algorithm")
for episode in range(episodes):

    total_reward = 0
    state = environment.get_initial_state()
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        next_state, reward, done, _, _ = environment.step_in_environment(action)
        total_reward += reward
        agent.collect_trajectories(state, action, reward, done, next_state)

        if done:
            break

        state = next_state

    agent.fit()

    history_rewards_ppo.append(total_reward / trajectory_len)
    print(f"\nEpisode {episode + 1}, total reward {total_reward / trajectory_len}")

print(f"\nBest reward: {max(history_rewards_ppo)}")
print(f"Average reward: {sum(history_rewards_ppo) / len(history_rewards_ppo)}")
plot_history_rewards(history_rewards_ppo, n=3, label="PPO")



