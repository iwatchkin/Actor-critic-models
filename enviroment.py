import gym
import warnings
from agent_ddpg import DDPGAgent
from agent_ppo import PPOAgent
from plot_history_rewards import plot_history_rewards
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

env = gym.make('Pendulum-v1', g=9.81)
#episodes, trajectory_len = 100, 200

#agent = DDPGAgent(state_dim=3, action_dim=1)
#history_rewards_ddpg = []

# print("DDPG algorithm\n")
# for episode in range(episodes):
#
#     total_reward = 0
#     state = env.reset()[0]
#     for _ in range(trajectory_len):
#         action = agent.get_action(state)
#         next_state, reward, done, _, _ = env.step(action)
#         total_reward += reward
#
#         agent.fit(state, action, reward, done, next_state)
#
#         if done:
#             break
#
#         state = next_state
#     history_rewards_ddpg.append(total_reward / trajectory_len)
#     print(f"\nEpisode {episode + 1}, total reward {total_reward / trajectory_len}")
#
# print(f"\nBest reward: {max(history_rewards_ddpg)}")
# print(f"Average reward: {sum(history_rewards_ddpg) / len(history_rewards_ddpg)}")

episodes, batch_size, num_batch = 100, 200, 10
trajectory_len = batch_size * num_batch

agent = PPOAgent(state_dim=3, action_dim=1, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
                 clip_ratio=0.2, action_scale=2, batch_size=batch_size)
history_rewards_ppo = []


print("\nPPO algorithm")
for episode in range(episodes):

    total_reward = 0
    state = env.reset()[0]
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
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
plot_history_rewards(history_rewards_ppo)
