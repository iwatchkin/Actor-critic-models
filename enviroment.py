import gym
import warnings
from agent_ddpg import DDPGAgent
from plot_history_rewards import plot_history_rewards
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

episodes = 100
trajectory_len = 200

env = gym.make('Pendulum-v1', g=9.81)
agent = DDPGAgent(state_dim=3, action_dim=1)

history_rewards_ddpg = []

for episode in range(episodes):

    total_reward = 0
    state = env.reset()[0]
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        agent.fit(state, action, reward, done, next_state)

        if done:
            break

        state = next_state
    history_rewards_ddpg.append(total_reward)
    print(f"Episode {episode + 1}, total reward {total_reward}")

print(f"\nBest reward: {max(history_rewards_ddpg)}")
print(f"Average reward: {sum(history_rewards_ddpg) / len(history_rewards_ddpg)}")

plot_history_rewards(history_rewards_ddpg)
