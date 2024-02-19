import matplotlib.pyplot as plt
import pandas as pd


def plot_history_rewards(history_rewards, n=3, label=None):
    moving_average_reward = pd.Series(history_rewards).rolling(window=n).mean().iloc[n - 1:].values
    plt.plot(moving_average_reward, label=label)
    plt.xlabel('Episode')
    plt.ylabel('Moving average of the reward')
    plt.title('History of learning')
    plt.legend()
    plt.show()


