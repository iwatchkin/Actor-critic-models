import matplotlib.pyplot as plt


def plot_history_rewards(history_rewards):
    plt.plot(history_rewards)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('History of learning')
    plt.grid()
    plt.show()
