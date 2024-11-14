import numpy as np
import matplotlib.pyplot as plt

ma_path="saved_models/Random_Multi/ma_reward.txt"
reward_path="saved_models/Random_Multi/reward.txt"
# multiagent-envs-ML/saved_models/Random_Multi/reward_20241020-224444.txt
reward = np.loadtxt(reward_path)
ma_reward = np.loadtxt(ma_path)
plt.plot(reward, label = "reward")
plt.plot(ma_reward, label="ma_reward")
plt.legend()
plt.grid()
plt.show()