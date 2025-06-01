import numpy as np
import os

src_folder = './npy/r_qlearning'

# 載入原始資料
agent_reward = np.load(os.path.join(src_folder, 'r_qlearning_agent_reward_per_iteration.npy'))  # (runs, episodes, agents)
total_reward = np.load(os.path.join(src_folder, 'r_qlearning_total_reward_per_iteration.npy'))  # (runs, episodes)

# ====== 轉換為 epsilon_greedy 的格式 ======

# 平均所有 runs 與 agents
agent_reward_mean = np.mean(agent_reward, axis=(0, 2), keepdims=True)  # shape = (1, episodes)
agent_cum_reward = np.cumsum(agent_reward_mean, axis=1)               # shape = (1, episodes)

# 對每集的總 reward 做 runs 平均（維度從 (runs, episodes) → (episodes,)）
total_reward_mean = np.mean(total_reward, axis=0)  # shape = (episodes,)
total_cum_reward = np.cumsum(total_reward_mean)    # shape = (episodes,)

# ====== 儲存新檔案（或覆蓋） ======
np.save(os.path.join(src_folder, 'r_qlearning_agent_reward_per_iteration_avg.npy'), agent_reward_mean)
np.save(os.path.join(src_folder, 'r_qlearning_agent_cumulative_reward_avg.npy'), agent_cum_reward)
np.save(os.path.join(src_folder, 'r_qlearning_total_reward_per_iteration_avg.npy'), total_reward_mean)
np.save(os.path.join(src_folder, 'r_qlearning_total_cumulative_reward_avg.npy'), total_cum_reward)

print("已轉換為 epsilon_greedy 可用的格式。")