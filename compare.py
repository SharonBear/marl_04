import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
# os.makedirs("./comparison", exist_ok=True)

# === Load Accuracy Data ===
lrs_mean = np.load('./npy/lrs/lrs_avg_mean.npy')
lrs_std = np.load('./npy/lrs/lrs_avg_std.npy')

ord_mean = np.load('./npy/ord/ord_avg_mean.npy')
ord_std = np.load('./npy/ord/ord_avg_std.npy')
# greedy
greedy_mean = np.load('./npy/epsilon_greedy/epsilon_greedy_avg_mean.npy')
greedy_std = np.load('./npy/epsilon_greedy/epsilon_greedy_avg_std.npy')
#restartQlearning
# rq_mean = np.load('./npy/r_qlearning/qlearning_ucb_restart_avg_mean.npy')
# rq_std = np.load('./npy/r_qlearning/qlearning_ucb_restart_avg_std.npy')
ql_mean = np.load('./npy/qlearning_ucb/qlearning_ucb_avg_mean.npy')
ql_std = np.load('./npy/qlearning_ucb/qlearning_ucb_avg_std.npy')


iters = range(len(lrs_mean))
print(len(lrs_mean))
print(len(ord_mean))
print(len(greedy_mean))
# print(len(rq_mean))
print(len(ql_mean))

# === Plot 1: Accuracy Mean + Std ===
plt.figure(figsize=(8, 5))
plt.plot(iters, lrs_mean, label='Policy Gradient lrs', color='blue')
plt.fill_between(iters, lrs_mean - lrs_std, lrs_mean + lrs_std, color='blue', alpha=0.3)

plt.plot(iters, ord_mean, label='Policy Gradient ord', color="#BE77FF")
plt.fill_between(iters, ord_mean - ord_std, ord_mean + ord_std, color="#BE77FF" , alpha=0.3)

plt.plot(iters, ql_mean, label='Q-Learning UCB', color='green')
plt.fill_between(iters, ql_mean - ql_std, ql_mean + ql_std, color='green', alpha=0.3)

plt.plot(iters, greedy_mean, label='Epsilon Greedy', color='orange')
plt.fill_between(iters, greedy_mean - greedy_std, greedy_mean + greedy_std, color='orange', alpha=0.3)
# plt.plot(iters, rq_mean, label='Restart Q-Learning UCB', color='red')
# plt.fill_between(iters, rq_mean - rq_std, rq_mean + rq_std, color='red', alpha=0.3)


plt.xlabel("Iterations")
plt.ylabel("L1 Accuracy")
plt.title("Mean L1 Accuracy Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("./pic/comparison/l1_accuracy_comparison.png", dpi=300)
plt.close()

# === Helper Function ===
def load_reward_data(algo):
    agent_reward = np.load(f'./npy/{algo}/{algo}_agent_reward_per_iteration.npy')
    agent_cum_reward = np.load(f'./npy/{algo}/{algo}_agent_cumulative_reward.npy')
    total_reward = np.load(f'./npy/{algo}/{algo}_total_reward_per_iteration.npy')
    total_cum_reward = np.load(f'./npy/{algo}/{algo}_total_cumulative_reward.npy')
    return agent_reward, agent_cum_reward, total_reward, total_cum_reward

algos = {
    "qlearning_ucb": "Q-Learning UCB",
    "ord": "Policy Gradient ORD",
    "lrs": "Policy Gradient LRS",
    "epsilon_greedy": "Epsilon Greedy",
}

# === Plot 2: Per-Agent Reward per Iteration ===
plt.figure(figsize=(8, 6))
for algo, label in algos.items():
    agent_reward, _, _, _ = load_reward_data(algo)
    mean_per_agent = np.mean(agent_reward, axis=0)
    plt.plot(iters, mean_per_agent, label=label)
plt.xlabel("Iteration")
plt.ylabel("Per-Agent Reward")
plt.title("Per-Agent Reward per Iteration")
plt.legend()
plt.tight_layout()
plt.savefig("./pic/comparison/agent_reward_per_iteration_comparison.png", dpi=300)
plt.close()

# === Plot 3: Per-Agent Cumulative Reward ===
plt.figure(figsize=(8, 6))
for algo, label in algos.items():
    _, agent_cum_reward, _, _ = load_reward_data(algo)
    mean_cum_per_agent = np.mean(agent_cum_reward, axis=0)
    plt.plot(iters, mean_cum_per_agent, label=label)
plt.xlabel("Iteration")
plt.ylabel("Cumulative Per-Agent Reward")
plt.title("Per-Agent Cumulative Reward")
plt.legend()
plt.tight_layout()
plt.savefig("./pic/comparison/agent_cumulative_reward_comparison.png", dpi=300)
plt.close()

# === Plot 4: Total Reward per Iteration ===
plt.figure(figsize=(8, 5))
for algo, label in algos.items():
    _, _, total_reward, _ = load_reward_data(algo)
    plt.plot(iters, total_reward, label=label)
plt.xlabel("Iteration")
plt.ylabel("Total Reward")
plt.title("Total Reward per Iteration")
plt.legend()
plt.tight_layout()
plt.savefig("./pic/comparison/total_reward_per_iteration_comparison.png", dpi=300)
plt.close()

# === Plot 5: Total Cumulative Reward ===
plt.figure(figsize=(8, 5))
for algo, label in algos.items():
    _, _, _, total_cum_reward = load_reward_data(algo)
    plt.plot(iters, total_cum_reward, label=label)
plt.xlabel("Iteration")
plt.ylabel("Cumulative Total Reward")
plt.title("Cumulative Total Reward")
plt.legend()
plt.tight_layout()
plt.savefig("./pic/comparison/total_cumulative_reward_comparison.png", dpi=300)
plt.close()

# # === Load Facility Densities ===
# lrs_density = np.load('./npy/lrs/lrs_facility_density.npy')
# ql_density = np.load('./npy/qlearning/qlearning_ucb_densities.npy')

# labels = ['A', 'B', 'C', 'D']
# x = np.arange(len(labels))
# width = 0.2

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.bar(x - width, lrs_density[0], width, label='lrs Safe', color='blue')
# ax.bar(x, ql_density[0], width, label='QL Safe', color='green')
# ax.bar(x + width, lrs_density[1], width, label='lrs Bad', color='red')
# ax.bar(x + 2 * width, ql_density[1], width, label='QL Bad', color='orange')

# ax.set_xlabel("Facility")
# ax.set_ylabel("Avg. Agents per Facility")
# ax.set_title("Facility Usage Comparison")
# ax.set_xticks(x + width / 2)
# ax.set_xticklabels(labels)
# ax.legend()
# plt.tight_layout()
# plt.savefig("./pic/comparison/facility_comparison.png", dpi=300)
# plt.close()
