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
ql_mean = np.load('./npy/qlearning/qlearning_ucb_avg_mean.npy')
ql_std = np.load('./npy/qlearning/qlearning_ucb_avg_std.npy')

iters = range(len(lrs_mean))

# === Plot 1: Accuracy Mean + Std ===
plt.figure(figsize=(8, 5))
plt.plot(iters, lrs_mean, label='Policy Gradient', color='blue')
plt.fill_between(iters, lrs_mean - lrs_std, lrs_mean + lrs_std, color='blue', alpha=0.3)

plt.plot(iters, ql_mean, label='Q-Learning UCB', color='green')
plt.fill_between(iters, ql_mean - ql_std, ql_mean + ql_std, color='green', alpha=0.3)

plt.xlabel("Iterations")
plt.ylabel("L1 Accuracy")
plt.title("Mean L1 Accuracy Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("./pic/comparison/l1_accuracy_comparison.png", dpi=300)
plt.close()

# # === Load Facility Densities ===
# pg_density = np.load('./npy/pg/pg_facility_density.npy')
# ql_density = np.load('./npy/qlearning/qlearning_ucb_facility_density.npy')

# labels = ['A', 'B', 'C', 'D']
# x = np.arange(len(labels))
# width = 0.2

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.bar(x - width, pg_density[0], width, label='PG Safe', color='blue')
# ax.bar(x, ql_density[0], width, label='QL Safe', color='green')
# ax.bar(x + width, pg_density[1], width, label='PG Bad', color='red')
# ax.bar(x + 2 * width, ql_density[1], width, label='QL Bad', color='orange')

# ax.set_xlabel("Facility")
# ax.set_ylabel("Avg. Agents per Facility")
# ax.set_title("Facility Usage Comparison")
# ax.set_xticks(x + width / 2)
# ax.set_xticklabels(labels)
# ax.legend()
# plt.tight_layout()
# plt.savefig("./comparison/facility_comparison.png", dpi=300)
# plt.close()
