import numpy as np
import random
from congestion_games import *
from time import process_time
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import statistics
import itertools

# 共用函式：將 Q 轉為 softmax policy
def q_to_policy(Q, tau=1.0):
    policy = np.zeros_like(Q)
    for s in range(Q.shape[1]):
        max_q = np.max(Q[:, s, :], axis=1, keepdims=True)
        exp_q = np.exp((Q[:, s, :] - max_q) * tau)
        exp_q_sum = np.sum(exp_q, axis=1, keepdims=True)
        policy[:, s, :] = exp_q / exp_q_sum
    return policy

# 共用函式：L1-accuracy
def policy_accuracy(policy_pi, policy_star):
    total_dif = policy_pi.shape[0] * [0]
    for agent in range(policy_pi.shape[0]):
        for state in range(policy_pi.shape[1]):
            total_dif[agent] += np.sum(np.abs(policy_pi[agent][state] - policy_star[agent][state]))
    return np.sum(total_dif) / policy_pi.shape[0]

# 將策略轉為密度（facility分布）
def policy_to_facility_density(policy, act_dic, state_dic):
    N, S, A = policy.shape
    D = state_dic[0].m
    densities = np.zeros((S, D))
    for s in range(S):
        for i in range(N):
            for a in range(A):
                for f in act_dic[a]:
                    densities[s][f] += policy[i][s][a]
    return densities / N

# 主訓練與繪圖程序
def q_learning_ucb_experiment(N=8, H=20, M=1001, gamma=0.99, samples=10, epsilon=0.1, tau=1.0, runs=10, restart_interval=200):
    safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
    distancing_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
    state_dic = {0: safe_state, 1: distancing_state}
    S = 2
    A = safe_state.num_actions
    act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

    def get_next_state(state, actions, t):
        acts = [act_dic[a] for a in actions]
        density = state_dic[state].get_counts(acts)
        max_density = max(density)
        threshold = N / 2 - min(t // 200, 2)
        if state == 0 and max_density > N/2:
            return 1
        elif state == 1 and max_density <= N / 4:
            return 0
        return state

    def ucb_bonus(n, H):
        if n == 0:
            return float('inf')
        return 0.05 * np.sqrt(H**2 / n)

    all_accuracies = []
    all_final_policies = []
    all_total_rewards = []
    all_episode_rewards = []

    for run in range(runs):
        Q = np.ones((N, S, A))
        N_sa = np.zeros((N, S, A))
        policy_hist = []
        total_reward = 0
        run_episode_rewards = []

        for episode in range(M):
            if episode % restart_interval == 0:
                Q = np.ones((N, S, A))
                N_sa = np.zeros((N, S, A))

            state = 0
            episode_reward = 0
            for step in range(H):
                actions = []
                for i in range(N):
                    if random.random() < epsilon:
                        action = random.randint(0, A - 1)
                    else:
                        ucb_vals = Q[i][state] + np.array([ucb_bonus(N_sa[i][state][a], H) for a in range(A)])
                        action = np.argmax(ucb_vals)
                    actions.append(action)

                acts = [act_dic[a] for a in actions]
                rewards = get_reward(state_dic[state], acts)
                next_state = get_next_state(state, actions, episode)

                episode_reward += sum(rewards)

                for i in range(N):
                    a = actions[i]
                    N_sa[i][state][a] += 1
                    max_q = np.max(Q[i][next_state])
                    Q[i][state][a] = (1 - 1 / N_sa[i][state][a]) * Q[i][state][a] + \
                                     (1 / N_sa[i][state][a]) * (rewards[i] + gamma * max_q)
                state = next_state

            total_reward += episode_reward
            run_episode_rewards.append(episode_reward)
            pi = q_to_policy(Q, tau)
            policy_hist.append(copy.deepcopy(pi))

        final_pi = policy_hist[-1]
        run_accs = [policy_accuracy(pi, final_pi) for pi in policy_hist]
        all_accuracies.append(run_accs)
        all_final_policies.append(final_pi)
        all_total_rewards.append(total_reward)
        all_episode_rewards.append(run_episode_rewards)

    np.save("./npy/r_qlearning/qlearning_ucb_restart_accuracies.npy", np.array(all_accuracies))
    np.save("./npy/r_qlearning/qlearning_ucb_restart_rewards.npy", np.array(all_total_rewards))
    np.save("./npy/r_qlearning/qlearning_ucb_restart_densities.npy", policy_to_facility_density(all_final_policies[-1], act_dic, state_dic))
    np.save("./npy/r_qlearning/qlearning_ucb_restart_plot_matrix.npy", np.array(list(itertools.zip_longest(*all_accuracies, fillvalue=np.nan))).T)
    np.save("./npy/r_qlearning/qlearning_ucb_restart_episode_rewards.npy", np.array(all_episode_rewards))

    piters = list(range(len(all_accuracies[0])))
    plot_accuracies = np.array(list(itertools.zip_longest(*all_accuracies, fillvalue=np.nan))).T
    fig1 = plt.figure()
    for i in range(len(plot_accuracies)):
        plt.plot(piters, plot_accuracies[i])
    plt.xlabel("Iterations")
    plt.ylabel("L1-accuracy")
    plt.title("Restart Q-learning UCB: individual runs")
    plt.grid(True)
    plt.savefig("./pic/r_qlearning/qlearning_ucb_restart_individual_runs.png", dpi=300)
    plt.close()

    pmean = np.nanmean(plot_accuracies, axis=0).tolist()
    pstdv = np.nanstd(plot_accuracies, axis=0).tolist()
    np.save("./npy/r_qlearning/qlearning_ucb_restart_avg_mean.npy", np.array(pmean))
    np.save("./npy/r_qlearning/qlearning_ucb_restart_avg_std.npy", np.array(pstdv))
    fig2 = plt.figure()
    clrs = sns.color_palette("husl", 3)
    ax = sns.lineplot(x=piters, y=pmean, color=clrs[0], label='Mean L1-accuracy')
    ax.fill_between(piters, np.subtract(pmean, pstdv), np.add(pmean, pstdv), alpha=0.3, facecolor=clrs[0])
    ax.legend()
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("L1-accuracy")
    plt.title("Restart Q-learning UCB: average accuracy")
    plt.savefig("./pic/r_qlearning/qlearning_ucb_restart_avg_runs.png", dpi=300)
    plt.close()

    final_density = policy_to_facility_density(all_final_policies[-1], act_dic, state_dic)
    fig3, ax = plt.subplots()
    index = np.arange(safe_state.m)
    bar_width = 0.35
    plt.bar(index, final_density[0], bar_width, alpha=0.7, color='b', label='Safe state')
    plt.bar(index + bar_width, final_density[1], bar_width, alpha=1.0, color='r', label='Distancing state')
    plt.xlabel("Facility")
    plt.ylabel("Average number of agents")
    plt.title("Restart Q-learning UCB: facility density")
    plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("./pic/r_qlearning/qlearning_ucb_restart_facilities.png", dpi=300)
    plt.close()

    fig4 = plt.figure()
    plt.plot(all_total_rewards, marker='o')
    plt.xlabel("Run")
    plt.ylabel("Cumulative Reward")
    plt.title("Restart Q-learning UCB: Cumulative Reward Across Runs")
    plt.grid(True)
    plt.savefig("./pic/r_qlearning/qlearning_ucb_restart_cumulative_reward.png", dpi=300)
    plt.close()

    optimal_per_episode = N * 6
    regret_matrix = np.cumsum(optimal_per_episode - np.array(all_episode_rewards), axis=1)
    np.save("./npy/r_qlearning/qlearning_ucb_restart_episode_regret.npy", regret_matrix)

    fig5 = plt.figure()
    for r in range(runs):
        plt.plot(range(M), regret_matrix[r], alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")
    plt.title("Restart Q-learning UCB: Cumulative Regret per Episode")
    plt.grid(True)
    plt.savefig("./pic/r_qlearning/qlearning_ucb_restart_episode_regret.png", dpi=300)
    plt.close()

    episode_means = np.mean(all_episode_rewards, axis=0)
    episode_stds = np.std(all_episode_rewards, axis=0)

    plt.figure()
    plt.plot(range(M), episode_means, label="Mean")
    plt.fill_between(range(M), episode_means - episode_stds, episode_means + episode_stds, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Restart Q-learning UCB: Episode Reward (mean ± std)")
    plt.grid(True)
    plt.savefig("./pic/r_qlearning/qlearning_ucb_restart_mean_std_episode_reward.png", dpi=300)
    plt.close()

    avg_regret_per_episode = np.mean(optimal_per_episode - np.array(all_episode_rewards), axis=0)
    plt.figure()
    plt.plot(range(M), avg_regret_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Average Regret")
    plt.title("Restart Q-learning UCB: Average Regret per Episode")
    plt.grid(True)
    plt.savefig("./pic/r_qlearning/qlearning_ucb_restart_avg_regret_per_episode.png", dpi=300)
    plt.close()

    avg_reward_per_episode = np.mean(all_episode_rewards, axis=0)
    plt.figure()
    plt.plot(range(M), avg_reward_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Restart Q-learning UCB: Average Reward per Episode")
    plt.grid(True)
    plt.savefig("./pic/r_qlearning/qlearning_ucb_restart_avg_reward_per_episode.png", dpi=300)
    plt.close()

    return

if __name__ == '__main__':
    start = process_time()
    q_learning_ucb_experiment(N=8, H=80, M=1001, epsilon=0.1, runs=10, restart_interval=200)
    print("Done. Time elapsed:", process_time() - start)
