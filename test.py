import numpy as np
import random
from congestion_games import *
from time import process_time
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import statistics
import itertools
import argparse
import time
from datetime import datetime

def q_to_policy(Q, tau=1.0):
    policy = np.zeros_like(Q)
    for s in range(Q.shape[1]):
        max_q = np.max(Q[:, s, :], axis=1, keepdims=True)
        exp_q = np.exp((Q[:, s, :] - max_q) * tau)
        exp_q_sum = np.sum(exp_q, axis=1, keepdims=True)
        policy[:, s, :] = exp_q / exp_q_sum
    return policy

def policy_accuracy(policy_pi, policy_star):
    total_dif = policy_pi.shape[0] * [0]
    for agent in range(policy_pi.shape[0]):
        for state in range(policy_pi.shape[1]):
            total_dif[agent] += np.sum(np.abs(policy_pi[agent][state] - policy_star[agent][state]))
    return np.sum(total_dif) / policy_pi.shape[0]

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

def q_learning_ucb_experiment(N=8, H=20, M=1001, gamma=0.99, samples=10, epsilon=0.1, tau=1.0, runs=10, restart_interval=1001):
    def compute_stage_thresholds(H, M):
        L = [H]
        while L[-1] < M:
            L.append(int(L[-1] * (1 + 1.0 / H)))
        for i in range(1, len(L)):
            L[i] += L[i - 1]
        return set(L)
    stage_thresholds = compute_stage_thresholds(H, M)

    safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
    distancing_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
    safe_reward_options = [
        [[1, 0], [2, 0], [4, 0], [60, 0]],
        [[2, 0], [1, 0], [60, 0], [4, 0]],
        [[60, 0], [2, 0], [4, 0], [1, 0]],
        [[4, 0], [60, 0], [2, 0], [1, 0]],
    ]
    distancing_reward_options = [
        [[1, -100], [2, -100], [4, -100], [60, -100]],
        [[2, -100], [1, -100], [60, -100], [4, -100]],
        [[60, -100], [2, -100], [4, -100], [1, -100]],
        [[4, -100], [60, -100], [2, -100], [1, -100]],
    ]

    S = 2
    A = safe_state.num_actions

    def get_next_state(state, actions, t):
        acts = [act_dic[a] for a in actions]
        density = state_dic[state].get_counts(acts)
        max_density = max(density)
        threshold = N / 2 - min(t // 200, 2)
        if state == 0 and max_density > N / 2:
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
    all_agent_rewards = []

    for run in range(runs):
        N_check = np.zeros((N, S, A))
        r_check = np.zeros((N, S, A))
        v_check = np.zeros((N, S, A))
        Q = np.ones((N, S, A))
        N_sa = np.zeros((N, S, A))
        policy_hist = []
        total_reward = 0
        run_episode_rewards = []
        run_agent_rewards = []

        for episode in range(M):
            if episode % restart_interval == 0:
                Q = np.ones((N, S, A))
                N_sa = np.zeros((N, S, A))
                N_check = np.zeros((N, S, A))   # <--- 建議補上這三行
                r_check = np.zeros((N, S, A))
                v_check = np.zeros((N, S, A))

            index = ((episode % 40) // 10) % 4
            index = 0
            # index = episode% 4
            safe_weights = safe_reward_options[index]
            distancing_weights = distancing_reward_options[index]
            safe_state = CongGame(N, 1, safe_weights)
            distancing_state = CongGame(N, 1, distancing_weights)
            state_dic = {0: safe_state, 1: distancing_state}
            act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

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
                    # Q[i][state][a] = (1 - 1 / N_sa[i][state][a]) * Q[i][state][a] + \
                    #                  (1 / N_sa[i][state][a]) * (rewards[i] + gamma * max_q)
                    N_check[i][state][a] += 1
                    r_check[i][state][a] += rewards[i]
                    if step != H - 1:
                        v_check[i][state][a] += max_q  # 使用 next state's V

                    if N_sa[i][state][a] in stage_thresholds:
                        bonus = 0.05 * np.sqrt(H**2 / N_check[i][state][a])
                        Q[i][state][a] = min(Q[i][state][a],
                                            r_check[i][state][a] / N_check[i][state][a] +
                                            v_check[i][state][a] / N_check[i][state][a] + bonus)
                        N_check[i][state][a] = 0
                        r_check[i][state][a] = 0
                        v_check[i][state][a] = 0
                state = next_state

            total_reward += episode_reward
            run_episode_rewards.append(episode_reward)
            run_agent_rewards.append(rewards)
            pi = q_to_policy(Q, tau)
            policy_hist.append(copy.deepcopy(pi))

        final_pi = policy_hist[-1]
        run_accs = [policy_accuracy(pi, final_pi) for pi in policy_hist]
        all_accuracies.append(run_accs)
        all_final_policies.append(final_pi)
        all_total_rewards.append(total_reward)
        all_episode_rewards.append(run_episode_rewards)
        all_agent_rewards.append(run_agent_rewards)

    np.save("./npy/qlearning_ucb/qlearning_ucb_accuracies.npy", np.array(all_accuracies))
    np.save("./npy/qlearning_ucb/qlearning_ucb_rewards.npy", np.array(all_total_rewards))
    np.save("./npy/qlearning_ucb/qlearning_ucb_densities.npy", policy_to_facility_density(all_final_policies[-1], act_dic, state_dic))
    np.save("./npy/qlearning_ucb/qlearning_ucb_plot_matrix.npy", np.array(list(itertools.zip_longest(*all_accuracies, fillvalue=np.nan))).T)
    np.save("./npy/qlearning_ucb/qlearning_ucb_episode_rewards.npy", np.array(all_episode_rewards))
    np.save("./npy/qlearning_ucb/qlearning_ucb_agent_episode_rewards.npy", np.array(all_agent_rewards))

    piters = list(range(len(all_accuracies[0])))
    plot_accuracies = np.array(list(itertools.zip_longest(*all_accuracies, fillvalue=np.nan))).T
    fig1 = plt.figure()
    for i in range(len(plot_accuracies)):
        plt.plot(piters, plot_accuracies[i])
    plt.xlabel("Iterations")
    plt.ylabel("L1-accuracy")
    plt.title("Q-learning UCB: individual runs")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_individual_runs.png", dpi=300)
    plt.close()

    pmean = np.nanmean(plot_accuracies, axis=0).tolist()
    pstdv = np.nanstd(plot_accuracies, axis=0).tolist()
    np.save("./npy/qlearning_ucb/qlearning_ucb_avg_mean.npy", np.array(pmean))
    np.save("./npy/qlearning_ucb/qlearning_ucb_avg_std.npy", np.array(pstdv))
    fig2 = plt.figure()
    clrs = sns.color_palette("husl", 3)
    ax = sns.lineplot(x=piters, y=pmean, color=clrs[0], label='Mean L1-accuracy')
    ax.fill_between(piters, np.subtract(pmean, pstdv), np.add(pmean, pstdv), alpha=0.3, facecolor=clrs[0])
    ax.legend()
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("L1-accuracy")
    plt.title("Q-learning UCB: average accuracy")
    plt.savefig("./pic/qlearning/qlearning_ucb_avg_runs.png", dpi=300)
    plt.close()

    final_density = policy_to_facility_density(all_final_policies[-1], act_dic, state_dic)
    fig3, ax = plt.subplots()
    index = np.arange(safe_state.m)
    bar_width = 0.35
    plt.bar(index, final_density[0], bar_width, alpha=0.7, color='b', label='Safe state')
    plt.bar(index + bar_width, final_density[1], bar_width, alpha=1.0, color='r', label='Distancing state')
    plt.xlabel("Facility")
    plt.ylabel("Average number of agents")
    plt.title("Q-learning UCB: facility density")
    plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("./pic/qlearning/qlearning_ucb_facilities.png", dpi=300)
    plt.close()

    fig4 = plt.figure()
    plt.plot(all_total_rewards, marker='o')
    plt.xlabel("Run")
    plt.ylabel("Cumulative Reward")
    plt.title("Q-learning UCB: Cumulative Reward Across Runs")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_cumulative_reward.png", dpi=300)
    plt.close()

    optimal_per_episode = N * 6
    regret_matrix = np.cumsum(optimal_per_episode - np.array(all_episode_rewards), axis=1)
    np.save("./npy/qlearning_ucb/qlearning_ucb_episode_regret.npy", regret_matrix)

    fig5 = plt.figure()
    for r in range(runs):
        plt.plot(range(M), regret_matrix[r], alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")
    plt.title("Q-learning UCB: Cumulative Regret per Episode")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_episode_regret.png", dpi=300)
    plt.close()

    episode_means = np.mean(all_episode_rewards, axis=0)
    episode_stds = np.std(all_episode_rewards, axis=0)

    plt.figure()
    plt.plot(range(M), episode_means, label="Mean")
    plt.fill_between(range(M), episode_means - episode_stds, episode_means + episode_stds, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning UCB: Episode Reward (mean ± std)")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_mean_std_episode_reward.png", dpi=300)
    plt.close()

    avg_regret_per_episode = np.mean(optimal_per_episode - np.array(all_episode_rewards), axis=0)
    plt.figure()
    plt.plot(range(M), avg_regret_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Average Regret")
    plt.title("Q-learning UCB: Average Regret per Episode")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_avg_regret_per_episode.png", dpi=300)
    plt.close()

    avg_reward_per_episode = np.mean(all_episode_rewards, axis=0)
    plt.figure()
    plt.plot(range(M), avg_reward_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Q-learning UCB: Average Reward per Episode")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_avg_reward_per_episode.png", dpi=300)
    plt.close()

    # ✅ 新增 agent reward 曲線圖
    fig = plt.figure()
    all_agent_rewards = np.array(all_agent_rewards)
    for agent in range(N):
        agent_rewards = np.mean(all_agent_rewards[:, :, agent], axis=0)
        plt.plot(range(M), agent_rewards, label=f'Agent {agent}')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning UCB: Per-agent Episode Rewards")
    plt.legend()
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_per_agent_reward_curve.png", dpi=300)
    plt.close()
    # (前略，其餘部分不變)

    all_agent_rewards = np.array(all_agent_rewards)  # (runs, episodes, agents)

    # ====== ✅ 累積與總 reward 統計 ======
    agent_cumulative = np.cumsum(all_agent_rewards, axis=1)  # (runs, episodes, agents)
    total_per_episode = np.sum(all_agent_rewards, axis=2)    # (runs, episodes)
    total_cumulative = np.cumsum(total_per_episode, axis=1)  # (runs, episodes)

    # ====== ✅ 儲存 npy 檔案 ======
    # 舊名字
    # np.save("./npy/qlearning_ucb/qlearning_ucb_agent_episode_rewards.npy", all_agent_rewards)
    # np.save("./npy/qlearning_ucb/qlearning_ucb_agent_cumulative_rewards.npy", agent_cumulative)
    # np.save("./npy/qlearning_ucb/qlearning_ucb_total_per_episode_rewards.npy", total_per_episode)
    # np.save("./npy/qlearning_ucb/qlearning_ucb_total_cumulative_rewards.npy", total_cumulative)
    np.save("./npy/qlearning_ucb/qlearning_ucb__agent_reward_per_iteration.npy", all_agent_rewards)
    np.save("./npy/qlearning_ucb/qlearning_ucb_agent_cumulative_reward.npy", agent_cumulative)
    np.save("./npy/qlearning_ucb/qlearning_ucb_total_reward_per_iteration.npy", total_per_episode)
    np.save("./npy/qlearning_ucb/qlearning_ucb_total_cumulative_reward.npy", total_cumulative)
    # ====== ✅ 繪圖：各 agent 的累積 reward 曲線圖 ======
    fig = plt.figure()
    for agent in range(N):
        avg_cum = np.mean(agent_cumulative[:, :, agent], axis=0)
        plt.plot(range(M), avg_cum, label=f'Agent {agent}')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Q-learning UCB: Per-agent Cumulative Rewards")
    plt.legend()
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_per_agent_cumulative_reward.png", dpi=300)
    plt.close()

    # ====== ✅ 繪圖：所有 agent 的單集總 reward 曲線圖 ======
    avg_total_per_episode = np.mean(total_per_episode, axis=0)
    fig = plt.figure()
    plt.plot(range(M), avg_total_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning UCB: Total Reward per Episode (All Agents)")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_total_reward_per_episode.png", dpi=300)
    plt.close()

    # ====== ✅ 繪圖：所有 agent 的累積總 reward 曲線圖 ======
    avg_total_cumulative = np.mean(total_cumulative, axis=0)
    fig = plt.figure()
    plt.plot(range(M), avg_total_cumulative)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Total Reward")
    plt.title("Q-learning UCB: Cumulative Total Reward (All Agents)")
    plt.grid(True)
    plt.savefig("./pic/qlearning/s_qlearning_ucb_total_cumulative_reward.png", dpi=300)
    plt.close()


    return

if __name__ == '__main__':
    start = process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default="5001",
                        help="Choose reward episode: default=5001")
    args = parser.parse_args()

    log_lines = []
    currentDateAndTime = datetime.now()
    formatted_time = currentDateAndTime.strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"<Qlearning> {formatted_time}")
    log_lines.append(f"Episode: {args.m}")
    q_learning_ucb_experiment(N=8, H=80, M=args.m, epsilon=0.1, runs=10, restart_interval=10000000)
    log_lines.append(f"Done. Time elapsed: {(process_time() - start):.4f} seconds\n")
    with open("log.txt", "a") as f:
        for line in log_lines:
            f.write(line + "\n")
    
    # print("Done. Time elapsed:", process_time() - start)