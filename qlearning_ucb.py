# # import numpy as np
# # import random
# # from congestion_games import *
# # from time import process_time
# # import matplotlib.pyplot as plt


# # def ucb_bonus(n, H):
# #     if n == 0:
# #         return float('inf')  # 強制探索
# #     return 0.05 * np.sqrt(H**2 / n)


# # def q_learning_ucb_agent(N=8, H=20, M=1000, gamma=0.99, samples=10):
# #     safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
# #     bad_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
# #     state_dic = {0: safe_state, 1: bad_state}
# #     S = 2
# #     A = safe_state.num_actions

# #     act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

# #     # Q = np.random.uniform(0.5, 1.5, size=(N, S, A))
# #     Q = np.ones((N, S, A))  # Q[agent][state][action]
# #     N_sa = np.zeros((N, S, A))
# #     iteration_rewards = np.zeros((N, M))

# #     def get_next_state(state, actions):
# #         acts = [act_dic[a] for a in actions]
# #         density = state_dic[state].get_counts(acts)
# #         max_density = max(density)
# #         if state == 0 and max_density > N/2:
# #             return 1
# #         elif state == 1 and max_density <= N/4:
# #             return 0
# #         return state

# #     for iteration in range(M):
# #         state = 0
# #         for step in range(H):
# #             actions = []
# #             for i in range(N):
# #                 q_vals = Q[i][state]
# #                 ucb_vals = q_vals + np.array([ucb_bonus(N_sa[i][state][a], H) for a in range(A)])
# #                 action = np.argmax(ucb_vals)
# #                 actions.append(action)

# #             acts = [act_dic[a] for a in actions]
# #             rewards = get_reward(state_dic[state], acts)
# #             next_state = get_next_state(state, actions)

# #             for i in range(N):
# #                 a = actions[i]
# #                 N_sa[i][state][a] += 1
# #                 max_next_q = np.max(Q[i][next_state])
# #                 Q[i][state][a] = (1 - 1 / N_sa[i][state][a]) * Q[i][state][a] + \
# #                                  (1 / N_sa[i][state][a]) * (rewards[i] + gamma * max_next_q)
# #                 iteration_rewards[i][iteration] += rewards[i]

# #             state = next_state

# #     return iteration_rewards, Q


# # def plot_agent_rewards(iteration_rewards):
# #     N = iteration_rewards.shape[0]
# #     for i in range(N):
# #         plt.plot(iteration_rewards[i], label=f"Agent {i}")
# #     plt.xlabel("iteration")
# #     plt.ylabel("Cumulative Reward")
# #     plt.title("Q-Learning-UCB Rewards per Agent")
# #     plt.legend()
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.savefig("qlearning_ucb_rewards.png", dpi=300)
# #     plt.show()


# # if __name__ == '__main__':
# #     start = process_time()
# #     rewards, Q = q_learning_ucb_agent(N=8, H=20, M=500)
# #     plot_agent_rewards(rewards)
# #     print("Done. Time elapsed:", process_time() - start)


# # # add greedy

# # import numpy as np
# # import random
# # from congestion_games import *
# # from time import process_time
# # import matplotlib.pyplot as plt


# # def ucb_bonus(n, H):
# #     if n == 0:
# #         return float('inf')  # 強制探索
# #     return 0.05 * np.sqrt(H**2 / n)


# # def q_learning_ucb_agent(N=8, H=20, M=1000, gamma=0.99, samples=10, epsilon=0.05):
# #     safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
# #     bad_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
# #     state_dic = {0: safe_state, 1: bad_state}
# #     S = 2
# #     A = safe_state.num_actions

# #     act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

# #     Q = np.ones((N, S, A))  # Q[agent][state][action]
# #     N_sa = np.zeros((N, S, A))
# #     iteration_rewards = np.zeros((N, M))

# #     def get_next_state(state, actions):
# #         acts = [act_dic[a] for a in actions]
# #         density = state_dic[state].get_counts(acts)
# #         max_density = max(density)
# #         if state == 0 and max_density > N/2:
# #             return 1
# #         elif state == 1 and max_density <= N/4:
# #             return 0
# #         return state

# #     for iteration in range(M):
# #         state = 0
# #         for step in range(H):
# #             actions = []
# #             for i in range(N):
# #                 if random.random() < epsilon:
# #                     action = random.randint(0, A - 1)
# #                 else:
# #                     q_vals = Q[i][state]
# #                     ucb_vals = q_vals + np.array([ucb_bonus(N_sa[i][state][a], H) for a in range(A)])
# #                     action = np.argmax(ucb_vals)
# #                 actions.append(action)

# #             acts = [act_dic[a] for a in actions]
# #             rewards = get_reward(state_dic[state], acts)
# #             next_state = get_next_state(state, actions)

# #             for i in range(N):
# #                 a = actions[i]
# #                 N_sa[i][state][a] += 1
# #                 max_next_q = np.max(Q[i][next_state])
# #                 Q[i][state][a] = (1 - 1 / N_sa[i][state][a]) * Q[i][state][a] + \
# #                                  (1 / N_sa[i][state][a]) * (rewards[i] + gamma * max_next_q)
# #                 iteration_rewards[i][iteration] += rewards[i]

# #             state = next_state

# #     return iteration_rewards, Q


# # def plot_agent_rewards(iteration_rewards):
# #     N = iteration_rewards.shape[0]
# #     for i in range(N):
# #         plt.plot(iteration_rewards[i], label=f"Agent {i}")
# #     plt.xlabel("iteration")
# #     plt.ylabel("Cumulative Reward")
# #     plt.title("Q-Learning-UCB Rewards per Agent")
# #     plt.legend()
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.savefig("qlearning_ucb_rewards.png", dpi=300)
# #     plt.show()


# # if __name__ == '__main__':
# #     start = process_time()
# #     rewards, Q = q_learning_ucb_agent(N=8, H=20, M=500, epsilon=0.1)
# #     plot_agent_rewards(rewards)
# #     print("Done. Time elapsed:", process_time() - start)


# 04
# l1-accuracy
import numpy as np
import random
from congestion_games import *
from time import process_time
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import statistics
import itertools
import os

def q_to_policy(Q, tau=1.0):
    policy = np.zeros_like(Q)
    for s in range(Q.shape[1]):
        for a in range(Q.shape[2]):
            policy[:, s, a] = np.exp(Q[:, s, a] * tau)
    policy_sum = np.sum(policy, axis=2, keepdims=True)
    return policy / policy_sum

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

def q_learning_ucb_experiment(N=8, H=20, M=1001, gamma=0.99, samples=10, epsilon=0.1, tau=1.0, runs=10):
    safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
    distancing_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
    state_dic = {0: safe_state, 1: distancing_state}
    safe_reward_options = [
    [[1, 0], [2, 0], [4, 0], [6, 0]],
    [[2, 0], [1, 0], [6, 0], [4, 0]],
    [[6, 0], [2, 0], [4, 0], [1, 0]],
    [[4, 0], [6, 0], [2, 0], [1, 0]],
    ]

    distancing_reward_options = [
    [[1, -100], [2, -100], [4, -100], [6, -100]],
    [[2, -100], [1, -100], [6, -100], [4, -100]],
    [[6, -100], [2, -100], [4, -100], [1, -100]],
    [[4, -100], [6, -100], [2, -100], [1, -100]],
    ]
    S = 2
    A = safe_state.num_actions
    # act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

    def get_next_state(state, actions, t):
        acts = [act_dic[a] for a in actions]
        density = state_dic[state].get_counts(acts)
        max_density = max(density)
        threshold = N/2
        if state == 0 and max_density > threshold:
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
    all_iteration_rewards = []
    total_reward_per_iteration = np.zeros(M)

    for run in range(runs):
        Q = np.ones((N, S, A))
        N_sa = np.zeros((N, S, A))
        policy_hist = []
        total_reward = 0
        run_iteration_rewards = []

        for iteration in range(M):
            index = ((iteration % 160) // 40) % 4
            safe_weights = safe_reward_options[index]
            distancing_weights = distancing_reward_options[index]
            safe_state = CongGame(N, 1, safe_weights)
            distancing_state = CongGame(N, 1, distancing_weights)
            state_dic = {0: safe_state, 1: distancing_state}
            
            act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

            state = 0
            iteration_reward = 0
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
                next_state = get_next_state(state, actions, iteration)

                iteration_reward += sum(rewards)

                for i in range(N):
                    a = actions[i]
                    N_sa[i][state][a] += 1
                    max_q = np.max(Q[i][next_state])
                    Q[i][state][a] = (1 - 1 / N_sa[i][state][a]) * Q[i][state][a] + \
                                     (1 / N_sa[i][state][a]) * (rewards[i] + gamma * max_q)
                state = next_state

            run_iteration_rewards.append(iteration_reward)
            total_reward += iteration_reward
            total_reward_per_iteration[iteration] += iteration_reward
            pi = q_to_policy(Q, tau)
            policy_hist.append(copy.deepcopy(pi))

        final_pi = policy_hist[-1]
        run_accs = [policy_accuracy(pi, final_pi) for pi in policy_hist]
        all_accuracies.append(run_accs)
        all_final_policies.append(final_pi)
        all_total_rewards.append(total_reward)
        all_iteration_rewards.append(run_iteration_rewards)

    cumulative_total_reward_per_iteration = np.cumsum(total_reward_per_iteration)

    os.makedirs("./npy/qlearning", exist_ok=True)
    os.makedirs("./pic/qlearning", exist_ok=True)

    np.save("./npy/qlearning/qlearning_ucb_accuracies.npy", np.array(all_accuracies))
    np.save("./npy/qlearning/qlearning_ucb_rewards.npy", np.array(all_total_rewards))
    np.save("./npy/qlearning/qlearning_ucb_densities.npy", policy_to_facility_density(all_final_policies[-1], act_dic, state_dic))
    np.save("./npy/qlearning/qlearning_ucb_plot_matrix.npy", np.array(list(itertools.zip_longest(*all_accuracies, fillvalue=np.nan))).T)
    np.save("./npy/qlearning/qlearning_ucb_iteration_rewards.npy", np.array(all_iteration_rewards))
    np.save("./npy/qlearning/qlearning_ucb_total_reward_per_iteration.npy", total_reward_per_iteration)
    np.save("./npy/qlearning/qlearning_ucb_total_cumulative_reward.npy", cumulative_total_reward_per_iteration)

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

    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = [statistics.stdev(list(col)) for col in zip(*plot_accuracies)]
    np.save("./npy/qlearning/qlearning_ucb_avg_mean.npy", np.array(pmean))
    np.save("./npy/qlearning/qlearning_ucb_avg_std.npy", np.array(pstdv))
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

    optimal_per_iteration = N * 6
    regret_matrix = np.cumsum(optimal_per_iteration - np.array(all_iteration_rewards), axis=1)
    np.save("./npy/qlearning/qlearning_ucb_iteration_regret.npy", regret_matrix)

    fig5 = plt.figure()
    for r in range(runs):
        plt.plot(range(M), regret_matrix[r], alpha=0.6)
    plt.xlabel("iteration")
    plt.ylabel("Cumulative Regret")
    plt.title("Q-learning UCB: Cumulative Regret per iteration")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_iteration_regret.png", dpi=300)
    plt.close()

    iteration_means = np.mean(all_iteration_rewards, axis=0)
    iteration_stds = np.std(all_iteration_rewards, axis=0)

    plt.figure()
    plt.plot(range(M), iteration_means, label="Mean")
    plt.fill_between(range(M), iteration_means - iteration_stds, iteration_means + iteration_stds, alpha=0.3)
    plt.xlabel("iteration")
    plt.ylabel("Reward")
    plt.title("Q-learning UCB: iteration Reward (mean ± std)")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_mean_std_iteration_reward.png", dpi=300)
    plt.close()

    fig6 = plt.figure()
    plt.plot(range(M), total_reward_per_iteration)
    plt.xlabel("iteration")
    plt.ylabel("Total Reward")
    plt.title("Q-learning UCB: Total Reward per iteration (All Agents)")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_total_reward_per_iteration.png", dpi=300)
    plt.close()

    fig7 = plt.figure()
    plt.plot(range(M), cumulative_total_reward_per_iteration)
    plt.xlabel("iteration")
    plt.ylabel("Cumulative Total Reward")
    plt.title("Q-learning UCB: Cumulative Total Reward up to iteration")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_total_cumulative_reward.png", dpi=300)
    plt.close()

        # === New: Agent-Level Rewards ===
    agent_reward_per_iteration = np.zeros((runs, N, M))
    for run in range(runs):
        Q = np.ones((N, S, A))
        N_sa = np.zeros((N, S, A))
        for iteration in range(M):
            state = 0
            rewards_this_iter = np.zeros(N)
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
                next_state = get_next_state(state, actions, iteration)
                for i in range(N):
                    a = actions[i]
                    N_sa[i][state][a] += 1
                    max_q = np.max(Q[i][next_state])
                    Q[i][state][a] = (1 - 1 / N_sa[i][state][a]) * Q[i][state][a] + \
                                     (1 / N_sa[i][state][a]) * (rewards[i] + gamma * max_q)
                rewards_this_iter = rewards
                state = next_state
            agent_reward_per_iteration[run, :, iteration] = rewards_this_iter

    agent_reward_mean = np.mean(agent_reward_per_iteration, axis=0)  # shape (N, M)
    agent_cumulative_reward_mean = np.cumsum(agent_reward_mean, axis=1)

    np.save("./npy/qlearning/qlearning_ucb_agent_reward_per_iteration.npy", agent_reward_mean)
    np.save("./npy/qlearning/qlearning_ucb_agent_cumulative_reward.npy", agent_cumulative_reward_mean)

    fig8 = plt.figure()
    for i in range(N):
        plt.plot(range(M), agent_reward_mean[i])
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Q-learning UCB: Per-Agent Reward per Iteration")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_agent_reward_per_iteration.png", dpi=300)
    plt.close()

    fig9 = plt.figure()
    for i in range(N):
        plt.plot(range(M), agent_cumulative_reward_mean[i])
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Reward")
    plt.title("Q-learning UCB: Per-Agent Cumulative Reward")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_agent_cumulative_reward.png", dpi=300)
    plt.close()

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

if __name__ == '__main__':
    start = process_time()
    q_learning_ucb_experiment(N=8, H=20, M=5001, epsilon=0.1, runs=10)
    print("Done. Time elapsed:", process_time() - start)

# # run_qlearning_ucb_experiment.py
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import itertools
# import statistics
# import os
# from congestion_games import CongGame, get_reward, get_next_state, build_act_dic

# sns.set()

# def q_learning_ucb_congestion(n_agents, n_states, n_actions, env_dict,
#                                iterations=1000, H=20, bonus_scale=0.05, act_dic=None):
#     Q = np.ones((n_states, n_agents, n_actions))
#     V = np.ones((n_states, n_agents))
#     N = np.zeros((n_states, n_agents, n_actions))
#     rewards_history = np.zeros(iterations)
#     agent_action_counts = np.zeros((n_states, n_actions))

#     for iteration in range(iterations):
#         state = 0
#         iteration_action_count = np.zeros((n_states, n_actions))

#         for h in range(H):
#             actions = []
#             for i in range(n_agents):
#                 ucb_values = Q[state, i] + bonus_scale * np.sqrt(
#                     H**2 / (N[state, i] + 1e-6))
#                 action = np.argmax(ucb_values)
#                 actions.append(action)
#                 iteration_action_count[state, action] += 1

#             act_profile = [act_dic[a] for a in actions]
#             rewards = get_reward(env_dict[state], act_profile)
#             next_state = get_next_state(state, actions, env_dict)

#             for i in range(n_agents):
#                 a = actions[i]
#                 r = rewards[i]
#                 Q[state, i, a] = min(
#                     Q[state, i, a],
#                     r + V[next_state, i] + bonus_scale * np.sqrt(H**2 / (N[state, i, a] + 1e-6))
#                 )
#                 V[state, i] = np.max(Q[state, i])
#                 N[state, i, a] += 1

#             state = next_state
#             rewards_history[iteration] += np.sum(rewards)

#         agent_action_counts += iteration_action_count

#     avg_action_density = agent_action_counts / iterations
#     return Q, rewards_history, avg_action_density

# def run_ucb_experiment():
#     N = 8
#     safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
#     bad_state = CongGame(N, 1, [[1, -120], [2, -130], [4, -140], [6, -150]])
#     state_dic = {0: safe_state, 1: bad_state}
#     M = safe_state.num_actions
#     S = 2
#     D = safe_state.m
#     act_dic = build_act_dic(safe_state)

#     runs = 10
#     iterations = 1000
#     H = 20

#     # os.makedirs("results", exist_ok=True)

#     all_rewards = []
#     all_densities = np.zeros((S, M))

#     for run in range(runs):
#         _, rewards, density = q_learning_ucb_congestion(
#             N, S, M, state_dic, iterations, H, bonus_scale=0.05, act_dic=act_dic)
#         all_rewards.append(rewards)
#         all_densities += density

#     avg_densities = all_densities / runs
#     plot_accuracies = np.array(list(itertools.zip_longest(*all_rewards, fillvalue=np.nan))).T

#     # Save .npy data
#     np.save("./pic/qlearning/ucb_rewards.npy", plot_accuracies)
#     np.save("./pic/qlearning/ucb_avg_densities.npy", avg_densities)

#     # Figure 1: Reward per iteration
#     fig1 = plt.figure(figsize=(6, 4))
#     for i in range(len(plot_accuracies)):
#         plt.plot(range(plot_accuracies.shape[1]), plot_accuracies[i])
#     plt.grid(linewidth=0.6)
#     plt.xlabel('iterations')
#     plt.ylabel('Total Reward')
#     plt.title(f'UCB Q-Learning: agents={N}, runs={runs}')
#     fig1.savefig("./pic/qlearning/ucb_individual_rewards.png", bbox_inches='tight')
#     plt.close()

#     # Figure 2: Mean reward with standard deviation
#     plot_accuracies = np.nan_to_num(plot_accuracies)
#     pmean = list(map(statistics.mean, zip(*plot_accuracies)))
#     pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))

#     fig2 = plt.figure(figsize=(6, 4))
#     ax = sns.lineplot(x=list(range(len(pmean))), y=pmean, label='Mean Total Reward')
#     ax.fill_between(range(len(pmean)), np.subtract(pmean, pstdv), np.add(pmean, pstdv), alpha=0.3, label='±1 std dev')
#     ax.legend()
#     plt.grid(linewidth=0.6)
#     plt.xlabel('iterations')
#     plt.ylabel('Total Reward')
#     plt.title(f'UCB Q-Learning: agents={N}, runs={runs}')
#     fig2.savefig("./pic/qlearning/ucb_avg_rewards.png", bbox_inches='tight')
#     plt.close()

#     # Figure 3: Facility bar plot
#     fig3, ax = plt.subplots()
#     index = np.arange(D)
#     bar_width = 0.35

#     plt.bar(index, avg_densities[0][:D], bar_width, alpha=0.7, label='Safe state')
#     plt.bar(index + bar_width, avg_densities[1][:D], bar_width, alpha=0.9, label='Bad state')

#     plt.xlabel('Facility')
#     plt.ylabel('Average number of agents')
#     plt.title(f'UCB Q-Learning: agents={N}, runs={runs}')
#     plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D'))
#     plt.legend()
#     plt.grid(True)
#     fig3.savefig("./pic/qlearning/ucb_facilities.png", bbox_inches='tight')
#     plt.close()

#     return fig1, fig2, fig3

# # Run the experiment
# if __name__ == '__main__':
#     run_ucb_experiment()
