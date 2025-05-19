# import numpy as np
# import random
# from congestion_games import *
# from time import process_time
# import matplotlib.pyplot as plt


# def ucb_bonus(n, H):
#     if n == 0:
#         return float('inf')  # 強制探索
#     return 0.05 * np.sqrt(H**2 / n)


# def q_learning_ucb_agent(N=8, H=20, M=1000, gamma=0.99, samples=10):
#     safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
#     bad_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
#     state_dic = {0: safe_state, 1: bad_state}
#     S = 2
#     A = safe_state.num_actions

#     act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

#     # Q = np.random.uniform(0.5, 1.5, size=(N, S, A))
#     Q = np.ones((N, S, A))  # Q[agent][state][action]
#     N_sa = np.zeros((N, S, A))
#     episode_rewards = np.zeros((N, M))

#     def get_next_state(state, actions):
#         acts = [act_dic[a] for a in actions]
#         density = state_dic[state].get_counts(acts)
#         max_density = max(density)
#         if state == 0 and max_density > N/2:
#             return 1
#         elif state == 1 and max_density <= N/4:
#             return 0
#         return state

#     for episode in range(M):
#         state = 0
#         for step in range(H):
#             actions = []
#             for i in range(N):
#                 q_vals = Q[i][state]
#                 ucb_vals = q_vals + np.array([ucb_bonus(N_sa[i][state][a], H) for a in range(A)])
#                 action = np.argmax(ucb_vals)
#                 actions.append(action)

#             acts = [act_dic[a] for a in actions]
#             rewards = get_reward(state_dic[state], acts)
#             next_state = get_next_state(state, actions)

#             for i in range(N):
#                 a = actions[i]
#                 N_sa[i][state][a] += 1
#                 max_next_q = np.max(Q[i][next_state])
#                 Q[i][state][a] = (1 - 1 / N_sa[i][state][a]) * Q[i][state][a] + \
#                                  (1 / N_sa[i][state][a]) * (rewards[i] + gamma * max_next_q)
#                 episode_rewards[i][episode] += rewards[i]

#             state = next_state

#     return episode_rewards, Q


# def plot_agent_rewards(episode_rewards):
#     N = episode_rewards.shape[0]
#     for i in range(N):
#         plt.plot(episode_rewards[i], label=f"Agent {i}")
#     plt.xlabel("Episode")
#     plt.ylabel("Cumulative Reward")
#     plt.title("Q-Learning-UCB Rewards per Agent")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("qlearning_ucb_rewards.png", dpi=300)
#     plt.show()


# if __name__ == '__main__':
#     start = process_time()
#     rewards, Q = q_learning_ucb_agent(N=8, H=20, M=500)
#     plot_agent_rewards(rewards)
#     print("Done. Time elapsed:", process_time() - start)


# # add greedy

# import numpy as np
# import random
# from congestion_games import *
# from time import process_time
# import matplotlib.pyplot as plt


# def ucb_bonus(n, H):
#     if n == 0:
#         return float('inf')  # 強制探索
#     return 0.05 * np.sqrt(H**2 / n)


# def q_learning_ucb_agent(N=8, H=20, M=1000, gamma=0.99, samples=10, epsilon=0.05):
#     safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
#     bad_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
#     state_dic = {0: safe_state, 1: bad_state}
#     S = 2
#     A = safe_state.num_actions

#     act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

#     Q = np.ones((N, S, A))  # Q[agent][state][action]
#     N_sa = np.zeros((N, S, A))
#     episode_rewards = np.zeros((N, M))

#     def get_next_state(state, actions):
#         acts = [act_dic[a] for a in actions]
#         density = state_dic[state].get_counts(acts)
#         max_density = max(density)
#         if state == 0 and max_density > N/2:
#             return 1
#         elif state == 1 and max_density <= N/4:
#             return 0
#         return state

#     for episode in range(M):
#         state = 0
#         for step in range(H):
#             actions = []
#             for i in range(N):
#                 if random.random() < epsilon:
#                     action = random.randint(0, A - 1)
#                 else:
#                     q_vals = Q[i][state]
#                     ucb_vals = q_vals + np.array([ucb_bonus(N_sa[i][state][a], H) for a in range(A)])
#                     action = np.argmax(ucb_vals)
#                 actions.append(action)

#             acts = [act_dic[a] for a in actions]
#             rewards = get_reward(state_dic[state], acts)
#             next_state = get_next_state(state, actions)

#             for i in range(N):
#                 a = actions[i]
#                 N_sa[i][state][a] += 1
#                 max_next_q = np.max(Q[i][next_state])
#                 Q[i][state][a] = (1 - 1 / N_sa[i][state][a]) * Q[i][state][a] + \
#                                  (1 / N_sa[i][state][a]) * (rewards[i] + gamma * max_next_q)
#                 episode_rewards[i][episode] += rewards[i]

#             state = next_state

#     return episode_rewards, Q


# def plot_agent_rewards(episode_rewards):
#     N = episode_rewards.shape[0]
#     for i in range(N):
#         plt.plot(episode_rewards[i], label=f"Agent {i}")
#     plt.xlabel("Episode")
#     plt.ylabel("Cumulative Reward")
#     plt.title("Q-Learning-UCB Rewards per Agent")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("qlearning_ucb_rewards.png", dpi=300)
#     plt.show()


# if __name__ == '__main__':
#     start = process_time()
#     rewards, Q = q_learning_ucb_agent(N=8, H=20, M=500, epsilon=0.1)
#     plot_agent_rewards(rewards)
#     print("Done. Time elapsed:", process_time() - start)

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

# 共用函式：將 Q 轉為 softmax policy
def q_to_policy(Q, tau=1.0):
    policy = np.zeros_like(Q)
    for s in range(Q.shape[1]):
        for a in range(Q.shape[2]):
            policy[:, s, a] = np.exp(Q[:, s, a] * tau)
    policy_sum = np.sum(policy, axis=2, keepdims=True)
    return policy / policy_sum

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
def q_learning_ucb_experiment(N=8, H=20, M=500, gamma=0.99, samples=10, epsilon=0.1, tau=1.0, runs=10):
    safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
    distancing_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
    state_dic = {0: safe_state, 1: distancing_state}
    S = 2
    A = safe_state.num_actions
    act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

    def get_next_state(state, actions):
        acts = [act_dic[a] for a in actions]
        density = state_dic[state].get_counts(acts)
        max_density = max(density)
        if state == 0 and max_density > N/2:
            return 1
        elif state == 1 and max_density <= N/4:
            return 0
        return state

    def ucb_bonus(n, H):
        if n == 0:
            return float('inf')
        return 0.05 * np.sqrt(H**2 / n)

    all_accuracies = []
    all_final_policies = []
    all_total_rewards = []

    for run in range(runs):
        Q = np.ones((N, S, A))
        N_sa = np.zeros((N, S, A))
        policy_hist = []
        total_reward = 0

        for episode in range(M):
            state = 0
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
                next_state = get_next_state(state, actions)

                total_reward += sum(rewards)

                for i in range(N):
                    a = actions[i]
                    N_sa[i][state][a] += 1
                    max_q = np.max(Q[i][next_state])
                    Q[i][state][a] = (1 - 1 / N_sa[i][state][a]) * Q[i][state][a] + \
                                     (1 / N_sa[i][state][a]) * (rewards[i] + gamma * max_q)
                state = next_state

            # 儲存策略 policy（每 1 輪記一次）
            pi = q_to_policy(Q, tau)
            policy_hist.append(copy.deepcopy(pi))

        final_pi = policy_hist[-1]
        run_accs = [policy_accuracy(pi, final_pi) for pi in policy_hist]
        all_accuracies.append(run_accs)
        all_final_policies.append(final_pi)
        all_total_rewards.append(total_reward)

    # Plot 1: individual accuracy
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

    # Plot 2: average accuracy
    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = [statistics.stdev(list(col)) for col in zip(*plot_accuracies)]
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

    # Plot 3: facility distribution
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

    # Plot 4: cumulative reward
    fig4 = plt.figure()
    plt.plot(all_total_rewards, marker='o')
    plt.xlabel("Run")
    plt.ylabel("Cumulative Reward")
    plt.title("Q-learning UCB: Cumulative Reward Across Runs")
    plt.grid(True)
    plt.savefig("./pic/qlearning/qlearning_ucb_cumulative_reward.png", dpi=300)
    plt.close()

    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    start = process_time()
    q_learning_ucb_experiment(N=8, H=20, M=500, epsilon=0.1, runs=10)
    print("Done. Time elapsed:", process_time() - start)
