# import numpy as np
# import random
# from congestion_games import *
# from time import process_time
# import matplotlib.pyplot as plt
# import seaborn as sns
# import copy
# import statistics
# import itertools
# import os

# def epsilon_greedy_exploration(Q, epsilon, num_actions):
#     def policy_exp(state):
#         probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
#         best_action = np.argmax(Q[state])
#         probs[best_action] += (1.0 - epsilon)
#         return probs
#     return policy_exp

# def greedy_congestion_experiment(N=8, H=20, M=1001, gamma=0.99, epsilon=0.1, tau=1.0, runs=10):
#     safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
#     distancing_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
#     state_dic = {0: safe_state, 1: distancing_state}
#     S = 2
#     A = safe_state.num_actions
#     act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

#     def get_next_state(state, actions, t):
#         acts = [act_dic[a] for a in actions]
#         density = state_dic[state].get_counts(acts)
#         max_density = max(density)
#         threshold = random.randint(0, N)
#         if state == 0 and max_density > threshold:
#             return 1
#         elif state == 1 and max_density <= N / 4:
#             return 0
#         return state

#     def q_to_policy(Q, tau=1.0):
#         policy = np.zeros_like(Q)
#         for s in range(Q.shape[1]):
#             for a in range(Q.shape[2]):
#                 policy[:, s, a] = np.exp(Q[:, s, a] * tau)
#         policy_sum = np.sum(policy, axis=2, keepdims=True)
#         policy_sum[policy_sum == 0] = 1e-8  # 加入 epsilon 避免除以 0
#         return policy / policy_sum

#     def policy_accuracy(policy_pi, policy_star):
#         total_dif = policy_pi.shape[0] * [0]
#         for agent in range(policy_pi.shape[0]):
#             for state in range(policy_pi.shape[1]):
#                 total_dif[agent] += np.sum(np.abs(policy_pi[agent][state] - policy_star[agent][state]))
#         return np.sum(total_dif) / policy_pi.shape[0]

#     def policy_to_facility_density(policy, act_dic, state_dic):
#         N, S, A = policy.shape
#         D = state_dic[0].m
#         densities = np.zeros((S, D))
#         for s in range(S):
#             for i in range(N):
#                 for a in range(A):
#                     for f in act_dic[a]:
#                         densities[s][f] += policy[i][s][a]
#         return densities / N

#     all_accuracies = []
#     all_final_policies = []
#     all_total_rewards = []
#     all_episode_rewards = []

#     for run in range(runs):
#         Q = np.zeros((N, S, A))
#         policy_hist = []
#         total_reward = 0
#         run_episode_rewards = []

#         for episode in range(M):
#             state = 0
#             episode_reward = 0
#             N_sa = np.zeros((N, S, A))

#             for step in range(H):
#                 actions = []
#                 for i in range(N):
#                     probs = np.ones(A) * epsilon / A
#                     best_a = np.argmax(Q[i][state])
#                     probs[best_a] += 1.0 - epsilon
#                     action = np.random.choice(A, p=probs)
#                     actions.append(action)

#                 acts = [act_dic[a] for a in actions]
#                 rewards = get_reward(state_dic[state], acts)
#                 next_state = get_next_state(state, actions, episode)
#                 episode_reward += sum(rewards)

#                 for i in range(N):
#                     a = actions[i]
#                     N_sa[i][state][a] += 1
#                     best_next_action = np.argmax(Q[i][next_state])
#                     td_target = rewards[i] + gamma * Q[i][next_state][best_next_action]
#                     alpha = 1 / ((step + 1) ** 0.85)
#                     Q[i][state][a] += alpha * (td_target - Q[i][state][a])
#                 state = next_state

#             run_episode_rewards.append(episode_reward)
#             total_reward += episode_reward
#             pi = q_to_policy(Q, tau)
#             policy_hist.append(copy.deepcopy(pi))

#         final_pi = policy_hist[-1]
#         run_accs = [policy_accuracy(pi, final_pi) for pi in policy_hist]
#         all_accuracies.append(run_accs)
#         all_final_policies.append(final_pi)
#         all_total_rewards.append(total_reward)
#         all_episode_rewards.append(run_episode_rewards)

#     os.makedirs("./npy/greedy", exist_ok=True)
#     os.makedirs("./pic/greedy", exist_ok=True)

#     np.save("./npy/greedy/greedy_accuracies.npy", np.array(all_accuracies))
#     np.save("./npy/greedy/greedy_rewards.npy", np.array(all_total_rewards))
#     np.save("./npy/greedy/greedy_densities.npy", policy_to_facility_density(all_final_policies[-1], act_dic, state_dic))
#     np.save("./npy/greedy/greedy_plot_matrix.npy", np.array(list(itertools.zip_longest(*all_accuracies, fillvalue=np.nan))).T)
#     np.save("./npy/greedy/greedy_episode_rewards.npy", np.array(all_episode_rewards))

#     piters = list(range(len(all_accuracies[0])))
#     plot_accuracies = np.array(list(itertools.zip_longest(*all_accuracies, fillvalue=np.nan))).T
#     fig1 = plt.figure()
#     for i in range(len(plot_accuracies)):
#         plt.plot(piters, plot_accuracies[i])
#     plt.xlabel("Iterations")
#     plt.ylabel("L1-accuracy")
#     plt.title("Greedy: individual runs")
#     plt.grid(True)
#     plt.savefig("./pic/greedy/greedy_individual_runs.png", dpi=300)
#     plt.close()

#     pmean = list(map(statistics.mean, zip(*plot_accuracies)))
#     pstdv = [statistics.stdev(list(col)) for col in zip(*plot_accuracies)]
#     np.save("./npy/greedy/greedy_avg_mean.npy", np.array(pmean))
#     np.save("./npy/greedy/greedy_avg_std.npy", np.array(pstdv))
#     fig2 = plt.figure()
#     clrs = sns.color_palette("husl", 3)
#     ax = sns.lineplot(x=piters, y=pmean, color=clrs[0], label='Mean L1-accuracy')
#     ax.fill_between(piters, np.subtract(pmean, pstdv), np.add(pmean, pstdv), alpha=0.3, facecolor=clrs[0])
#     ax.legend()
#     plt.grid(True)
#     plt.xlabel("Iterations")
#     plt.ylabel("L1-accuracy")
#     plt.title("Greedy: average accuracy")
#     plt.savefig("./pic/greedy/greedy_avg_runs.png", dpi=300)
#     plt.close()

#     final_density = policy_to_facility_density(all_final_policies[-1], act_dic, state_dic)
#     fig3, ax = plt.subplots()
#     index = np.arange(safe_state.m)
#     bar_width = 0.35
#     plt.bar(index, final_density[0], bar_width, alpha=0.7, color='b', label='Safe state')
#     plt.bar(index + bar_width, final_density[1], bar_width, alpha=1.0, color='r', label='Distancing state')
#     plt.xlabel("Facility")
#     plt.ylabel("Average number of agents")
#     plt.title("Greedy: facility density")
#     plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("./pic/greedy/greedy_facilities.png", dpi=300)
#     plt.close()

#     fig4 = plt.figure()
#     plt.plot(all_total_rewards, marker='o')
#     plt.xlabel("Run")
#     plt.ylabel("Cumulative Reward")
#     plt.title("Greedy: Cumulative Reward Across Runs")
#     plt.grid(True)
#     plt.savefig("./pic/greedy/greedy_cumulative_reward.png", dpi=300)
#     plt.close()

#     optimal_per_episode = N * 6
#     regret_matrix = np.cumsum(optimal_per_episode - np.array(all_episode_rewards), axis=1)
#     np.save("./npy/greedy/greedy_episode_regret.npy", regret_matrix)

#     fig5 = plt.figure()
#     for r in range(runs):
#         plt.plot(range(M), regret_matrix[r], alpha=0.6)
#     plt.xlabel("Episode")
#     plt.ylabel("Cumulative Regret")
#     plt.title("Greedy: Cumulative Regret per Episode")
#     plt.grid(True)
#     plt.savefig("./pic/greedy/greedy_episode_regret.png", dpi=300)
#     plt.close()

#     return fig1, fig2, fig3, fig4, fig5

# if __name__ == '__main__':
#     start = process_time()
#     greedy_congestion_experiment(N=8, H=20, M=1001, epsilon=0.1, runs=10)
#     print("Done. Time elapsed:", process_time() - start)
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statistics
import itertools
from congestion_games import *
from time import process_time

# def q_to_policy(Q, tau=1.0):
#     policy = np.zeros_like(Q)
#     for s in range(Q.shape[1]):
#         for a in range(Q.shape[2]):
#             policy[:, s, a] = np.exp(Q[:, s, a] * tau)
#     policy_sum = np.sum(policy, axis=2, keepdims=True)
#     return policy / policy_sum
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

def q_learning_epsilon_greedy_experiment(N=8, H=20, M=1001, gamma=0.99, epsilon=0.1, tau=1.0, runs=10):
    safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
    distancing_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
    # state_dic = {0: safe_state, 1: distancing_state}
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
    act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}

    def get_next_state(state, actions, t):
        acts = [act_dic[a] for a in actions]
        density = state_dic[state].get_counts(acts)
        max_density = max(density)
        if state == 0 and max_density > N / 2:
            return 1
        elif state == 1 and max_density <= N / 4:
            return 0
        return state

    all_accuracies = []
    all_final_policies = []
    all_total_rewards = []
    all_episode_rewards = []

    for run in range(runs):
        Q = np.ones((N, S, A))
        policy_hist = []
        total_reward = 0
        run_episode_rewards = []

        for episode in range(M):
            state = 0
            episode_reward = 0
            index = ((episode % 160) // 40) % 4
            safe_weights = safe_reward_options[index]
            distancing_weights = distancing_reward_options[index]
            safe_state = CongGame(N, 1, safe_weights)
            distancing_state = CongGame(N, 1, distancing_weights)
            state_dic = {0: safe_state, 1: distancing_state}
            
            act_dic = {idx: act for idx, act in enumerate(safe_state.actions)}


            for step in range(H):
                actions = []
                for i in range(N):
                    if random.random() < epsilon:
                        action = random.randint(0, A - 1)
                    else:
                        action = np.argmax(Q[i][state])
                    actions.append(action)

                acts = [act_dic[a] for a in actions]
                rewards = get_reward(state_dic[state], acts)
                next_state = get_next_state(state, actions, episode)

                episode_reward += sum(rewards)

                for i in range(N):
                    a = actions[i]
                    max_q = np.max(Q[i][next_state])
                    Q[i][state][a] = (1 - 0.1) * Q[i][state][a] + 0.1 * (rewards[i] + gamma * max_q)

                state = next_state

            run_episode_rewards.append(episode_reward)
            total_reward += episode_reward
            pi = q_to_policy(Q, tau)
            policy_hist.append(np.copy(pi))

        final_pi = policy_hist[-1]
        run_accs = [policy_accuracy(pi, final_pi) for pi in policy_hist]
        all_accuracies.append(run_accs)
        all_final_policies.append(final_pi)
        all_total_rewards.append(total_reward)
        all_episode_rewards.append(run_episode_rewards)

    os.makedirs("./npy/epsilon_greedy", exist_ok=True)
    os.makedirs("./pic/epsilon_greedy", exist_ok=True)

    np.save("./npy/epsilon_greedy/epsilon_greedy_accuracies.npy", np.array(all_accuracies))
    np.save("./npy/epsilon_greedy/epsilon_greedy_rewards.npy", np.array(all_total_rewards))
    np.save("./npy/epsilon_greedy/epsilon_greedy_densities.npy", policy_to_facility_density(all_final_policies[-1], act_dic, state_dic))
    np.save("./npy/epsilon_greedy/epsilon_greedy_plot_matrix.npy", np.array(list(itertools.zip_longest(*all_accuracies, fillvalue=np.nan))).T)
    np.save("./npy/epsilon_greedy/epsilon_greedy_episode_rewards.npy", np.array(all_episode_rewards))

    piters = list(range(len(all_accuracies[0])))
    plot_accuracies = np.array(list(itertools.zip_longest(*all_accuracies, fillvalue=np.nan))).T
    fig1 = plt.figure()
    for i in range(len(plot_accuracies)):
        plt.plot(piters, plot_accuracies[i])
    plt.xlabel("Iterations")
    plt.ylabel("L1-accuracy")
    plt.title("Epsilon-Greedy: individual runs")
    plt.grid(True)
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_individual_runs.png", dpi=300)
    plt.close()

    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = [statistics.stdev(list(col)) for col in zip(*plot_accuracies)]
    np.save("./npy/epsilon_greedy/epsilon_greedy_avg_mean.npy", np.array(pmean))
    np.save("./npy/epsilon_greedy/epsilon_greedy_avg_std.npy", np.array(pstdv))
    fig2 = plt.figure()
    clrs = sns.color_palette("husl", 3)
    ax = sns.lineplot(x=piters, y=pmean, color=clrs[0], label='Mean L1-accuracy')
    ax.fill_between(piters, np.subtract(pmean, pstdv), np.add(pmean, pstdv), alpha=0.3, facecolor=clrs[0])
    ax.legend()
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("L1-accuracy")
    plt.title("Epsilon-Greedy: average accuracy")
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_avg_runs.png", dpi=300)
    plt.close()

    final_density = policy_to_facility_density(all_final_policies[-1], act_dic, state_dic)
    fig3, ax = plt.subplots()
    index = np.arange(safe_state.m)
    bar_width = 0.35
    plt.bar(index, final_density[0], bar_width, alpha=0.7, color='b', label='Safe state')
    plt.bar(index + bar_width, final_density[1], bar_width, alpha=1.0, color='r', label='Distancing state')
    plt.xlabel("Facility")
    plt.ylabel("Average number of agents")
    plt.title("Epsilon-Greedy: facility density")
    plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_facilities.png", dpi=300)
    plt.close()

    fig4 = plt.figure()
    plt.plot(all_total_rewards, marker='o')
    plt.xlabel("Run")
    plt.ylabel("Cumulative Reward")
    plt.title("Epsilon-Greedy: Cumulative Reward Across Runs")
    plt.grid(True)
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_cumulative_reward.png", dpi=300)
    plt.close()

    optimal_per_episode = N * 6
    regret_matrix = np.cumsum(optimal_per_episode - np.array(all_episode_rewards), axis=1)
    np.save("./npy/epsilon_greedy/epsilon_greedy_episode_regret.npy", regret_matrix)

    fig5 = plt.figure()
    for r in range(runs):
        plt.plot(range(M), regret_matrix[r], alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")
    plt.title("Epsilon-Greedy: Cumulative Regret per Episode")
    plt.grid(True)
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_episode_regret.png", dpi=300)
    plt.close()

    episode_means = np.mean(all_episode_rewards, axis=0)
    episode_stds = np.std(all_episode_rewards, axis=0)

    plt.figure()
    plt.plot(range(M), episode_means, label="Mean")
    plt.fill_between(range(M), episode_means - episode_stds, episode_means + episode_stds, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Epsilon-Greedy: Episode Reward (mean ± std)")
    plt.grid(True)
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_mean_std_episode_reward.png", dpi=300)
    plt.close()
    all_agent_rewards = np.array(all_episode_rewards)  # shape = (runs, M)
    agent_rewards_per_episode = np.mean(all_agent_rewards, axis=0).reshape(1, -1)  # shape = (1, M)
    agent_cum_rewards = np.cumsum(agent_rewards_per_episode, axis=1)
    total_reward_per_episode = np.sum(all_agent_rewards, axis=0)  # shape = (M,)
    total_cum_reward = np.cumsum(total_reward_per_episode)

    np.save("./npy/epsilon_greedy/epsilon_greedy_agent_reward_per_episode.npy", agent_rewards_per_episode)
    np.save("./npy/epsilon_greedy/epsilon_greedy_agent_cumulative_reward.npy", agent_cum_rewards)
    np.save("./npy/epsilon_greedy/epsilon_greedy_total_reward_per_episode.npy", total_reward_per_episode)
    np.save("./npy/epsilon_greedy/epsilon_greedy_total_cumulative_reward.npy", total_cum_reward)

    plt.figure()
    plt.plot(range(M), agent_rewards_per_episode[0])
    plt.xlabel("Episode")
    plt.ylabel("Average Agent Reward")
    plt.title("Epsilon-Greedy: Agent Reward per Episode")
    plt.grid(True)
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_agent_reward_per_episode.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(range(M), agent_cum_rewards[0])
    plt.xlabel("Episode")
    plt.ylabel("Agent Cumulative Reward")
    plt.title("Epsilon-Greedy: Agent Cumulative Reward")
    plt.grid(True)
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_agent_cumulative_reward.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(range(M), total_reward_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (All Agents)")
    plt.title("Epsilon-Greedy: Total Reward per Episode")
    plt.grid(True)
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_total_reward_per_episode.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(range(M), total_cum_reward)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Total Reward")
    plt.title("Epsilon-Greedy: Cumulative Total Reward")
    plt.grid(True)
    plt.savefig("./pic/epsilon_greedy/epsilon_greedy_total_cumulative_reward.png", dpi=300)
    plt.close()
    

if __name__ == '__main__':
    start = process_time()
    q_learning_epsilon_greedy_experiment(N=8, H=20, M=5001, epsilon=0.1, runs=10)
    print("Done. Time elapsed:", process_time() - start)