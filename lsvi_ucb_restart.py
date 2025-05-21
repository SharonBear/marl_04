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
import math


def phi(s, a, S, A):
    vec = np.zeros(S * A)
    vec[s * A + a] = 1.0
    return vec


def lsvi_ucb_restart_experiment(N=8, H=20, M=1001, gamma=0.99, runs=10, variation=5):
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
        threshold = random.randint(0, N)
        if state == 0 and max_density > threshold:
            return 1
        elif state == 1 and max_density <= N / 4:
            return 0
        return state

    os.makedirs("./npy/lsvi_ucb_restart", exist_ok=True)
    os.makedirs("./pic/lsvi_ucb_restart", exist_ok=True)

    all_episode_rewards = []
    all_cumulative_rewards = []
    all_regrets = []

    optimal_reward = N * H * 6
    for run in range(runs):
        episode_rewards = np.zeros(M)
        histS = [[0 for _ in range(H)] for _ in range(M)]
        histA = [[0 for _ in range(H)] for _ in range(M)]
        histR = np.zeros((M, H))
        Q = np.ones((M, H, S, A))
        D = variation
        K = M // D

        for d in range(D):
            tau = d * K
            for i_episode in range(K):
                k = tau + i_episode
                state = 0
                for h in range(H - 1, -1, -1):
                    lam = np.eye(S * A)
                    for l in range(tau, k):
                        vec = phi(histS[l][h], histA[l][h], S, A)
                        lam += np.outer(vec, vec)
                    tmp = np.zeros(S * A)
                    for l in range(tau, k):
                        if h == H - 1:
                            tmp += phi(histS[l][h], histA[l][h], S, A) * histR[l][h]
                        else:
                            tmp += phi(histS[l][h], histA[l][h], S, A) * (histR[l][h] + np.max(Q[k - 1][h + 1][histS[l][h + 1]]))
                    w = np.linalg.inv(lam) @ tmp
                    for s in range(S):
                        for a in range(A):
                            vec = phi(s, a, S, A)
                            norm = math.sqrt(vec.T @ np.linalg.inv(lam) @ vec)
                            beta = 0.4
                            Q[k][h][s][a] = min(w.T @ vec + beta * norm, H)

                episode_reward = 0
                for h in range(H):
                    actions = []
                    for i in range(N):
                        action = np.argmax(Q[k][h][state])
                        actions.append(action)

                    acts = [act_dic[a] for a in actions]
                    rewards = get_reward(state_dic[state], acts)
                    next_state = get_next_state(state, actions, k)
                    r = sum(rewards)

                    episode_reward += r
                    histS[k][h] = int(state)
                    histA[k][h] = int(np.argmax(Q[k][h][state]))
                    histR[k][h] = r
                    state = next_state

                episode_rewards[k] = episode_reward

        all_episode_rewards.append(episode_rewards)
        all_cumulative_rewards.append(np.cumsum(episode_rewards))
        all_regrets.append(np.cumsum(optimal_reward - episode_rewards))

    np.save("./npy/lsvi_ucb_restart/episode_rewards.npy", np.array(all_episode_rewards))
    np.save("./npy/lsvi_ucb_restart/cumulative_rewards.npy", np.array(all_cumulative_rewards))
    np.save("./npy/lsvi_ucb_restart/cumulative_regrets.npy", np.array(all_regrets))

    fig1 = plt.figure()
    for r in all_episode_rewards:
        plt.plot(r)
    plt.title("LSVI-UCB Restart: Per Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("./pic/lsvi_ucb_restart/episode_rewards.png", dpi=300)
    plt.close()

    fig2 = plt.figure()
    for r in all_regrets:
        plt.plot(r)
    plt.title("LSVI-UCB Restart: Cumulative Regret")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")
    plt.grid(True)
    plt.savefig("./pic/lsvi_ucb_restart/cumulative_regret.png", dpi=300)
    plt.close()

    fig3 = plt.figure()
    for r in all_cumulative_rewards:
        plt.plot(r)
    plt.title("LSVI-UCB Restart: Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.savefig("./pic/lsvi_ucb_restart/cumulative_reward.png", dpi=300)
    plt.close()

    return fig1, fig2, fig3


if __name__ == '__main__':
    start = process_time()
    lsvi_ucb_restart_experiment(N=8, H=20, M=1001, runs=10, variation=5)
    print("Done. Time elapsed:", process_time() - start)
