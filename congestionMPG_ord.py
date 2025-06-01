from congestion_games import *
import matplotlib.pyplot as plt
import itertools
import numpy as np
import random
import copy
import statistics
import seaborn as sns; sns.set()
from time import process_time
import argparse
from datetime import datetime
myp_start = process_time()

def projection_simplex_sort(v, z=1):
    # Courtesy: EdwardRaff/projection_simplex.py
    if v.sum() == z and np.alltrue(v >= 0):
        return v
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

# Define the states and some necessary info
N = 8 #number of agents

safe_state = CongGame(N,1,[[1, 0], [2, 0], [4, 0], [60, 0]])
bad_state = CongGame(N,1,[[1, -100], [2, -100], [4, -100], [60, -100]])
state_dic = {0: safe_state, 1: bad_state}
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
M = safe_state.num_actions 
D = safe_state.m #number facilities
S = 2

# Dictionary to store the action profiles and rewards to
selected_profiles = {}

# Dictionary associating each action (value) to an integer (key)
act_dic = {}
counter = 0
for act in safe_state.actions:
    act_dic[counter] = act 
    counter += 1

def get_next_state(state, actions):
    acts_from_ints = [act_dic[i] for i in actions]
    density = state_dic[state].get_counts(acts_from_ints)
    max_density = max(density)
    threshold = N/2
    if state == 0 and max_density > N/2 or state == 1 and max_density > N/4:
      # if state == 0 and max_density > N/2 and np.random.uniform() > 0.2 or state == 1 and max_density > N/4 and np.random.uniform() > 0.1:
        return 1
    return 0

def pick_action(prob_dist):
    # np.random.choice(range(len(prob_dist)), 1, p = prob_dist)[0]
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action[0]

def visit_dist(state, policy, gamma, T,samples):
    # This is the unnormalized visitation distribution. Since we take finite trajectories, the normalization constant is (1-gamma**T)/(1-gamma).
    visit_states = {st: np.zeros(T) for st in range(S)}        
    for i in range(samples):
        curr_state = state
        for t in range(T):
            visit_states[curr_state][t] += 1
            actions = [pick_action(policy[curr_state, i]) for i in range(N)]
            curr_state = get_next_state(curr_state, actions)
    dist = [np.dot(v/samples,gamma**np.arange(T)) for (k,v) in visit_states.items()]
    return dist 

def value_function(policy, gamma, T,samples):
    value_fun = {(s,i):0 for s in range(S) for i in range(N)}
    for k in range(samples):
        for state in range(S):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]
                q = tuple(actions+[curr_state])
                rewards = selected_profiles.setdefault(q,get_reward(state_dic[curr_state], [act_dic[i] for i in actions]))                  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]
                curr_state = get_next_state(curr_state, actions)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(agent, state, action, policy, gamma, value_fun, samples):
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        actions[agent] = action
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q,get_reward(state_dic[state], [act_dic[i] for i in actions]))
        tot_reward += rewards[agent] + gamma*value_fun[get_next_state(state, actions), agent]
    return (tot_reward / samples)

def policy_accuracy(policy_pi, policy_star):
    total_dif = N * [0]
    for agent in range(N):
        for state in range(S):
            total_dif[agent] += np.sum(np.abs((policy_pi[state, agent] - policy_star[state, agent])))
      # total_dif[agent] += np.sqrt(np.sum((policy_pi[state, agent] - policy_star[state, agent])**2))
    return np.sum(total_dif) / N

def policy_gradient(mu, max_iters, gamma, eta, T, samples):

    policy = {(s,i): [1/M]*M for s in range(S) for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in range(max_iters):
        # 動態變換 state reward
        index = ((t % 40) // 10) % 4
        index = 0
        safe_weights = safe_reward_options[index]
        distancing_weights = distancing_reward_options[index]
        safe_state = CongGame(N, 1, safe_weights)
        distancing_state = CongGame(N, 1, distancing_weights)
        global state_dic  # 更新全域變數
        state_dic = {0: safe_state, 1: distancing_state}

        if t % 50 == 0:
            print(t)

        b_dist = M * [0]
        for st in range(S):
            a_dist = visit_dist(st, policy, gamma, T, samples)

            b_dist[st] = np.dot(a_dist, mu)
            
        grads = np.zeros((N, S, M))
        value_fun = value_function(policy, gamma, T, samples)
    
        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads[agent, st, act] = b_dist[st] * Q_function(agent, st, act, policy, gamma, value_fun, samples)

        for agent in range(N):
            for st in range(S):
                policy[st, agent] = projection_simplex_sort(np.add(policy[st, agent], eta[agent] * grads[agent,st]), z=1)
        policy_hist.append(copy.deepcopy(policy))

        # # 提早結束並補齊長度
        # if policy_accuracy(policy_hist[t], policy_hist[t-1]) < 1e-16:
        #     last_policy = policy_hist[-1]
        #     while len(policy_hist) < max_iters + 1:  # +1 是因為第 0 策略也算
        #         policy_hist.append(copy.deepcopy(last_policy))
        #     return policy_hist

    return policy_hist

def get_accuracies(policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies

def full_experiment(runs,iters,eta,T,samples):
    densities = np.zeros((S,M))
    raw_accuracies = []
    all_agent_rewards_per_episode = []
    all_agent_cum_rewards = []
    total_rewards_per_episode = []
    for k in range(runs):
        print(k)
        policy_hist = policy_gradient([0.5, 0.5],iters,0.99,eta,T,samples)
        raw_accuracies.append(get_accuracies(policy_hist))

        converged_policy = policy_hist[-1]
        agent_rewards = np.zeros((N, T))
        total_rewards = np.zeros(T)

        for ep in range(T):
            curr_state = 0
            ep_rewards = np.zeros(N)
            for t in range(1):
                actions = [pick_action(converged_policy[curr_state, i]) for i in range(N)]
                rewards = get_reward(state_dic[curr_state], [act_dic[i] for i in actions])
                ep_rewards += rewards
                curr_state = get_next_state(curr_state, actions)
            agent_rewards[:, ep] = ep_rewards
            total_rewards[ep] = np.sum(ep_rewards)

        all_agent_rewards_per_episode.append(agent_rewards)
        all_agent_cum_rewards.append(np.cumsum(agent_rewards, axis=1))
        total_rewards_per_episode.append(total_rewards)

        for i in range(N):
            for s in range(S):
                densities[s] += converged_policy[s,i]

    densities = densities / runs

    plot_accuracies = np.array(list(itertools.zip_longest(*raw_accuracies, fillvalue=np.nan))).T
    clrs = sns.color_palette("husl", 3)
    piters = list(range(plot_accuracies.shape[1]))

    fig2 = plt.figure(figsize=(6,4))
    for i in range(len(plot_accuracies)):
        plt.plot(piters, plot_accuracies[i])
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = 0.0001'.format(N, runs))
    plt.show()
    fig2.savefig('./pic/ord/individual_runs_n{}.png'.format(N),bbox_inches='tight')
    #plt.close()
    
    plot_accuracies = np.nan_to_num(plot_accuracies)
    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))
    
    fig1 = plt.figure(figsize=(6,4))
    # ax = sns.lineplot(piters, pmean, color = clrs[0],label= 'Mean L1-accuracy')
    ax = sns.lineplot(x=piters, y=pmean, color=clrs[0], label='Mean L1-accuracy')
    ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[0],label="1-standard deviation")
    ax.legend()
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = 0.0001'.format(N, runs))
    plt.show()
    fig1.savefig('./pic/ord/avg_runs_n{}.png'.format(N),bbox_inches='tight')
    #plt.close()
    
    #print(densities)

    fig3, ax = plt.subplots()
    index = np.arange(D)
    bar_width = 0.35
    opacity = 1

    #print(len(index))
    #print(len(densities[0]))
    rects1 = plt.bar(index, densities[0], bar_width,
    alpha= .7 * opacity,
    color='b',
    label='Safe state')

    rects2 = plt.bar(index + bar_width, densities[1], bar_width,
    alpha= opacity,
    color='r',
    label='Distancing state')

    plt.gca().set(xlabel='Facility',ylabel='Average number of agents', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = 0.0001'.format(N,runs))
    plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
    plt.legend()
    fig3.savefig('./pic/ord/facilities_n{}.png'.format(N),bbox_inches='tight')
   #plt.close()
    plt.show()
    np.save("./npy/ord/ord_accuracies.npy", plot_accuracies)
    np.save("./npy/ord/ord_avg_mean.npy", np.array(pmean))
    np.save("./npy/ord/ord_avg_std.npy", np.array(pstdv))
    np.save("./npy/ord/ord_facility_density.npy", densities)
    # agent_reward_mean = np.mean(np.array(all_agent_rewards_per_episode), axis=0)
    # agent_cum_reward_mean = np.mean(np.array(all_agent_cum_rewards), axis=0)
    # total_reward_mean = np.mean(np.array(total_rewards_per_episode), axis=0)
    # total_cum_reward_mean = np.cumsum(total_reward_mean)

    # fig4 = plt.figure()
    # for i in range(N):
    #     plt.plot(range(T), agent_reward_mean[i])
    # plt.xlabel("Episode")
    # plt.ylabel("Agent Reward")
    # plt.title("Per-Agent Total Reward per Episode")
    # plt.grid(True)
    # fig4.savefig("./pic/ord/ord_agent_reward_per_episode.png", dpi=300)
    # plt.close()

    # fig5 = plt.figure()
    # for i in range(N):
    #     plt.plot(range(T), agent_cum_reward_mean[i])
    # plt.xlabel("Episode")
    # plt.ylabel("Agent Cumulative Reward")
    # plt.title("Per-Agent Cumulative Reward")
    # plt.grid(True)
    # fig5.savefig("./pic/ord/ord_agent_cumulative_reward.png", dpi=300)
    # plt.close()

    # fig6 = plt.figure()
    # plt.plot(range(T), total_reward_mean)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward (All Agents)")
    # plt.title("Total Reward per Episode")
    # plt.grid(True)
    # fig6.savefig("./pic/ord/ord_total_reward_per_episode.png", dpi=300)
    # plt.close()

    # fig7 = plt.figure()
    # plt.plot(range(T), total_cum_reward_mean)
    # plt.xlabel("Episode")
    # plt.ylabel("Cumulative Total Reward (All Agents)")
    # plt.title("Cumulative Total Reward")
    # plt.grid(True)
    # fig7.savefig("./pic/ord/ord_total_cumulative_reward.png", dpi=300)
    # plt.close()

    # iteration-based rewards
    agent_reward_iteration_mean = np.zeros((N, len(policy_hist)))
    agent_cum_reward_iteration_mean = np.zeros((N, len(policy_hist)))
    total_reward_iteration_mean = np.zeros(len(policy_hist))
    total_cum_reward_iteration_mean = np.zeros(len(policy_hist))

    for t_idx, policy in enumerate(policy_hist):
        all_rewards = np.zeros(N)
        for _ in range(samples):
            curr_state = 0
            actions = [pick_action(policy[curr_state, i]) for i in range(N)]
            rewards = get_reward(state_dic[curr_state], [act_dic[i] for i in actions])
            all_rewards += rewards
        agent_reward_iteration_mean[:, t_idx] = all_rewards / samples
        total_reward_iteration_mean[t_idx] = np.sum(all_rewards) / samples

    agent_cum_reward_iteration_mean = np.cumsum(agent_reward_iteration_mean, axis=1)
    total_cum_reward_iteration_mean = np.cumsum(total_reward_iteration_mean)

    fig8 = plt.figure()
    for i in range(N):
        plt.plot(range(len(policy_hist)), agent_reward_iteration_mean[i])
    plt.xlabel("Iteration")
    plt.ylabel("Agent Reward")
    plt.title("Per-Agent Total Reward per Iteration")
    plt.grid(True)
    fig8.savefig("./pic/ord/ord_agent_reward_per_iteration.png", dpi=300)
    plt.close()

    fig9 = plt.figure()
    for i in range(N):
        plt.plot(range(len(policy_hist)), agent_cum_reward_iteration_mean[i])
    plt.xlabel("Iteration")
    plt.ylabel("Agent Cumulative Reward")
    plt.title("Per-Agent Cumulative Reward per Iteration")
    plt.grid(True)
    fig9.savefig("./pic/ord/ord_agent_cumulative_reward.png", dpi=300)
    plt.close()

    fig10 = plt.figure()
    plt.plot(range(len(policy_hist)), total_reward_iteration_mean)
    plt.xlabel("Iteration")
    plt.ylabel("Total Reward (All Agents)")
    plt.title("Total Reward per Iteration")
    plt.grid(True)
    fig10.savefig("./pic/ord/ord_total_reward_per_iteration.png", dpi=300)
    plt.close()

    fig11 = plt.figure()
    plt.plot(range(len(policy_hist)), total_cum_reward_iteration_mean)
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Total Reward")
    plt.title("Cumulative Total Reward per Iteration")
    plt.grid(True)
    fig11.savefig("./pic/ord/ord_total_cumulative_reward.png", dpi=300)
    plt.close()

    # np.save("./npy/ord/ord_agent_reward_per_episode.npy", agent_reward_mean)
    # np.save("./npy/ord/ord_agent_cumulative_reward.npy", agent_cum_reward_mean)
    # np.save("./npy/ord/ord_total_reward_per_episode.npy", total_reward_mean)
    # np.save("./npy/ord/ord_total_cumulative_reward.npy", total_cum_reward_mean)

    np.save("./npy/ord/ord_agent_reward_per_iteration.npy", agent_reward_iteration_mean)
    np.save("./npy/ord/ord_agent_cumulative_reward.npy", agent_cum_reward_iteration_mean)
    np.save("./npy/ord/ord_total_reward_per_iteration.npy", total_reward_iteration_mean)
    np.save("./npy/ord/ord_total_cumulative_reward.npy", total_cum_reward_iteration_mean)

    return fig1, fig2, fig3

# #full_experiment(10,1000,0.0001,20,10)
# eta = [.0001 for i in range(N)]

# full_experiment(10,10000,eta,20,10)

# myp_end = process_time()
# elapsed_time = myp_end - myp_start
# print(elapsed_time)



if __name__ == '__main__':
    start = process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default="5000",
                        help="Choose reward episode: default=5001")
    args = parser.parse_args()

    log_lines = []
    currentDateAndTime = datetime.now()
    formatted_time = currentDateAndTime.strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"<MPG_ord> {formatted_time}")
    log_lines.append(f"Episode: {args.m}")
    et1 = [.0001 for i in range(N)]
    full_experiment(runs=10,iters=args.m,eta=et1,T=80,samples=10)
    log_lines.append(f"Done. Time elapsed: {(process_time() - start):.4f} seconds\n")

    with open("log.txt", "a") as f:
        for line in log_lines:
            f.write(line + "\n")