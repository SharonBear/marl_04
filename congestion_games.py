import itertools as it
import numpy as np
from math import comb

class CongGame:
	#inputs: num players, max facilities per player, list of linear multiplier on utility for num of players
	def __init__(self, n, d, weights):
		self.n = n
		self.d = d
		self.weights = weights
		self.m = len(weights) #number of facilities
		self.num_actions = sum(comb(self.m,k) for k in range(1,self.d+1))
		self.facilities = [i for i in range(self.m)]
		self.actions = list(it.chain.from_iterable(it.combinations(self.facilities, r) for r in range(1,self.d+1)))

	def get_counts(self, actions):
		count = dict.fromkeys(range(self.m),0)
		for action in actions:
			for facility in action:
				count[facility] += 1
		return list(count.values())

	def get_facility_rewards(self, actions):
		density = self.get_counts(actions)
		facility_rewards = self.m * [0]
		for j in range(self.m):
			facility_rewards[j] = density[j] * self.weights[j][0] + self.weights[j][1]
		return facility_rewards

def get_agent_reward(cong_game, actions, agent_action):
	agent_reward = 0
	facility_rewards = cong_game.get_facility_rewards(actions)
	for facility in agent_action:
		agent_reward += facility_rewards[facility]
	return agent_reward

def get_reward(cong_game, actions):
	rewards = cong_game.n * [0]
	for i in range(cong_game.n):
		rewards[i] = get_agent_reward(cong_game, actions, actions[i])
	return rewards

def get_next_state(state, actions):
    N = len(actions)
    act_dic = {}
    counter = 0
    m = len(actions[0])
    all_facilities = list(set(f for a in actions for f in a))
    for act in it.chain.from_iterable(it.combinations(all_facilities, r) for r in range(1, m + 1)):
        act_dic[counter] = act
        counter += 1

    acts_from_ints = [act_dic[i] if isinstance(i, int) else i for i in actions]
    density = state_dic[state].get_counts(acts_from_ints)
    max_density = max(density)

    if state == 0 and max_density > N / 2 or state == 1 and max_density > N / 4:
        return 1
    return 0

# Construct action dictionary globally
act_dic = {}
def build_act_dic(game):
    global act_dic
    act_dic = {i: act for i, act in enumerate(game.actions)}

