'''
Testing Dynamic Programming algorithms on the Frozen Lake environment - Small & Discrete state & action space, non-deterministic
Policy Iteration & Value Iteration
The dynamics are known - these are planning (!= learning) algorithms
Even with a good strategy, it is impossible to win 100% of the time in this environment
'''

from random import seed
import gym
import numpy as np
import time


# Hyperparameters
discount = 1
epsilon = 0.05

# Setting a seed
RANDOM_SEED = 1
# env.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)


def get_probs(V, arr):
    res = []
    for i in arr:
        res.append(i[0]*(i[2]+discount*V[i[1]]))
    return sum(res)

def policy_evaluation(env, V, pi, discount):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    improvement = 1.0
    while improvement > 0.01:
        for s in range(n_states):
            oldV = V[s]

            # new V(s)
            V[s] = 0.0
            # for every action from s, every resulting state after a
            # this could be vectorized instead of for loops
            for a in range(n_actions):
                for P_s_a_next_s, next_s, reward, _ in env.env.P[s][a]:
                    V[s] += pi[s][a]*P_s_a_next_s*(reward+discount*V[next_s])

            improvement = np.abs(oldV-V[s])
    return V


def policy_improvement(env, V, pi):
    n_states = env.observation_space.n
    for s in range(n_states):

        sums = list(map(
            get_probs, [V for i in range(n_states)], list(
                env.env.P[s].values())
        ))

        best_action = np.argmax(sums)
        for i in range(len(pi[s])):
            if i == best_action:
                pi[s][i] = 1.
            else:
                pi[s][i] = 0.
    return pi


def policy_iteration(env, discount):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros([n_states])
    pi = np.ones([n_states, n_actions])/n_actions

    st = time.time()
    stable = False
    while not stable:
        stable = True
        V = policy_evaluation(env, V, pi, discount)

        oldPi = np.copy(pi)
        pi = policy_improvement(env, V, pi)
        if (oldPi != pi).any():
            stable = False
    print("     Computed optimal policy with policy iteration in",
          round(time.time()-st, 3), "seconds")
    return V, pi


def value_iteration(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros([n_states])
    pi = np.ones([n_states, n_actions])/n_actions

    st = time.time()
    delta = 1.0
    while delta > 0.0001:
        delta = 0.0
        for s in range(n_states):
            oldV = V[s]

            # candidates for V[s]
            cand = [0.0 for i in range(n_actions)]
            # for every action from s, every resulting state after a
            for a in range(n_actions):
                for P_s_a_next_s, next_s, reward, _ in env.env.P[s][a]:
                    cand[a] += (P_s_a_next_s*(reward+discount*V[next_s]))

            V[s] = max(cand)
            delta = max(delta, np.abs(oldV-V[s]))
    pi = policy_improvement(env, V, pi)
    print("     Computed optimal policy with value iteration in",
          round(time.time()-st, 3), "seconds")
    return V, pi


def test_policy(pi, render=False):
    state = env.reset()
    done = False
    score = 0
    step = 0
    while not done:
        if render:
            env.render()
        new_state, reward, done, _ = env.step(
            np.random.choice(n_actions, p=pi[state]))
        score += reward
        step += 1
        if step > 100:
            return score
        state = new_state
    return score


def evaluate_policy(pi, n_episodes=1000):
    total = 0.0
    for _ in range(n_episodes):
        total += test_policy(pi)
    average_score = total/n_episodes
    return average_score


env = gym.make('FrozenLake-v1')
n_states = env.observation_space.n
n_actions = env.action_space.n

print(" -   -   -   Testing value iteration & policy iteration algorithms on FrozenLake-v1  -   -   -\n")
print("Number of states :", n_states)
print("Number of actions :", n_actions, "\n")

# Evaluating a random policy for comparison
init_pi = np.ones([n_states, n_actions])/n_actions
V = np.zeros([n_states])
print("Success rate for random policy :", evaluate_policy(init_pi)*100, "%\n")

V,pi=value_iteration(env)
print("Success rate for value_iteration policy :",evaluate_policy(pi)*100, "%\n")

# to visualize the resulting policy
print("Testing value_iteration policy \n")
test_policy(pi,render=True)

V, pi = policy_iteration(env, discount)
print("Success rate for policy_iteration policy :",
      evaluate_policy(pi)*100, "%\n")

# to visualize the resulting policy
print("Testing policy_iteration policy \n")
test_policy(pi, render=True)
