'''
Testing TD Learning algorithms on the CliffWalking environment - Small & Discrete state & action space, deterministic
On-policy Monte Carlo Control, SARSA, Expected SARSA, Q-Learning, Double Q-Learning, Dyna-Q
Unknown dynamics - actual learning algorithms.
The first 3 algorithms derive a non-optimal, "safe" policy (walking far from the cliff, despite not actually risking to fall)
More info about this in Sutton & Barto's RL book, page 132
'''

import gym
import numpy as np
import utils
import TD_algorithms
import time

env = gym.make ('CliffWalking-v0')
# 0:UP 1:RIGHT 2:DOWN 3:LEFT

n_actions = env.action_space.n
n_states = env.observation_space.n

# Hyperparameters
num_samples=500
discount=1
step_size=0.1
epsilon=0.5      # Exploration rate

# initial Q function
init_Q=np.zeros([n_states,n_actions])

print(" -   -   -   Testing TD Learning algorithms on CliffWalking-v0   -   -   -\n")
print("Optimal policy return : -12\n")

# These algorithms compute greedy policies (decaying epsilon in the case of MC) - in a deterministic environment, they always get the same behavior

st=time.time()
on_MC_policy=TD_algorithms.on_policy_mc_control(env,num_samples,discount,epsilon) #,show_Qvalues=True)
print("Execution time :",round(time.time()-st,3),"seconds")
utils.evaluate_policy(on_MC_policy,env,10)
print()


st=time.time()
sarsa_policy=TD_algorithms.sarsa(env,num_samples,discount,step_size,epsilon,init_Q=init_Q) #,show_Qvalues=True)
print("     Execution time :",round(time.time()-st,3),"seconds")
utils.evaluate_policy(sarsa_policy,env,1)
print()


st=time.time()
exp_policy=TD_algorithms.expected_sarsa(env,num_samples,discount,step_size,epsilon,init_Q=init_Q) #,show_Qvalues=True)
print("Execution time :",round(time.time()-st,3),"seconds")
utils.evaluate_policy(exp_policy,env,1)
print()


st=time.time()
q_policy=TD_algorithms.q_learning(env,num_samples,discount,step_size,epsilon,init_Q=init_Q) #,show_Qvalues=True)
print("     Execution time :",round(time.time()-st,3),"seconds")
utils.evaluate_policy(q_policy,env,1)
print()


st=time.time()
double_policy=TD_algorithms.double_q_learning(env,num_samples,discount,step_size,epsilon,init_Q=init_Q) #,show_Qvalues=True)
print("     Execution time :",round(time.time()-st,3),"seconds")
utils.evaluate_policy(double_policy,env,1)
print()


# I implemented early exit for dyna-q (if the optimal return is obtained, we stop training.)
st=time.time()
exp_policy=TD_algorithms.dyna_q(env,num_samples,discount,step_size,epsilon,model_queries=5,optimal_return=-13) #,show_Qvalues=True)
print("     Execution time :",round(time.time()-st,3),"seconds")
utils.evaluate_policy(exp_policy,env,1)
print()



# utils.evaluate_policy(policy,env,1, render=True)                  # to visualize the result
# utils.save_policy(policy,"results/cliff_double_q_learning.csv")   # to save the policy

