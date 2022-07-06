import gym
import numpy as np
import utils
import algorithms
import time

env = gym.make ('CliffWalking-v0')
# #0:UP 1:RIGHT 2:DOWN 3:LEFT

n_actions = env.action_space.n
n_states = env.observation_space.n

init_Q=np.zeros([n_states,n_actions])

st=time.time()
policy=algorithms.double_q_learning(env,500,1,0.1,0.5,episode_maxlength=1000,init_Q=init_Q) #,show_Qvalues=True)
print("Execution time :",time.time()-st,"seconds")

st=time.time()
policy=algorithms.q_learning(env,500,1,0.1,0.5,episode_maxlength=1000,init_Q=init_Q) #,show_Qvalues=True)
print("Execution time :",time.time()-st,"seconds")

utils.save_policy(policy,"results/cliff_double_q_learning.csv")
utils.evaluate_policy(policy,env,1)

