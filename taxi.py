import gym
import numpy as np
import utils
import algorithms

env = gym.make ('Taxi-v3')
# #0:UP 1:RIGHT 2:DOWN 3:LEFT

n_actions = env.action_space.n
n_states = env.observation_space.n

init_Q=np.zeros([n_states,n_actions])

policy=algorithms.double_q_learning(env,10000,1,0.1,0.5,episode_maxlength=1000,init_Q=init_Q) #,show_Qvalues=True)
utils.save_policy(policy,"results/taxi_double_q_learning.csv")
utils.evaluate_policy(policy,env,1,render=True)

