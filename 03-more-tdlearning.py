'''
Further testing for TD algorithms on a bigger & random environment - Taxi-v3
- Comparing the 5 TD algorithms with fixed HPs :
    - None of them actually "solve" the problem
    - Q-Learning is fastest
    - SARSA doesn't seem to converge, even after some HP tuning
    - Dyna-Q converges in less episodes than the rest but is SUPER SLOW (way more than 6 times slower with 5 model updates per interaction)
'''
import gym
import numpy as np
import utils
import TD_algorithms
import time
import matplotlib.pyplot as plt

env = gym.make ('Taxi-v3')
# #0:UP 1:RIGHT 2:DOWN 3:LEFT

n_actions = env.action_space.n
n_states = env.observation_space.n
init_Q=np.zeros([n_states,n_actions])

num_episodes=10000

# Hyperparameters (there are more)
discount=0.9
step_size=0.1
exploration_rate=0.1

# plot parameters
xlims=(0,num_episodes)
ylims=(-100,10)
EWMA_factor=0.90 # curve smoothing (exponentially weighted moving average)

print(" -   -   -   Testing TD Learning algorithms on Taxi-v3   -   -   -\n")
print("Taxi-v3 is considered solved with an average return of at least 9.7 over 100 consecutive trials\n")


# Comparing TD Learning algorithms on this problem
st=time.time()
policy=TD_algorithms.q_learning     (env,num_episodes,discount,step_size,exploration_rate,episode_maxlength=1000,
                            plot_return=True, plot_color='b',plot_label='Q-Learning',xlims=xlims,ylims=ylims,EWMA_factor=EWMA_factor) #,show_Qvalues=True)
print("     Execution time :",round(time.time()-st),"seconds")
utils.evaluate_policy(policy,env,100)                     #   to display the average score of the trained policy
print()

st=time.time()
policy=TD_algorithms.double_q_learning (env,num_episodes,discount,step_size,exploration_rate,episode_maxlength=1000, 
                            plot_return=True, plot_color='c',plot_label='Double Q-Learning',xlims=xlims,ylims=ylims,EWMA_factor=EWMA_factor)
print("     Execution time :",round(time.time()-st),"seconds")
utils.evaluate_policy(policy,env,100)                     #   to display the average score of the trained policy
print()

st=time.time()
policy=TD_algorithms.sarsa          (env,num_episodes,discount,step_size,exploration_rate,episode_maxlength=1000,
                            plot_return=True, plot_color='g',plot_label='SARSA',xlims=xlims,ylims=ylims,EWMA_factor=EWMA_factor)
print("     Execution time :",round(time.time()-st),"seconds")
utils.evaluate_policy(policy,env,100)                     #   to display the average score of the trained policy
print()

st=time.time()
policy=TD_algorithms.expected_sarsa (env,num_episodes,discount,step_size,exploration_rate,episode_maxlength=1000, 
                            plot_return=True, plot_color='r',plot_label='Expected SARSA',xlims=xlims,ylims=ylims,EWMA_factor=EWMA_factor)
print("     Execution time :",round(time.time()-st),"seconds")
utils.evaluate_policy(policy,env,100)                     #   to display the average score of the trained policy
print()


# note : My Dyna-Q implementation is super slow here & there's no reason to use model updates instead of environment interactions in this context
# - it does converge in less episodes than the rest of the algorithms
st=time.time()
policy=TD_algorithms.dyna_q (env,num_episodes,discount,step_size,exploration_rate,episode_maxlength=1000, 
                            plot_return=True, plot_color='y',plot_label='Dyna-Q (3 model updates)',xlims=xlims,ylims=ylims,EWMA_factor=EWMA_factor,model_queries=3)
print("     Execution time :",round(time.time()-st,3),"seconds")
utils.evaluate_policy(policy,env,100)                     #   to display the average score of the trained policy
print()

plt.savefig('results/tdlearning_taxi.png')

# time.sleep(5)

# utils.save_policy(policy,"results/taxi_q_learning.csv")
# utils.evaluate_policy(policy,env,100)                     #   to display the average score of the trained policy

# utils.evaluate_policy(policy,env,1,render=True)           # to visualize the result policy

