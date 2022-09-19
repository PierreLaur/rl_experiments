'''
Tuning Q-Learning HPs with Random Search, Coarse to Fine on the Taxi-v3 environment
Still can't "solve" it with this algorithm
'''
import gym
import numpy as np
import utils
import TD_algorithms
import time

env = gym.make ('Taxi-v3')
# #0:UP 1:RIGHT 2:DOWN 3:LEFT

n_actions = env.action_space.n
n_states = env.observation_space.n

num_episodes=20000

# plot parameters
xlims=(0,num_episodes)
ylims=(-100,10)
EWMA_factor=0.95 # curve smoothing (exponentially weighted moving average)

print(" -   -   -   Tuning four Q-Learning HPs with Random Search on the Taxi-v3 environment    -   -   -\n")
print("Discount, step size, exploration rate & exploration rate decay rate")


# Q-Learning HP Tuning

def try_hps (discount,step_size,exploration_rate,decay_rate,plot_return=False,color='b') :
    st=time.time()
    policy=TD_algorithms.q_learning     (env,num_episodes,discount,step_size,exploration_rate,stepsize_decay_rate=decay_rate,exploration_decay_rate=decay_rate,episode_maxlength=1000,
                                plot_return=plot_return, plot_color=color,plot_label=str(HPs),xlims=xlims,ylims=ylims,EWMA_factor=EWMA_factor) #,show_Qvalues=True)
    return utils.evaluate_policy(policy,env,100),round(time.time()-st,1)

def generate_random_HPs(num,ranges) :
    if num!=len(ranges) :
        print('Error - Different number of HPs and ranges')
        return 1
    HPs = np.random.rand(num)
    for i,r in enumerate(ranges) :
        HPs[i]*=abs(r[1]-r[0])
        HPs[i]+=r[0]
    return tuple(HPs.round(6))

# # Baseline
# st=time.time()
# policy=TD_algorithms.q_learning     (env,num_episodes,1,0.8,0.1,0.001,episode_maxlength=1000,
#                             plot_return=True, plot_color='r--',plot_label='q-learning',xlims=xlims,ylims=ylims,EWMA_factor=EWMA_factor) #,show_Qvalues=True)
# utils.evaluate_policy(policy,env,100),round(time.time()-st,1)


colors=[c+l for (c,l) in zip('bgrcmykbgrcmyk','-------:::::::')]
results=[]
hprange=[[0.8,1], [0.01,1], [0.01,0.5], [0.00001,0.001]]
for i in range(14) :
    HPs=generate_random_HPs(4,hprange)
    print("\nTrying HPs",HPs)
    res,exec_time=try_hps(*HPs,plot_return=True,color=colors[i]) 
    results.append([*HPs,res,exec_time])

# np.savetxt('results/hptuning.txt',np.array(results),fmt='%f')

# time.sleep(5)

