'''
Implementing Bayes Search for HP optimization.
A first try, with the "probability of improvement" acquisition function
We fix all parameters to a good value except the step size, and show that the GaussianProcessRegressor achieves a decent approximation
    of the average return distribution - it finds out that the best values for this case are between 0.25 and 1.1 by trying between -2 and 2
Tests on Q-Learning, Taxi environment. Training limited to 1000 episodes
'''
import gym
import numpy as np
import utils
import TD_algorithms
from sklearn.gaussian_process import GaussianProcessRegressor
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

env = gym.make ('Taxi-v3')
# #0:UP 1:RIGHT 2:DOWN 3:LEFT

n_actions = env.action_space.n
n_states = env.observation_space.n

num_episodes=1000

# plot parameters
xlims=(0,num_episodes)
ylims=(-100,10)
EWMA_factor=0.95 # curve smoothing (exponentially weighted moving average)


def generate_random_HPs(num,ranges) :
    if num!=len(ranges) :
        print('Error - Different number of HPs and ranges')
        return 1
    HPs = np.random.rand(num)
    for i,r in enumerate(ranges) :
        HPs[i]*=abs(r[1]-r[0])
        HPs[i]+=r[0]
    return tuple(HPs.round(6))

# The function from HP space to R (average return) we want to optimize :
def objective(discount,step_size,exploration_rate,decay_rate) :
    policy=TD_algorithms.q_learning     (env,num_episodes,discount,step_size,exploration_rate,stepsize_decay_rate=decay_rate,exploration_decay_rate=decay_rate,episode_maxlength=1000)
    return utils.evaluate_policy(policy,env,100)

def opt_acquisition(X,y,model,num_samples,num_hps,ranges) :
    # Generating random samples - this can be vectorized for better performance
    samples=np.ndarray((num_samples,num_hps))
    for i in range(num_samples) :
        samples[i]=generate_random_HPs(num_hps,ranges)
    # Scoring samples with the acquisition function
    scores = acquisition(X,samples,model)
    return samples[np.argmax(scores)]

# the surrogate function approximates the objective. We fit it to all our previous data (previous HP combinations tries)
# surrogate : "Approximately how good is this sample ?"
def surrogate(X,model) :
    yhat = model.predict(X,return_std=True)[0]
    return yhat

def fit_surrogate(X,y,model) :
    model.fit(X,y)

# acquisition : "Approximately how good is it to evaluate this sample ?"
def acquisition(X,samples,model,function='probability_of_improvement') :
    # Finding the best score so far, according to the surrogate
    if function=='surrogate' : return surrogate(samples,model)
    elif function=='probability_of_improvement' :
        # calculate the best surrogate score found so far
        if len(X) >0 :
            yhat, _ = model.predict(np.array(X).reshape(len(X),4),return_std=True)
            best = max(yhat)
        else : best=0
        # calculate mean and stdev via surrogate function
        mu, std = model.predict(samples,return_std=True)
        # calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std+1E-9))
        return probs
    else :
        print('not implemented yet - using surrogate instead')
        return surrogate(samples,model)

def print_stepsize_distribution (model,iteration) :
    srange = np.arange(-2, 2, 0.1)
    X = np.array([[1,s,0.1,0] for s in srange])
    yreal=[-451.0, -397.0, -334.0, -352.0, -424.0, -451.0, -442.0, -451.0, -496.0, -451.0, -550.0, -532.0, -550.0, -460.0, -550.0, -523.0, -586.0, -568.0, -667.0, -766.0, -163.0, -44.61, -5.58, 2.73, 6.44, 3.22, 6.83, 9.32, 8.9, 7.73, 9.0, 1.42, -267.9, -442.11, -418.24, -330.58, -628.93, -611.92, -559.36, -629.02]
    yhat = surrogate(X, model)
    colors=[c+l for c,l in zip('bgrcmyr','::::::-')]
    if iteration==0 :
        plt.figure(figsize=(7,3),dpi=200)
        plt.plot(srange,yreal,'k',label='Real')
        plt.xlabel('Step Size')
        plt.ylabel('Average return after training'+str(num_episodes)+'episodes')
        plt.xlim((-2,2))
        plt.ylim((-800,100))

    if iteration%2==1 :
        plt.plot(srange,yhat,colors[(iteration-1)//2],label='Estimation '+str(iteration))
        plt.legend()
        plt.title('Average return distribution as estimated by GP function')
        plt.show(block=False)
        plt.pause(1)


X=[]
y=[]
num_hps=4
# fixing every hp except the step size to good values
hpranges=[[1,1], [-2,2], [0.1,0.1], [0,0]]
model=GaussianProcessRegressor()

print(" -   -   -   Testing Bayesian Optimization on Taxi-v3    -   -   -\n")
print("Optimizing step size with limited samples")

# The HP optimization process :
for i in range(14) :

    # print the surrogate distribution for the step size
    print_stepsize_distribution(model,i)
    # select a point & evaluate it
    x = opt_acquisition(X, y,model,num_samples=10,num_hps=num_hps,ranges=hpranges)
    print("\nSampling point",x)
    result = objective(*x)
    print("     Result :",result)

    # add it to the dataset
    X.append(x)
    y.append(result)

    # update the surrogate function
    fit_surrogate(X,y,model)
plt.savefig('results/bayes_stepsize_approximation.png')
