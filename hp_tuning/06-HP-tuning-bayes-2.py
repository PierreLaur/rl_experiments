'''
More Bayes Search for HP optimization - 4 HPs, wider HP range, 1000 episodes training
With 10 samples, Bayes gets 8.6 avg max return, Random gets 7.
With only 5, Bayes gets 7.1, Random gets 4

Tests on Q-Learning, Taxi environment.
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

def opt_acquisition(X,y,model,num_samples,num_hps,ranges,acq_function) :
    # Generating random samples - this can be vectorized for better performance
    samples=np.ndarray((num_samples,num_hps))
    for i in range(num_samples) :
        samples[i]=generate_random_HPs(num_hps,ranges)
    # Scoring samples with the acquisition function
    scores = acquisition(X,samples,model,acq_function)
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
    elif function=='random' :
        return np.ones(samples.shape[0])/samples.shape[0]
    else :
        print('not implemented yet - using surrogate instead')
        return surrogate(samples,model)

def bayes_search(num_samples,acq_function='probability_of_improvement',display=True) :
    if display :
        print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        print("Performing Bayes search on {} samples to optimize hyperparameters. Acquisition function : {}\n".format(num_samples,acq_function))
    X=[]
    y=[]
    num_hps=4
    # fixing every hp except the step size to good values
    hpranges=[[0,1], [0,1], [0,1], [0,0.001]]
    model=GaussianProcessRegressor()

    # The HP optimization process :
    for i in range(num_samples) :
        # select a point & evaluate it
        x = opt_acquisition(X, y,model,num_samples=10,num_hps=num_hps,ranges=hpranges,acq_function=acq_function)
        if display : print("Sampling point",x)
        result = objective(*x)

        # add it to the dataset
        X.append(x)
        y.append(result)

        # update the surrogate function
        fit_surrogate(X,y,model)

    ix = np.argmax(y)
    if display : 
        print('\nBest Result : {} with HPs {}'.format(y[ix], X[ix]))
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")

    return y[ix]

print(' -   -   -   Comparing Bayes Optimization & Random Search on Taxi-v3   -   -   -\n')
print('Optimizing 4 hyperparameters with wide ranges')

bayes_search(20)
bayes_search(20,acq_function='random')