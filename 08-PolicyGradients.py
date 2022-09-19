'''
Trying out Policy Gradient algorithms
REINFORCE, with value function baseline
It does seem to start learning but doesn't converge (see results/REINFORCE* - 10k episodes run)
Visualize with tensorboard (tensorboard --logdir results/tboard)
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import time

# hide tensorflow message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

env = gym.make('CartPole-v1')

class policy_network(tf.keras.Model) :
    def __init__(self, l1_dim, l2_dim,nA) :
        super().__init__()
        self.l1 = tf.keras.layers.Dense(l1_dim, activation='relu')
        self.l2 = tf.keras.layers.Dense(l2_dim, activation='relu')
        self.out = tf.keras.layers.Dense(nA, activation = 'softmax')

    def call(self, input) :
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class value_network(tf.keras.Model) :
    def __init__(self,l1_dim,l2_dim,lr) :
        super().__init__()
        self.l1 = tf.keras.layers.Dense(l1_dim,activation='relu')
        self.l2 = tf.keras.layers.Dense(l2_dim,activation='relu')
        self.out = tf.keras.layers.Dense(1,activation=None)
        self.compile(optimizer=tf.keras.optimizers.Adam(lr))

    def call(self, input) :
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class agent() :
    def __init__(self, discount, lr, nA, l1_dim, l2_dim) :
        self.state_mem = []
        self.action_mem = []
        self.reward_mem = []

        self.nA = nA
        self.discount= discount
        self.lr = lr
        self.policy = policy_network(l1_dim,l2_dim,nA)
        self.policy.compile(optimizer=tf.keras.optimizers.Adam(lr))

    def select_action(self,state) :
        state = tf.convert_to_tensor([state],dtype=tf.float32)
        probs = self.policy(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample().numpy()[0]
        return action

    def remember(self,state,action,reward) :
        self.state_mem.append(state)
        self.action_mem.append(action)
        self.reward_mem.append(reward)

    def learn(self, update='REINFORCE',baseline=None) :

        if update == 'REINFORCE' :
            avg_loss = 0
            # Compute reward-to-go :
            gammas = [self.discount**i for i in range(len(self.reward_mem))]
            Gs = [np.dot(gammas[:len(gammas)-i], self.reward_mem[i:]) for i in range(len(self.reward_mem))]
            # Update for every step
            for step, (state,action,g) in enumerate(zip(self.state_mem,self.action_mem,Gs)) :
                state = tf.convert_to_tensor([state],dtype=tf.float32)
                g = tf.convert_to_tensor(g,dtype=tf.float32)

                if baseline == None : 
                    delta = g
                else : 
                    # update the baseline
                    with tf.GradientTape() as tape :
                        baseline_loss = (g-baseline(state))**2
                        baseline_grads = tape.gradient(baseline_loss, baseline.trainable_variables)
                        baseline.optimizer.apply_gradients(zip(baseline_grads,baseline.trainable_variables))   
                    delta = g - baseline(state)

                with tf.GradientTape() as tape :
                    # compute the loss
                    probs = self.policy(state)
                    probs = tf.clip_by_value(probs, clip_value_min=1e-8, clip_value_max=1-1e-8)
                    dist = tfp.distributions.Categorical(probs)
                    log_prob = dist.log_prob(action)
                    loss = - log_prob * delta * (self.discount**step)

                    # apply gradients
                    grads = tape.gradient(loss, self.policy.trainable_variables)
                    self.policy.optimizer.apply_gradients(zip(grads,self.policy.trainable_variables))   

                    avg_loss+=loss.numpy()[0][0]
            avg_loss/=len(Gs)                     

        self.state_mem=[]
        self.action_mem=[]
        self.reward_mem=[]

        return avg_loss

nA = env.action_space.n

agent = agent(0.99,0.0003,nA,256,256)
baseline = value_network(256,256,0.001)


# To plot score & loss in tboard
train_writer = tf.summary.create_file_writer('results/tboard')
avg_score = 0
avg_loss = 0

print(" -   -   -   Testing REINFORCE with baseline on CartPole-v1   -   -   -\n")
print("Run 'tensorboard --logdir resullts/tboard' to visualize the average score\n")

num_episodes = input('Choose the number of episodes to train for (default : 1000)\n')
try :
    num_episodes = int(num_episodes)
except :
    num_episodes = 1000

for ep in range(num_episodes) :
    done = False
    score=0
    state = env.reset()
    while not done :
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        score += reward
        agent.remember(state,action,reward)
        state=new_state

    print("ep",ep,"score",score)
    avg_loss =  0.95*avg_loss +  0.05*agent.learn(update='REINFORCE',baseline=baseline)
    avg_score = 0.95*avg_score + 0.05*score

    with train_writer.as_default() :
        tf.summary.scalar("REINFORCE w/ baseline Average score",avg_score,step = ep)

print('\n Training complete')
done = False
state = env.reset()
while not done :
    action = agent.select_action(state)
    new_state, reward, done, info = env.step(action)
    env.render()
    state=new_state
    
time.sleep(1)
env.close()
