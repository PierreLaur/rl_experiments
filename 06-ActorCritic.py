'''
Trying out Advantage Actor Critic methods
It does seem to start learning but doesn't converge (see results/ActorCritic* - 10k episodes run)
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

class Actor(tf.keras.Model) :
    def __init__(self,lr,nA) :
        super().__init__()
        self.l1 = tf.keras.layers.Dense(256, activation='relu')
        self.l2 = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(nA, activation='softmax')
        self.compile(optimizer=tf.keras.optimizers.Adam(lr))

        
    def call (self, input) :
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class Critic(tf.keras.Model) :
    def __init__(self,lr) :
        super().__init__()
        self.l1 = tf.keras.layers.Dense(256, activation='relu')
        self.l2 = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation=None)
        self.compile(optimizer=tf.keras.optimizers.Adam(lr*2))
        
    def call (self, input) :
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class Agent() :
    def __init__(self, nA, discount, lr) :
        self.actor = Actor(lr,nA)
        self.critic = Critic(lr)
        self.discount = discount

    def learn (self,state,action,reward,new_state, done) :
        state = tf.convert_to_tensor([state])
        new_state = tf.convert_to_tensor([new_state])

        # update the critic
        with tf.GradientTape() as tape :
            delta = reward + discount*self.critic(new_state)*(1-int(done)) - self.critic(state)
            loss = tf.squeeze(delta**2)                                 # try without squeezing
            grads = tape.gradient(loss,self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(grads,self.critic.trainable_variables))

        # update the actor
        with tf.GradientTape() as tape :
            delta = reward + discount*self.critic(new_state)*(1-int(done)) - self.critic(state)
            probs = self.actor(state)
            clipped_probs = tf.clip_by_value(probs,clip_value_min=1e-8, clip_value_max=1-1e-8)
            dist = tfp.distributions.Categorical(clipped_probs)
            loss = - tf.squeeze(dist.log_prob(action)) * delta          # try without squeezing
            grads = tape.gradient(loss,self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(grads,self.actor.trainable_variables))

    def select_action (self,state) :
        state = tf.convert_to_tensor([state])
        probs = self.actor(state)
        clipped_probs = tf.clip_by_value(probs,clip_value_min=1e-8, clip_value_max=1-1e-8)
        dist = tfp.distributions.Categorical(clipped_probs)
        action = dist.sample()
        return action.numpy()[0]

env = gym.make('CartPole-v1')
discount = 0.99
lr = 0.0003
nA = env.action_space.n

agent = Agent(nA,discount,lr)

avg_score = 10
train_writer = tf.summary.create_file_writer('results/tboard')

print(" -   -   -   Testing A2C on CartPole-v1   -   -   -\n")
print("Run 'tensorboard --logdir resullts/tboard' to visualize the average score\n")

num_episodes = input('Choose the number of episodes to train for (default : 1000)\n')
try :
    num_episodes = int(num_episodes)
except :
    num_episodes = 1000

for ep in range(num_episodes) :
    state = env.reset()
    done = False
    score = 0
    while not done :
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        agent.learn(state,action,reward,new_state,done)
        state = new_state
        score += reward
    
    avg_score = 0.95 * avg_score + 0.05 * score
    print("ep",ep,"score",score)
    with train_writer.as_default() :
        tf.summary.scalar('A2C Average Score',avg_score,step=ep)

print('\n Training complete')
done = False
state = env.reset()
while not done :
    action = agent.select_action(state)
    new_state, reward, done, info = env.step(action)
    env.render()
    state=new_state
    
env.close()
