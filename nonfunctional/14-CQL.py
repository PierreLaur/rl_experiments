'''
Implementing & testing the Conservative Q-Learning algorithm for offline reinforcement learning
Not functional as of now (01/09)
'''

import numpy as np
import tensorflow as tf
import gym
import os
from collections import deque
import time
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ValueNet (tf.keras.Model) :
   def __init__(self, n_actions) -> None:
      super().__init__()
      self.l1 = tf.keras.layers.Dense(24,activation='relu')
      self.l2 = tf.keras.layers.Dense(24,activation='relu')
      self.out = tf.keras.layers.Dense(n_actions,activation='linear')
      self.compile(loss='mse',optimizer=tf.keras.optimizers.Adam())

   def call(self,input) :
      x = self.l1(input)
      x = self.l2(x)
      x = self.out(x)
      return x

class Agent:
    def __init__(self, env) -> None:
        self.n_actions = env.action_space.n
        self.epsilon = 0.3
        self.discount = 0.99
        self.batch_size = 256

        self.Q_function = ValueNet(self.n_actions)
        
        self.replay_buffer_capacity = 100000
        self.states = deque([],maxlen=self.replay_buffer_capacity)
        self.actions = deque([],maxlen=self.replay_buffer_capacity)
        self.rewards = deque([],maxlen=self.replay_buffer_capacity)
        self.new_states = deque([],maxlen=self.replay_buffer_capacity)
        self.dones = deque([],maxlen=self.replay_buffer_capacity)

    def select_action(self, state, greedy=False):
        if not greedy and np.random.uniform(0,1) < self.epsilon :
            action = np.random.choice(self.n_actions)
            return action
        else :
            input = tf.convert_to_tensor([state])
            q_values = self.Q_function(input)
            action = tf.argmax(q_values[0]).numpy()
            return action

    def store_experience(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(done)

    def generate_batch(self) :
        batch_indices = np.random.choice(np.arange(len(self.states)),size=self.batch_size)
        states = np.array(self.states)[batch_indices]
        actions = np.array(self.actions)[batch_indices]
        rewards = np.array(self.rewards)[batch_indices]
        new_states = np.array(self.new_states)[batch_indices]
        dones = np.array(self.dones)[batch_indices]
        return states,actions,rewards,new_states,dones

    def learn(self, no_cql=False):
        if len(self.states) < self.batch_size : return

        states,actions,rewards,new_states,dones = self.generate_batch()
        
        states = tf.convert_to_tensor(states)
        new_states = tf.convert_to_tensor(new_states)

        with tf.GradientTape() as tape :
            q_values = self.Q_function(states)
            new_q_values = self.Q_function(new_states)

            q_updates = rewards + (1-dones) * tf.math.reduce_max(new_q_values, axis = 1) * self.discount

            # compute targets  
            targets = np.array(q_values)
            targets[np.arange(self.batch_size), actions] = q_updates
            targets = tf.convert_to_tensor(targets)

            bellman_mse = tf.keras.losses.MSE(q_values, targets)
            loss = bellman_mse
            if no_cql == False :
                cql_loss = tf.reduce_mean(tf.math.reduce_logsumexp(q_values, axis = 1)) - tf.reduce_mean(q_values)
                loss = cql_loss + 0.5 * bellman_mse
        grads = tape.gradient(loss, self.Q_function.trainable_variables)
        self.Q_function.optimizer.apply_gradients(zip(grads,self.Q_function.trainable_variables))
        
def online_loop (num=1) :
    # Online loop (to gather data)
    env = gym.make('CartPole-v1')
    n_episodes = 1000
    average_score = 10
    agent = Agent(env)
    finished = False
    for episode in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        while not done :
            action = agent.select_action(state)
            new_state,reward,done,info = env.step(action)
            agent.store_experience(state,action,reward,new_state,done)
            if len(agent.states) == 25000 :
                np.savetxt('data/DQN_CartPole_states_0'+str(num),agent.states)
                np.savetxt('data/DQN_CartPole_actions_0'+str(num),agent.actions)
                np.savetxt('data/DQN_CartPole_rewards_0'+str(num),agent.rewards)
                np.savetxt('data/DQN_CartPole_new_states_0'+str(num),agent.new_states)
                np.savetxt('data/DQN_CartPole_dones_0'+str(num),agent.dones)
                finished = True
            score += reward
            state = new_state

            agent.learn(no_cql=True)
        if finished :
            break

        if episode %5 == 0 :
            print(f'        Buffer state : {(num-1) * 25 + 100*len(agent.states)/agent.replay_buffer_capacity} %')

def test_agent(agent,env,n_episodes) :
    average_score = 0
    agent = Agent(env)
    for _ in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        while not done :
            action = agent.select_action(state, greedy=True)
            new_state,reward,done,info = env.step(action)
            score += reward
            state = new_state    
        average_score += score
    print(f'Agent average score over {n_episodes} episodes : {round(average_score/n_episodes)}')
    
env = gym.make('CartPole-v1')
agent = Agent(env)

int('   -   -   - Testing CQL on CartPole-v1 with a dataset from 4 DQN trainings    -   -   -\n')

# gathering the data
try :
    agent.states = deque(np.vstack([np.loadtxt('data/DQN_CartPole_states_0'+str(num)) for num in [1,2,3,4]]))
    agent.actions = deque(np.concatenate([np.loadtxt('data/DQN_CartPole_actions_0'+str(num)).astype(np.int64) for num in [1,2,3,4]]))
    agent.rewards = deque(np.concatenate([np.loadtxt('data/DQN_CartPole_rewards_0'+str(num)) for num in [1,2,3,4]]))
    agent.new_states = deque(np.vstack([np.loadtxt('data/DQN_CartPole_new_states_0'+str(num)) for num in [1,2,3,4]]))
    agent.dones = deque(np.concatenate([np.loadtxt('data/DQN_CartPole_dones_0'+str(num)).astype(np.int64) for num in [1,2,3,4]]))
except FileNotFoundError :
    print('Data not found, generating. make sure a ./data folder exists')
    for num in [1,2,3,4] :
        online_loop(num)
    print('Data generated, please relaunch')
    exit()

# The offline loop
training_steps = 10000
for step in tqdm(range(training_steps)) :
    agent.learn(no_cql=False)
    if step % 10 == 0 :
        test_agent(agent,env,5)