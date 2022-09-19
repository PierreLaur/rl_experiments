'''
Testing different DQN improvements
'''
from collections import deque
import numpy as np
import tensorflow as tf
import time
import os
import gym

import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ValueNet(tf.keras.Model):
    def __init__(self, n_actions) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(24, activation='relu')
        self.l2 = tf.keras.layers.Dense(24, activation='relu')
        self.out = tf.keras.layers.Dense(n_actions, activation='linear')
        self.compile(loss='MSE', optimizer="Adam")

    def call(self, input):
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class Agent :
    def __init__(self, n_actions, double=False) -> None:
        self.n_actions = n_actions
        self.Q_function = ValueNet(n_actions)
        self.epsilon = 0.3
        self.batch_size = 32
        self.discount = 0.99
        self.min_capacity = 2000
        self.max_capacity = 2000

        self.states = deque([],maxlen=self.max_capacity)
        self.actions = deque([],maxlen=self.max_capacity)
        self.rewards = deque([],maxlen=self.max_capacity)
        self.new_states = deque([],maxlen=self.max_capacity)
        self.dones = deque([],maxlen=self.max_capacity)

        if double : 
            self.double = True
            self.Q_aux = ValueNet(n_actions)
        else : self.double = False

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

    def average_weights(self) :
        Q_weights = self.Q_function.get_weights()
        aux_weights = self.Q_aux.get_weights()

        new_aux_weights = []

        for q,aux in zip(Q_weights,aux_weights) :
            new_aux_weights.append(0.01*q + 0.99*aux)

        self.Q_aux.set_weights(new_aux_weights)

    def learn(self) :
        if len(self.states) < self.batch_size : return

        states,actions,rewards,new_states,dones = self.generate_batch()

        states = tf.convert_to_tensor(states)
        new_states = tf.convert_to_tensor(new_states)

        q_values = self.Q_function(states).numpy()
        new_q_values = self.Q_function(new_states).numpy()

        if self.double :
            aux_q_values = self.Q_aux(new_states).numpy()
            updates = rewards + self.discount*new_q_values[np.arange(self.batch_size),np.argmax(aux_q_values,axis=1)] * (1-dones)
        else :
            updates = rewards + self.discount*np.max(new_q_values,axis=1) * (1-dones)

        q_values[np.arange(self.batch_size),actions] = updates
        targets = tf.convert_to_tensor(q_values)

        self.Q_function.fit(states,targets,verbose=0)

        if self.double :
            self.average_weights()


def train_agent(env, agent, n_episodes, graph_name='') :
    
    print('Training...')
    writer = tf.summary.create_file_writer('results/tboard')
    avg_score = 10
    start_time = time.process_time()
    last_plot_time = 0
    for ep in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        while not done :
            action = agent.select_action(state)
            new_state, reward, done, info = env.step(action)
            agent.store_experience(state,action,reward,new_state,done)
            state = new_state
            score += reward

            agent.learn()

        avg_score = 0.95*avg_score + 0.05*score
        # write every second
        exec_time = round(time.process_time() - start_time)
        if exec_time > last_plot_time :
            last_plot_time = exec_time
            with writer.as_default() :
                tf.summary.scalar('Rainbow/Average_Score_'+graph_name, avg_score, step=exec_time)

def test_agent(env, agent, n_episodes) :
    for ep in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        while not done :
            action = agent.select_action(state,greedy=True)
            new_state, reward, done, info = env.step(action)
            state = new_state
            score += reward

print(' -   -   -   Testing DQN -   -   -\n')

env = gym.make('CartPole-v1')
n_episodes = 1000
n_actions = env.action_space.n
agent = Agent(n_actions)
t1 = threading.Thread(target=train_agent, args =(env,agent,n_episodes,'single'))

env2 = gym.make('CartPole-v1')
agent2 = Agent(n_actions,double=True)
t2 = threading.Thread(target=train_agent, args =(env2,agent2,n_episodes,'double'))

t1.start()
t2.start()
t1.join()
t2.join()