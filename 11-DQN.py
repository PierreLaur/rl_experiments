'''
Implementing DQN to solve CartPole-v1
- Experience Replay
- Batch updates
'''

import numpy as np
import tensorflow as tf
import gym
import os
from collections import deque
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ValueNet(tf.keras.Model):
    def __init__(self, n_actions) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(24, activation='relu')
        self.l2 = tf.keras.layers.Dense(24, activation='relu')
        self.out = tf.keras.layers.Dense(n_actions, activation='linear')
        self.compile(loss='mse', optimizer='Adam')

    def call(self, input):
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x


class Agent:
    def __init__(self, n_actions) -> None:

        self.n_actions = n_actions
        self.epsilon = 0.3
        self.discount = 0.99
        self.batch_size = 32

        self.Q_function = ValueNet(n_actions)

        self.states = deque([],maxlen=2000)
        self.actions = deque([],maxlen=2000)
        self.rewards = deque([],maxlen=2000)
        self.new_states = deque([],maxlen=2000)
        self.dones = deque([],maxlen=2000)


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


    def learn(self):
        if len(self.states) < self.batch_size : return

        states,actions,rewards,new_states,dones = self.generate_batch()
        
        states = tf.convert_to_tensor(states)
        new_states = tf.convert_to_tensor(new_states)

        q_values = self.Q_function(states).numpy()
        new_q_values = self.Q_function(new_states).numpy()

        updates = np.array(rewards) + np.max(new_q_values, axis=1) * (1-dones)
        q_values[np.arange(self.batch_size),actions] = updates
        targets = tf.convert_to_tensor(q_values)

        self.Q_function.fit(states,targets,verbose=0)




print(' -   -   -   Testing DQN on CartPole-v1   -   -   -\n')

n_episodes = input('Choose the number of episodes to train for (default : 1000)\n')
try :
    n_episodes = int(n_episodes)
except :
    n_episodes = 1000

env = gym.make('CartPole-v1')
n_actions = env.action_space.n
agent = Agent(n_actions)

writer = tf.summary.create_file_writer('results/tboard')

avg_score = 10
start_time = time.process_time()
second = 0
for ep in range(n_episodes):
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
    current = round(time.process_time() - start_time)
    if  current > second :
        with writer.as_default() :
            tf.summary.scalar('DQN/step_update',avg_score,step=second)
        second = current
    print(f"Ep {ep} score {score}       avg {round(avg_score)}")

print("\nTraining complete - testing the policy")

score = 0
for _ in range(10) :
    done = False
    state = env.reset()
    while not done :
        action = agent.select_action(state,greedy=True)
        new_state,reward,done,info = env.step(action)
        env.render()
        state = new_state
        score += reward

print(f"    Final average score {score/10}")