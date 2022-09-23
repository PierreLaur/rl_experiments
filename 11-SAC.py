'''
Implementing Soft Actor-Critic (the latest version, from Haarnoja et al. 2019)
no automatic temperature learning for now (fixed to 1) & a single Q function instead of 2
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import gym
from collections import deque
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Qnet (tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(256, activation='relu')
        self.l2 = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(1)
        self.compile(optimizer=tf.keras.optimizers.Adam())

    def call(self, states, actions):
        input = tf.concat([states, actions], axis=1)
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x


class PolicyNet (tf.keras.Model):
    def __init__(self, env) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(256, activation='relu')
        self.l2 = tf.keras.layers.Dense(256, activation='relu')
        self.mu = tf.keras.layers.Dense(env.action_space.shape[0])
        self.log_sigma = tf.keras.layers.Dense(env.action_space.shape[0])
        self.compile(optimizer=tf.keras.optimizers.Adam())
        self.action_scale = env.action_space.high

    def call(self, input, greedy=False):
        x = self.l1(input)
        x = self.l2(x)
        mu = self.mu(x)

        log_sigma = self.log_sigma(x)
        sigma = tf.exp(log_sigma)

        dist = tfp.distributions.Normal(mu, sigma)
        if greedy :
            actions = mu
        else :
            actions = dist.sample()
        tanh_actions = tf.tanh(actions)
        log_mu = dist.log_prob(actions)
        log_probs = log_mu - tf.reduce_sum(tf.math.log(
            1 - tanh_actions**2 + 1e-6), axis=1, keepdims=True)  # try without 1e-6
        return tanh_actions*self.action_scale, log_probs


class SACAgent:
    def __init__(self, env, discount=0.99, tau=0.005, alpha=1, batch_size=1_000_000, capacity=1_000_000, reward_scale=1) -> None:
        self.n_actions = env.action_space.shape[0]
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        # self.alpha = tf.Variable(0.0,dtype='float32')
        # self.alpha_optimizer=tf.keras.optimizers.Adam(lr=0.0003)
        # self.target_entropy = -tf.constant(self.n_actions,dtype='float32')
        self.reward_scale = reward_scale

        self.policy = PolicyNet(env)
        self.Q = Qnet()
        self.Q_target = Qnet()
        self.Q_target.set_weights(self.Q.get_weights())

        self.batch_size = batch_size
        replay_buffer_capacity = capacity
        self.states = deque([], maxlen=replay_buffer_capacity)
        self.actions = deque([], maxlen=replay_buffer_capacity)
        self.rewards = deque([], maxlen=replay_buffer_capacity)
        self.new_states = deque([], maxlen=replay_buffer_capacity)
        self.dones = deque([], maxlen=replay_buffer_capacity)

    def update_target(self):
        new_weights = []
        for target, main in zip(self.Q_target.get_weights(), self.Q.get_weights()):
            new_weights.append(
                self.tau * main + (1-self.tau) * target
            )
        self.Q_target.set_weights(new_weights)

    def select_action(self, state):
        state = tf.convert_to_tensor([state])
        action, _ = self.policy(state)
        return action.numpy()[0]

    def store_experience(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(done)

    def generate_batch(self):
        batch_indices = np.random.choice(
            np.arange(len(self.states)), size=min(self.batch_size, len(self.states))
            )
        states = np.array(self.states)[batch_indices]
        actions = np.array(self.actions)[batch_indices]
        rewards = np.array(self.rewards)[batch_indices]
        new_states = np.array(self.new_states)[batch_indices]
        dones = np.array(self.dones)[batch_indices]
        return states, actions, rewards, new_states, dones

    def save_model(self) :
        self.policy.save_weights('results/SAC_policy_pendulum')
        self.Q.save_weights('results/SAC_Q_pendulum')
        self.Q_target.save_weights('results/SAC_Qtarget_pendulum')

    def load_model(self) :
        self.policy.load_weights('results/SAC_policy_pendulum')
        self.Q.load_weights('results/SAC_Q_pendulum')
        self.Q_target.load_weights('results/SAC_Qtarget_pendulum')

    def learn(self):

        states, actions, rewards, new_states, dones = self.generate_batch()

        # Update Q function
        with tf.GradientTape() as tape:
            q = tf.squeeze(self.Q(states, actions), 1)
            new_actions, log_probs = self.policy(new_states,greedy=True)
            v = self.Q_target(new_states, new_actions) - self.alpha * log_probs
            better_q = self.reward_scale* rewards + (1-dones) * self.discount * tf.squeeze(v, 1)
            q_loss = 0.5*tf.keras.losses.MSE(q, better_q)
        q_grads = tape.gradient(q_loss, self.Q.trainable_variables)
        self.Q.optimizer.apply_gradients(
            zip(q_grads, self.Q.trainable_variables))

        # Update policy
        with tf.GradientTape() as tape:
            new_actions, log_probs = self.policy(states)
            q = self.Q(states, new_actions)
            pi_loss = tf.reduce_mean(self.alpha * log_probs - q)
        pi_grads = tape.gradient(pi_loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(
            zip(pi_grads, self.policy.trainable_variables))

        # update alpha
        # with tf.GradientTape() as tape :
        #     new_actions, log_probs = self.policy(states,greedy=True)
        #     alpha_loss = tf.reduce_mean(- self.alpha * (log_probs + self.target_entropy))
        # variables = [self.alpha]
        # grads = tape.gradient(alpha_loss,variables)
        # self.alpha_optimizer.apply_gradients(zip(grads,variables))

        self.update_target()


print(' -   -   -   Testing Soft Actor-Critic    -   -   -\n')

render = input("Render ? (y/n)")
render = True if render=="y" else False
env = gym.make('InvertedPendulum')
n_episodes = 500
average_score = 0
agent = SACAgent(env, reward_scale=1)

for episode in range(n_episodes):
    done = False
    state = env.reset()
    score = 0

    while not done:
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        if render : env.render()

        agent.store_experience(state, action, reward, new_state, done)
        score += reward
        state = new_state

        agent.learn()

    average_score = 0.95*average_score + 0.05*score
    print(
        f'Episode {episode} score {round(score)}      average score {round(average_score/(1-0.95**(episode+1)))}')
