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
        self.mu = tf.keras.layers.Dense(1)
        self.log_sigma = tf.keras.layers.Dense(1)
        self.compile(optimizer=tf.keras.optimizers.Adam())
        self.action_scale = 1
        self.tau = 0.1

    def call(self, input, greedy=False,discrete=True):
        x = self.l1(input)
        x = self.l2(x)
        mu = self.mu(x)

        log_sigma = self.log_sigma(x)
        sigma = tf.exp(log_sigma)

        dist = tfp.distributions.Normal(mu, sigma)
        if discrete :
            possible_actions = [0,1]
            logits = tf.exp(-0.5*(possible_actions-tf.tanh(mu))**2/tf.tanh(sigma)**2)/(tf.sqrt(2*np.pi)*tf.tanh(sigma))
            gumbel_dist = tfp.distributions.Gumbel(0,1)
            noise = gumbel_dist.sample(sample_shape=tf.shape(logits))
            noisy_logits = tf.exp((tf.math.log(logits) + noise)/self.tau)
            normalized_noisy_logits = noisy_logits/tf.reduce_sum(noisy_logits)
            tanh_actions = tf.reshape(tf.argmax(normalized_noisy_logits,axis=1),(-1,1))            

            dist = tfp.distributions.Categorical(probs=logits)
            log_probs = tf.reshape(dist.log_prob(tf.squeeze(tanh_actions)),(-1,1))

        else :
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
        self.n_actions = env.action_space.n
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
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
        return action.numpy()[0][0]

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

    def learn(self):

        if len(self.states) < 4 : return

        states, actions, rewards, new_states, dones = self.generate_batch()

        actions = tf.convert_to_tensor(actions,dtype=tf.float32)
        # Update Q function
        with tf.GradientTape() as tape:
            q = tf.squeeze(self.Q(states, actions), 1)
            new_actions, log_probs = self.policy(new_states,greedy=False)
            new_actions = tf.convert_to_tensor(actions,dtype=tf.float32)
            v = self.Q_target(new_states, new_actions) - self.alpha * log_probs
            better_q = self.reward_scale* rewards + (1-dones) * self.discount * tf.squeeze(v, 1)
            q_loss = 0.5*tf.keras.losses.MSE(q, better_q)
        q_grads = tape.gradient(q_loss, self.Q.trainable_variables)
        self.Q.optimizer.apply_gradients(
            zip(q_grads, self.Q.trainable_variables))

        # Update policy
        with tf.GradientTape() as tape:
            new_actions, log_probs = self.policy(states)
            new_actions = tf.convert_to_tensor(actions,dtype=tf.float32)

            q = self.Q(states, new_actions)
            pi_loss = tf.reduce_mean(self.alpha * log_probs - q)
        pi_grads = tape.gradient(pi_loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(
            zip(pi_grads, self.policy.trainable_variables))

        self.update_target()


print(' -   -   -   Testing Soft Actor-Critic    -   -   -\n')

# envname = int(input('Choose an environment :\n\tAnt\n\tInvertedPendulum')
render = input("Render ? (y/n)")
render = True if render=="y" else False
env = gym.make('CartPole')
n_episodes = 500
average_score = 0
agent = SACAgent(env, reward_scale=5)
for episode in range(n_episodes):
    done = False
    state = env.reset()
    score = 0
    st = time.process_time()

    while not done:
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        if render : env.render()

        agent.store_experience(state, [action], reward, new_state, done)
        score += reward
        state = new_state

        agent.learn()

    average_score = 0.95*average_score + 0.05*score
    print(
        f'Episode {episode} score {round(score)}      average score {round(average_score/(1-0.95**(episode+1)))}')
