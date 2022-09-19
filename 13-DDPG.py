import numpy as np
import tensorflow as tf
from collections import deque
import os
import gym

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Critic(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(400, activation='relu')
        self.l2 = tf.keras.layers.Dense(300, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation=None)
        self.compile(optimizer=tf.keras.optimizers.Adam(0.002))

    def call(self, state, action):
        input = tf.concat([state, action], axis=1)
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x


class Actor(tf.keras.Model):
    def __init__(self, n_actions) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(400, activation='relu')
        self.l2 = tf.keras.layers.Dense(300, activation='relu')
        self.out = tf.keras.layers.Dense(n_actions, activation='tanh')
        self.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.out(x)
        return x


class Agent:
    def __init__(self, env) -> None:
        self.batch_size = 64
        self.tau = 0.005
        self.discount = 0.99
        self.n_actions = env.action_space.shape[0]
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]
        self.capacity = 1000000

        self.actor = Actor(self.n_actions)
        self.target_actor = Actor(self.n_actions)
        self.target_actor.set_weights(self.actor.get_weights())

        self.critic = Critic()
        self.target_critic = Critic()
        self.target_critic.set_weights(self.critic.get_weights())

        self.states = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.new_states = deque(maxlen=self.capacity)
        self.dones = deque(maxlen=self.capacity)

    def store_experience(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(done)

    def update_networks(self):
        for main, target in zip([self.actor, self.critic], [
                                self.target_actor, self.target_critic]):
            new_weights = []
            for m, t in zip(main.get_weights(), target.get_weights()):
                new_weights.append(
                    self.tau * m + (1 - self.tau) * t
                )
            target.set_weights(new_weights)

    def select_action(self, state):
        state = tf.convert_to_tensor([state])
        action = self.actor(state)[0]
        noise = tf.random.normal(shape=action.shape, stddev=0.1)
        action += noise
        action = tf.clip_by_value(action, self.min_action, self.max_action)
        return action.numpy()

    def generate_batch(self):
        batch_indices = np.random.choice(
            np.arange(len(self.states)), size=self.batch_size)
        states = np.array(self.states)[batch_indices]
        actions = np.array(self.actions)[batch_indices]
        rewards = np.array(self.rewards)[batch_indices]
        new_states = np.array(self.new_states)[batch_indices]
        dones = np.array(self.dones)[batch_indices]
        return states, actions, rewards, new_states, dones

    def learn(self):
        if len(self.states) < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.generate_batch()

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards, dtype='float32')
        new_states = tf.convert_to_tensor(new_states)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.target_critic(
                new_states, self.target_actor(new_states)
            ), 1)
            y = rewards + self.discount * value * (1 - dones)
            loss = tf.keras.losses.MSE(
                y, tf.squeeze(
                    self.critic(
                        states, actions), 1))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(-self.critic(states, self.actor(states)))
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables))

        self.update_networks()


env = gym.make('Pendulum-v1')
agent = Agent(env)
n_episodes = 250

avg_score = -1000
for ep in range(n_episodes):
    done = False
    state = env.reset()
    score = 0
    while not done:
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        agent.store_experience(state, action, reward, new_state, done)
        state = new_state
        score += reward

        agent.learn()

    avg_score = 0.95 * avg_score + 0.05 * score
    print(f"Ep {ep} score {score}       avg {round(avg_score)}")
