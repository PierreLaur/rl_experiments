import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import gym
import time
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DiscreteActionPolicyNet (tf.keras.Model) :
    def __init__(self,n_actions) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(256,activation='relu')
        self.l2 = tf.keras.layers.Dense(256,activation='relu')
        self.out = tf.keras.layers.Dense(n_actions,activation='softmax')
        self.compile(optimizer=tf.keras.optimizers.Adam(0.0001))

    def call(self,input) :
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class DiscreteActionValueNet(tf.keras.Model) :
    def __init__(self,n_actions) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(256,activation='relu')
        self.l2 = tf.keras.layers.Dense(256,activation='relu')
        self.out = tf.keras.layers.Dense(n_actions,activation=None)
        self.compile(optimizer=tf.keras.optimizers.Adam(0.0002))

    def call(self,input) :
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class SACAgent:
    def __init__(self, env, discount=0.99, tau=0.005, alpha=1, batch_size=1_000_000, capacity=1_000_000) -> None:
        self.n_actions = env.action_space.n
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        
        self.policy = DiscreteActionPolicyNet(self.n_actions)
        self.Q = DiscreteActionValueNet(self.n_actions)
        self.Q_target = DiscreteActionValueNet(self.n_actions)
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
        
    def select_action(self, state, greedy=False):
        state = tf.convert_to_tensor([state])
        probs = self.policy(state)
        dist = tfp.distributions.Categorical(probs)
        if greedy :
            action = dist.mode()
        else :
            action = dist.sample()
        return action.numpy()[0]
        
    def sample_actions(self,states,mode=False) :
        action_probs = tfp.distributions.Categorical(self.policy(states))
        if mode :
            actions = action_probs.mode()
        else :
            actions = action_probs.sample()
        log_probs = action_probs.log_prob(actions)
        return actions,log_probs

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

    def expected_Q(self,states) :
        q_values = self.Q_target(states)
        action_probs = self.policy(states)
        expected_Q = q_values * action_probs
        return expected_Q

    def V(self,states,use_main_network=False) :
        if use_main_network :
            q_values = self.Q(states)
        else :
            q_values = self.Q_target(states)
        action_probs = self.policy(states)
        expected_Q = tf.reduce_sum(q_values * action_probs,axis=1,keepdims=True)
        expected_log_probs = tf.reduce_sum(tf.math.log(action_probs) * action_probs,axis=1,keepdims=True)
        V = expected_Q - self.alpha * expected_log_probs 
        return V

    def learn(self):

        states, actions, rewards, new_states, dones = self.generate_batch()
        
        with tf.GradientTape() as tape :
            V_ = self.V(new_states)
            Q = self.Q(states)
            better_Q = np.array(Q)
            td_error = rewards + (1-dones) * self.discount * tf.squeeze(V_,1)
            better_Q[np.arange(len(states)),actions] = td_error
            q_loss = 0.5 * tf.reduce_mean(tf.reduce_sum((Q-better_Q),axis=1)**2)
            print(f'                                                                            {q_loss.numpy()=}')
        q_grads = tape.gradient(q_loss, self.Q.trainable_variables)
        self.Q.optimizer.apply_gradients(
            zip(q_grads, self.Q.trainable_variables))

        with tf.GradientTape() as tape :
            V = self.V(states,use_main_network=True)
            pi_loss = tf.reduce_mean(-V)
            print(f'{pi_loss.numpy()=}')

        pi_grads = tape.gradient(pi_loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(
            zip(pi_grads, self.policy.trainable_variables))
        
        self.update_target()

env = gym.make('CartPole')
n_episodes = 1000
average_score = 10
agent = SACAgent(env)
for episode in range(n_episodes) :
    done = False
    state = env.reset()
    score = 0
    while not done :
        action = agent.select_action(state)
        new_state,reward,done,info = env.step(action)
        if len(agent.states) <= 256 :
            agent.store_experience(state,action,reward,new_state,done)
        else :
            print('buffer ok')
        score += reward
        state = new_state

        agent.learn()

    average_score = 0.95*average_score + 0.05*score
    print(f'Episode {episode} score {score}      average score {round(average_score)}')