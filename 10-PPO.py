'''
PPO implementation with accessible HPs for tuning
Solves CartPole-v1 in less than 100 episodes with certain setups
High learning rates (0.005) make it get to 100 avg very fast, but then it drops
Learning rate decay available for the actor
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Actor (tf.keras.Model) :
    def __init__(self,l1_dims,l2_dims,lr,optimizer,nA) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(l1_dims,activation='relu')
        self.l2 = tf.keras.layers.Dense(l2_dims,activation='relu')
        self.out = tf.keras.layers.Dense(nA,activation='softmax')
        self.compile(optimizer=optimizer(lr))

    def call(self,input) :
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class Critic (tf.keras.Model) :
    def __init__(self,l1_dims,l2_dims,lr,optimizer) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(l1_dims,activation='relu')
        self.l2 = tf.keras.layers.Dense(l2_dims,activation='relu')
        self.out = tf.keras.layers.Dense(1,activation=None)
        
        self.optimizer = optimizer
        self.compile(optimizer=optimizer(lr))

    def call(self,input) :
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class Agent() :
    def __init__(self,\
        actor_l1_dims, actor_l2_dims, actor_lr, actor_lr_decayrate, actor_optimizer,\
        critic_l1_dims, critic_l2_dims, critic_lr, critic_optimizer,\
        nA, nS, discount, gae_lambda, clip_epsilon,\
        batch_size, n_epochs, horizon) -> None:
        
        self.actor = Actor(actor_l1_dims,actor_l2_dims,actor_lr, actor_optimizer,nA)
        self.critic = Critic(critic_l1_dims, critic_l2_dims, critic_lr, critic_optimizer)
        
        self.actor_lr = actor_lr
        self.actor_lr_decayrate = actor_lr_decayrate
        self.actor_optimizer = actor_optimizer

        self.nS = nS
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.horizon = horizon

        self.state_mem = np.empty((0,nS))
        self.action_mem = np.array([])
        self.reward_mem = np.array([])
        self.done_mem = np.array([])
        self.prob_mem = np.array([])

    def update_lr(self,learning_step) :
        lr = self.actor_lr * (self.actor_lr_decayrate**learning_step)
        if lr != self.actor_lr and learning_step % 20 == 0 :
            print("\t learning rate :", '{:.2e}'.format(lr))
        self.actor.compile(optimizer=self.actor_optimizer(lr))

    def select_action(self,state) :
        state = tf.convert_to_tensor([state])
        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        prob = dist.prob(action)

        return action.numpy()[0], prob.numpy()[0]

    def remember(self,state,action,reward,done,action_prob) :
        self.state_mem = np.vstack((self.state_mem,state))
        self.action_mem = np.append(self.action_mem,action)
        self.reward_mem = np.append(self.reward_mem,reward)
        self.done_mem = np.append(self.done_mem,done)
        self.prob_mem = np.append(self.prob_mem,action_prob)

    def clear_memory(self) :
        self.state_mem = np.empty((0,self.nS))
        self.action_mem = np.array([])
        self.reward_mem = np.array([])
        self.done_mem = np.array([])
        self.prob_mem = np.array([])

    def compute_advantages(self,values) :        
        next_states_values = np.roll(values,-1)
        discounts = np.array([(self.discount*self.gae_lambda)**i for i in range(self.horizon)])
        deltas = self.reward_mem + self.discount * next_states_values * (1 - self.done_mem) - values
        deltas[-1] = 0.0
        advantages = np.array([np.dot(discounts[:self.horizon-t],deltas[t:self.horizon]) for t in range(self.horizon)])
        return advantages

    def generate_batches(self) :
        indices = np.arange(self.horizon)
        np.random.shuffle(indices)
        batch_starts = np.arange(0,self.horizon,self.batch_size)
        batches = [indices[i:i+self.batch_size] for i in batch_starts]
        return batches

    def compute_actor_loss(self,batch,advantages) :
        probs = self.actor(self.state_mem[batch])
        dist = tfp.distributions.Categorical(probs)
        new_probs = dist.prob(self.action_mem[batch])
        old_probs = tf.convert_to_tensor(self.prob_mem[batch], dtype='float32')
        ratios = new_probs / old_probs
        weighted_ratios = ratios * advantages[batch]
        weighted_clipped_ratios = tf.clip_by_value(ratios,1-self.clip_epsilon,1+self.clip_epsilon) * advantages[batch]
        loss = - tf.reduce_mean(tf.math.minimum(weighted_clipped_ratios,weighted_ratios))
        return loss

    def compute_critic_loss(self,batch,values,advantages) :
        returns = values[batch] + advantages[batch]
        new_values = tf.reshape(self.critic(self.state_mem[batch]),(-1))
        loss = tf.reduce_mean((returns - new_values)**2)
        return loss

    def learn(self) :
        values = self.critic(self.state_mem).numpy().reshape(-1)
        advantages = self.compute_advantages(values)
        for _ in range(self.n_epochs) :
            batches = self.generate_batches()
            for batch in batches :
                with tf.GradientTape(persistent=True) as tape :
                    actor_loss = self.compute_actor_loss(batch,advantages)
                    critic_loss = self.compute_critic_loss(batch,values,advantages)
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(actor_grads,self.actor.trainable_variables))
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic.optimizer.apply_gradients(zip(critic_grads,self.critic.trainable_variables))

        self.clear_memory()


# Environment setup
env = gym.make('CartPole-v1')
nA = env.action_space.n
nS = env.observation_space.shape[0]
writer =tf.summary.create_file_writer('results/tboard')

# Hyperparameters
    # Could be added : 
    #   - loss function (huber vs MSE for the critic)
    #   - activation functions (relu vs leaky relu vs tanh)
    #   - number of layers
    #   - regularization, batch norm, dropout
    #   - network init

actor_l1_dims = 256
actor_l2_dims = 256
actor_lr = 0.0004
actor_lr_decayrate = 1-1e-3         # probably useless since we're using Adam
actor_optimizer = tf.keras.optimizers.Adam

critic_l1_dims = 256
critic_l2_dims = 256
critic_lr = 0.0008
critic_optimizer = tf.keras.optimizers.Adam

discount = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2

batch_size = 8
n_epochs = 4
horizon = 64

agent = Agent(actor_l1_dims,actor_l2_dims,actor_lr, actor_lr_decayrate, actor_optimizer,\
    critic_l1_dims, critic_l2_dims, critic_lr, critic_optimizer,\
    nA, nS, discount, gae_lambda, clip_epsilon,\
    batch_size, n_epochs, horizon)

avg_score = 0
timesteps = 0
learning_step = 0

print(" -   -   -   Testing PPO on CartPole-v1   -   -   -\n")
print("Run 'tensorboard --logdir resullts/tboard' to visualize the average score\n")

n_episodes = input('Choose the number of episodes to train for (default : 1000)\n')
try :
    n_episodes = int(n_episodes)
except :
    n_episodes = 1000

for ep in range(n_episodes) :
    done = False
    state = env.reset()
    score = 0
    while not done :
        action, prob = agent.select_action(state)
        new_state, reward, done, _ = env.step(action)
        agent.remember(state,action,reward,done,prob)
        state = new_state
        score += reward

        timesteps += 1
        if timesteps == horizon :
            agent.learn()
            timesteps = 0
            
            # LR decay
            learning_step+=1
            agent.update_lr(learning_step)

    avg_score = 0.05 * score + 0.95 * avg_score
    print("Episode",ep,"score",score)
    with writer.as_default() :
        tf.summary.scalar('PPO Average Score', avg_score, step=ep)


print('\n Training complete')
done = False
state = env.reset()
while not done :
    action, prob = agent.select_action(state)
    new_state, reward, done, info = env.step(action)
    env.render()
    state=new_state
    
time.sleep(1)
env.close()
