import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
import gym
import numpy as np
from threading import Thread
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

gamma = 0.9
update_interval = 5
actor_lr = 0.001
critic_lr = 0.002
episodes = 2000

total_episode = 0
episode_max = 200

reward_list = []


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)
        self.entropy_beta = 0.01

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(32, activation='relu')(state_input)
        dense_2 = Dense(32, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim)

    # log(p(a)) of A~N(mu, std)
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * advantages
        return tf.reduce_sum(-loss_policy)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


# main thread
class Agent:
    def __init__(self, env_name):
        env = gym.make(env_name)
        self.env_name = env_name
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 0.5]
        self.global_actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.std_bound)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = cpu_count()

    def train(self):
        workers = []
        for i in range(self.num_workers):
            env = gym.make(self.env_name)
            workers.append(WorkerAgent(env, self.global_actor, self.global_critic, episodes))
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()


# child thread
class WorkerAgent(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes):
        Thread.__init__(self)
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 0.5]

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    # calculate td_target in reverse order
    def td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        if not done:
            cumulative = next_v_value
        else:
            cumulative = 0
        for k in reversed(range(0, len(rewards))):
            cumulative = gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        global total_episode, episode_max

        while self.max_episodes >= total_episode:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward = 0

            state = self.env.reset()

            for i in range(episode_max):
                action = self.actor.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) == update_interval:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)

                    next_v_value = self.critic.model(next_state)
                    td_targets = self.td_target(rewards/8, next_v_value, done)
                    advantages = td_targets - self.critic.model(states)
                    # update global actor and critic
                    self.global_actor.train(states, actions, advantages)
                    self.global_critic.train(states, td_targets)
                    self.actor.model.set_weights(self.global_actor.model.get_weights())
                    self.critic.model.set_weights(self.global_critic.model.get_weights())
                    # clear batch
                    state_batch = []
                    action_batch = []
                    reward_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]
            reward_list.append(episode_reward)
            print('episode: {} rewards: {}'.format(total_episode, episode_reward))
            total_episode += 1

    def run(self):
        self.train()


if __name__ == "__main__":
    env_name = 'Pendulum-v1'
    agent = Agent(env_name)
    agent.train()
    plt.plot(reward_list, color='green')
    plt.show()