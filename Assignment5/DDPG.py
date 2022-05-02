from tensorflow.keras import optimizers, layers, losses
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym

devices = tf.config.experimental.list_physical_devices('GPU')
if devices:
    gpu0 = devices[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpu0], "GPU")

class Replay_buffer():
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, a, r, ns = [], [], [], []
        for i in ind:
            S, A, R, NS = self.storage[i]
            s.append(S)
            a.append(A)
            r.append(R)
            ns.append(NS)
        return np.array(s), np.array(a), np.array(r).reshape(-1, 1), np.array(ns)


class DDPG(object):
    def __init__(self, env):
        actor_LR = 0.0001
        critic_LR = 0.001
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high

        self.actor = self.create_actor()
        self.critic = self.create_critic()
        self.actor_target = self.create_actor()
        self.critic_target = self.create_critic()
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        self.actor_opt = optimizers.Adam(actor_LR)
        self.critic_opt = optimizers.Adam(critic_LR)

        self.buffer = Replay_buffer()
        self.batch_size = 100
        self.var = 3
        self.var_decay = 0.995
        self.gamma = 0.99
        self.tau = 0.01

    def create_actor(self):
        state_input = [Input(shape=[self.state_dim])]
        dense = Dense(400, activation='relu')(state_input[0])
        fc = Dense(300, activation='relu')(dense)
        fc = layers.Dense(self.action_dim, activation='tanh')(fc)
        out = tf.multiply(fc, self.action_bound)
        return tf.keras.models.Model(state_input, out)

    def create_critic(self):
        critic_input = [Input(shape=[self.state_dim]), Input(shape=[self.action_dim])]
        C_concat = layers.concatenate(critic_input)
        dense = Dense(400, activation='relu')(C_concat)
        fc = Dense(300, activation='relu')(dense)
        out = layers.Dense(1, activation=None)(fc)
        return tf.keras.models.Model(critic_input, out)

    def choose_action(self, state):
        a = np.array(self.actor(np.array([state]))[0])
        return np.clip(np.random.normal(a, self.var), -2, 2).astype(np.float32)

    def update(self):
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = actor_target_weights[i] * (1 - self.tau) + actor_weights[i] * self.tau
        for i in range(len(critic_weights)):
            critic_target_weights[i] = critic_target_weights[i] * (1 - self.tau) + critic_weights[i] * self.tau
        self.actor_target.set_weights(actor_target_weights)
        self.critic_target.set_weights(critic_target_weights)

    def train(self):
        self.var *= self.var_decay
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        next_actions = self.actor_target(next_states)
        Q = self.critic_target([next_states, next_actions])
        target_Q = rewards + self.gamma * Q

        with tf.GradientTape() as tape:
            predict_Q = self.critic([states, actions])
            td_error = losses.mean_squared_error(target_Q, predict_Q)
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            predict_actions = self.actor(states)
            val = self.critic([states, predict_actions])
            actor_loss = -tf.reduce_mean(val)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

        self.update()


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    env.seed(1)
    DDPG_agent = DDPG(env)

    episodes = 500
    max_steps = 200
    all_rewards = []
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = DDPG_agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            DDPG_agent.buffer.push((state, action, reward, next_state))
            total_reward += reward
            state = next_state
        all_rewards.append(total_reward)
        print("episode: {}/{}, rewards: {}".format(e + 1, episodes, total_reward))
        for i in range(50):
            DDPG_agent.train()
    plt.plot(all_rewards)
    plt.show()
    env.close()