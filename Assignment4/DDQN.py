import tensorflow as tf
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
from collections import deque
import random
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt


class DQN(object):
    def __init__(self):
        self.step = 0
        self.update_freq = 600
        self.replay_size = 10000
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.greedy = 1

    def create_model(self):
        STATE_DIM, ACTION_DIM = 2, 3
        model = models.Sequential([
            layers.Dense(64, input_dim=STATE_DIM, activation='tanh'),
            layers.Dense(64, input_dim=64, activation='tanh'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model

    # 选择最佳动作，如果小于greedy的值，就随机，否则挑选最佳的
    def get_best_action(self, s):
        if np.random.uniform() < self.greedy:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]))[0])

    def save_model(self, file_path='mountaincar-ddqn.h5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward, done):
        self.replay_queue.append((s, a, next_s, reward, done))

    def update_greedy(self):
        # 小于最小探索率的时候就不进行更新了。
        if self.greedy > 0.01:
            self.greedy *= 0.99

    def train(self, batch_size=32, factor=1):
        self.step += 1
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q1 = self.target_model.predict(next_s_batch)
        Q2 = self.model.predict(next_s_batch)
        next_action = np.argmax(Q2, axis=1)

        for i, replay in enumerate(replay_batch):
            _, a, next_s, reward, done = replay
            Q[i][a] = reward + factor * Q1[i][next_action[i]] * (1 - done)
        self.model.fit(s_batch, Q, verbose=0)


env = gym.make('MountainCar-v0')
episodes = 1000
reward_list = []
agent = DQN()
for episode in range(episodes):
    s = env.reset()
    total_reward = 0
    if episode >= 5:
        agent.update_greedy()
    while True:
        a = agent.get_best_action(s)
        next_s, reward, done, _ = env.step(a)
        agent.remember(s, a, next_s, reward, done)
        total_reward += reward
        if episode >= 5:
            agent.train()
        s = next_s
        if done:
            reward_list.append(total_reward)
            print('episode:', episode, 'reward:', total_reward, 'max_reward:', max(reward_list))
            break
    if np.mean(reward_list[-10:]) > -130:
        agent.save_model()
        break
env.close()

plt.plot(reward_list, color='green')
plt.show()