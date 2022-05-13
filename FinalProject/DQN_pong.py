import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import gym
import random
from collections import namedtuple
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))

# epsilon = 0.9
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
RENDER = False
lr = 1e-3
INITIAL_MEMORY = 10000
MEMORY_SIZE = 10 * INITIAL_MEMORY
n_episode = 2000

MODEL_STORE_PATH = os.getcwd()
print(MODEL_STORE_PATH)
modelname = 'DQN_Pong'
madel_path = MODEL_STORE_PATH + '/' + 'model/' + 'DQN_Pong_episode900.pt'


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn is not None else x
        return x


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        super(DQN, self).__init__()
        # (1, 3, 210, 160)
        # o = (i + 2p - k) / s + 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(8, 8), padding=(4, 4), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), padding=(2, 2), stride=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        self.conv = BasicConv2d(in_channels, 32, kernel_size=(8, 8), bn=True)

    def forward(self, x):
        x = x.float() / 255
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class DQN_agent():
    def __init__(self, in_channels=1, action_space=[], learning_rate=1e-3, memory_size=100000, epsilon=0.99):
        self.in_channels = in_channels
        self.action_space = action_space
        self.action_dim = self.action_space.n

        self.memory_buffer = ReplayMemory(memory_size)
        self.stepdone = 0
        self.DQN = DQN(self.in_channels, self.action_dim).cuda()
        self.target_DQN = DQN(self.in_channels, self.action_dim).cuda()
        # 加载之前训练好的模型
        # self.DQN.load_state_dict(torch.load(madel_path))
        # self.target_DQN.load_state_dict(self.DQN.state_dict())
        self.optimizer = optim.RMSprop(self.DQN.parameters(), lr=learning_rate, eps=0.001, alpha=0.95)

    def select_action(self, state):
        self.stepdone += 1
        state = state.to(device)
        epsilon = 0.99
        if random.random() < epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        else:
            action = self.DQN(state).detach().max(1)[1].view(1, 1)
        return action

    def learn(self):
        if self.memory_buffer.__len__() < BATCH_SIZE:
            return
        transitions = self.memory_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8).bool()
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to('cuda')
        state_batch = torch.cat(batch.state).to('cuda')
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        state_action_values = self.DQN(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_DQN(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.DQN.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class Trainer():
    def __init__(self, env, agent, n_episode):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        self.rewardlist = []

    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)  # 转化为四维的数据结构

    def train(self):
        for episode in range(0, self.n_episode):
            obs = self.env.reset()
            # state shape of (1, 3, 210, 160)
            state = self.get_state(obs)
            episode_reward = 0.0
            print('episode:', episode)
            for t in count():
                action = self.agent.select_action(state)
                if RENDER:
                    self.env.render()
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                if not done:
                    next_state = self.get_state(obs)
                else:
                    next_state = None
                reward = torch.tensor([reward], device=device)
                # 将四元组存到memory中
                '''
                state: batch_size channel h w    size: batch_size * 4
                action: size: batch_size * 1
                next_state: batch_size channel h w    size: batch_size * 4
                reward: size: batch_size * 1                
                '''
                self.agent.memory_buffer.push(state, action.to('cpu'), next_state, reward.to('cpu'))
                state = next_state
                # 经验池满了之后开始学习
                if self.agent.stepdone > INITIAL_MEMORY:
                    self.agent.learn()
                    if self.agent.stepdone % TARGET_UPDATE == 0:
                        self.agent.target_DQN.load_state_dict(self.agent.DQN.state_dict())
                if done:
                    break
            print("reward = ", episode_reward)
            if episode % 20 == 0:
                torch.save(self.agent.DQN.state_dict(),
                           MODEL_STORE_PATH + '/' + "model/{}_episode{}.pt".format(modelname, episode))
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.agent.stepdone, episode, t,
                                                                                     episode_reward))
            self.rewardlist.append(episode_reward)
            self.env.close()
        return

    def plot_reward(self):
        plt.plot(self.rewardlist)
        plt.xlabel("episode")
        plt.ylabel("episode_reward")
        plt.title('train_reward')
        plt.show()


if __name__ == '__main__':
    # create environment
    env = gym.make("PongNoFrameskip-v4")
    action_space = env.action_space
    state_channel = env.observation_space.shape[2]
    agent = DQN_agent(in_channels=state_channel, action_space=action_space)
    trainer = Trainer(env, agent, n_episode)
    trainer.train()
    trainer.plot_reward()






