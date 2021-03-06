import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from DDQN.wrapper import make_atari

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
POLICY_UPDATE = 2
lr = 1e-3
INITIAL_MEMORY = 10000
MEMORY_SIZE = 100000
n_episode = 1000


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
    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            stride=stride
        )
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Qnet(nn.Module):
    def __init__(self, in_channels, n_actions):
        super(Qnet, self).__init__()
        # input of (1, 3, 210, 160)
        self.conv1 = BasicConv2d(in_channels, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = BasicConv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = BasicConv2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
        self.fc4 = nn.Linear(14 * 11 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class DDQNAgent:
    def __init__(self, in_channels, action_space, memory_size=MEMORY_SIZE, epsilon=EPS_START):
        self.in_channels = in_channels
        self.action_space = action_space
        self.action_dim = self.action_space.n
        self.memory_buffer = ReplayMemory(memory_size)
        self.stepdone = 0
        self.net = Qnet(self.in_channels, self.action_dim).cuda()
        self.target_DDQN = Qnet(self.in_channels, self.action_dim).cuda()
        self.target_DDQN.load_state_dict(self.net.state_dict())
        self.learning_rate = lr
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.learning_rate, eps=0.001, alpha=0.95)
        self.epsilon = epsilon

    def select_action(self, state):
        self.stepdone += 1
        state = state.to(device)
        if self.epsilon >= EPS_END:
            self.epsilon *= 0.995
        if random.random() < self.epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        else:
            action = self.net(state).detach().max(1)[1].view(1, 1)
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
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to('cuda')
        state_batch = torch.cat(batch.state).to('cuda')
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        state_action_values = self.net(state_batch).gather(1, action_batch)
        Q1 = self.target_DDQN(non_final_next_states)
        Q2 = self.net(non_final_next_states)
        next_action = torch.argmax(Q2, dim=1)
        expected_state_action_values = []
        count = 0
        for i in range(BATCH_SIZE):
            if non_final_mask[i].item():
                val = Q1[count][next_action[count]].item() * GAMMA + reward_batch[i].item()
                expected_state_action_values.append([val])
                count += 1
            else:
                val = reward_batch[i].item()
                expected_state_action_values.append([val])
        expected_state_action_values = torch.Tensor(expected_state_action_values).to(device)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        '''
        Q = self.model.predict(s_batch)
        Q1 = self.target_model.predict(next_s_batch)
        Q2 = self.model.predict(next_s_batch)
        next_action = np.argmax(Q2, axis=1)

        for i, replay in enumerate(replay_batch):
            _, a, next_s, reward, done = replay
            Q[i][a] = reward + factor * Q1[i][next_action[i]] * (1 - done)
        self.model.fit(s_batch, Q, verbose=0)
        '''


class Trainer:
    def __init__(self, env, agent: DDQNAgent, n_episode):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        self.rewardlist = []

    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self):
        for episode in range(0, self.n_episode):
            obs = self.env.reset()
            # state shape of (1, 3, 210, 160)
            state = self.get_state(obs)
            episode_reward = 0.0
            for t in count():
                action = self.agent.select_action(state)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                if not done:
                    next_state = self.get_state(obs)
                else:
                    next_state = None
                reward = torch.tensor([reward], device=device)
                # ??????????????????memory???
                self.agent.memory_buffer.push(state, action.to('cpu'), next_state, reward.to('cpu'))
                state = next_state
                # ?????????????????????????????????
                if self.agent.stepdone > INITIAL_MEMORY and self.agent.stepdone % POLICY_UPDATE == 0:
                    self.agent.learn()
                    if self.agent.stepdone % TARGET_UPDATE == 0:
                        self.agent.target_DDQN.load_state_dict(self.agent.net.state_dict())
                if done:
                    break
            print('Episode: {}/{} \t Total steps: {} \t Total reward: {}'
                  .format(episode, self.n_episode, self.agent.stepdone, episode_reward))
            if episode % 100 == 0:
                torch.save(self.agent.net.state_dict(), "DDQN/model/DDQN_Pong_episode_{}.pt".format(episode))
            self.rewardlist.append(episode_reward)
            self.env.close()
        return

    def plot_reward(self):
        plt.plot(self.rewardlist)
        plt.xlabel("episode")
        plt.ylabel("episode_reward")
        plt.title('train_reward')
        plt.show()


def train_ddqn():
    env = make_atari("PongNoFrameskip-v4", max_episode_steps=10000)
    action_space = env.action_space
    state_channel = env.observation_space.shape[2]
    agent = DDQNAgent(in_channels=state_channel, action_space=action_space)
    trainer = Trainer(env, agent, n_episode)
    trainer.train()
    trainer.plot_reward()
