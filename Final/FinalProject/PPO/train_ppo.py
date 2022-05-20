import gym
from collections import deque
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA = 0.99
LAMBDA = 0.95
BATCH_SIZE = 64
actor_lr = 0.0003
critic_lr = 0.0003
l2_rate = 0.001
CLIP_EPISILON = 0.2
UPDATE_STEPS = 2048


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_size)
        self.sigma = nn.Linear(64, action_size)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        sigma = torch.exp(log_sigma)
        return mu, sigma, log_sigma


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


def get_gae(rewards, masks, values):
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + GAMMA * running_returns * masks[t]
        running_tderror = rewards[t] + GAMMA * previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + GAMMA * LAMBDA * running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def get_action(mu, sigma):
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample().cpu().numpy()
    return action


class PPOAgent:
    def __init__(self, state_size, action_size):
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=l2_rate)

    def train(self, memory):
        memory = np.array(memory)
        states = torch.Tensor(np.vstack(memory[:, 0])).to(device)
        actions = torch.Tensor(list(memory[:, 1])).to(device)
        rewards = torch.Tensor(list(memory[:, 2])).to(device)
        masks = torch.Tensor(list(memory[:, 3])).to(device)
        values = self.critic(states)

        returns, advants = get_gae(rewards, masks, values)
        old_mu, old_std, old_log_std = self.actor(states)
        pi = torch.distributions.Normal(old_mu, old_std)
        old_log_prob = pi.log_prob(actions).sum(1, keepdim=True)

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n//BATCH_SIZE):
                b_index = arr[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                mu, std, log_std = self.actor(b_states)
                pi = torch.distributions.Normal(mu, std)
                new_prob = pi.log_prob(b_actions).sum(1, keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                ratio = torch.exp(new_prob-old_prob)

                surrogate_loss = ratio * b_advants
                values = self.critic(b_states)
                critic_loss = criterion(values, b_returns)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ratio = torch.clamp(ratio, 1.0 - CLIP_EPISILON, 1.0 + CLIP_EPISILON)
                clipped_loss = ratio * b_advants
                actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()


class Normalize:
    def __init__(self, state_size):
        self.mean = np.zeros((state_size,))
        self.std = np.zeros((state_size, ))
        self.stdd = np.zeros((state_size, ))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean
        x = x - self.mean
        x = x / (self.std + 1e-8)
        x = np.clip(x, -5, +5)
        return x


class Trainer:
    def __init__(self, env, agent: PPOAgent, normal: Normalize):
        self.env = env
        self.agent = agent
        self.rewardlist = []
        self.normalize = normal

    def train(self):
        episodes = 0
        for iter in range(3000):
            memory = deque()
            scores = []
            steps = 0
            while steps < 2048:
                state = self.normalize(self.env.reset())
                score = 0
                for _ in range(10000):
                    steps += 1
                    mu, std, _ = self.agent.actor(torch.Tensor(state).unsqueeze(0).to(device))
                    action = get_action(mu, std)[0]
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.normalize(next_state)
                    mask = (1 - done) * 1
                    memory.append([state, action, reward, mask])
                    score += reward
                    state = next_state
                    if done:
                        break
                episodes += 1
                scores.append(score)
                self.rewardlist.append(score)
                if episodes % 1000 == 0:
                    torch.save({'actor': self.agent.actor.state_dict(), 'critic': self.agent.critic.state_dict(),
                                'norm': [self.normalize.mean, self.normalize.std, self.normalize.stdd, self.normalize.n]},
                               "model/PPO_Ant_{}.pt".format(episodes))
            score_avg = np.mean(scores)
            print('{} episode score is {}'.format(episodes, score_avg))
            self.agent.train(memory)

    def plot_reward(self):
        plt.plot(self.rewardlist)
        plt.xlabel("episode")
        plt.ylabel("episode_reward")
        plt.title('train_reward')
        plt.show()


def train_ppo():
    env = gym.make('Ant-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = PPOAgent(state_size=state_size, action_size=action_size)
    normalize = Normalize(state_size=state_size)
    trainer = Trainer(env, agent, normalize)
    trainer.train()
    trainer.plot_reward()
