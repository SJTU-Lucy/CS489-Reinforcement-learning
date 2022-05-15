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
MAX_STEPS = 3e5
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

    def surrogate_loss(self, advants, states, old_policy, actions, index):
        mu, sigma, log_sigma = self.actor(states)
        pi = torch.distributions.Normal(mu, sigma)
        new_policy = pi.log_prob(actions).sum(1, keepdim=True)
        old_policy = old_policy[index]
        ratio = torch.exp(new_policy - old_policy)
        surrogate = ratio * advants
        return surrogate, ratio

    def train(self, memory):
        memory = np.array(memory, dtype=object)
        states = torch.Tensor(np.vstack(memory[:, 0])).to(device)
        actions = torch.Tensor(list(memory[:, 1])).to(device)
        rewards = torch.Tensor(list(memory[:, 2])).to(device)
        masks = torch.Tensor(list(memory[:, 3])).to(device)
        values = self.critic(states)

        returns, advants = get_gae(rewards, masks, values)
        mu, sigma, log_sigma = self.actor(states)

        pi = torch.distributions.Normal(mu, sigma)
        old_policy = pi.log_prob(actions).sum(1, keepdim=True)

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)
        for epoch in range(10):
            np.random.shuffle(arr)

            for i in range(n // BATCH_SIZE):
                batch_index = arr[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]  # batch_size * state_size
                inputs = states[batch_index]
                returns_samples = returns.unsqueeze(1)[batch_index]  # batch_size * 1
                advants_samples = advants.unsqueeze(1)[batch_index]  # batch_size * 1
                actions_samples = actions[batch_index]  # batch_size * action_size

                loss, ratio = self.surrogate_loss(advants_samples, inputs,
                                                  old_policy.detach(), actions_samples, batch_index)

                values = self.critic(inputs)
                critic_loss = criterion(values, returns_samples)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPISILON, 1.0 + CLIP_EPISILON)
                clipped_loss = clipped_ratio * advants_samples
                actor_loss = -torch.min(loss, clipped_loss).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()


class Trainer:
    def __init__(self, env, agent: PPOAgent):
        self.env = env
        self.agent = agent
        self.rewardlist = []

    def train(self):
        steps = 0
        episode = 0
        memory = deque()
        while steps < MAX_STEPS:
            state = self.env.reset()
            reward_sum = 0
            for __ in range(3000):
                steps += 1
                mu, std, _ = self.agent.actor(torch.Tensor(state).unsqueeze(0).to(device))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = self.env.step(action)
                memory.append([state, action, reward, 1 - done])
                reward_sum += reward
                state = next_state
                if steps % UPDATE_STEPS == 0:
                    self.agent.train(memory)
                    memory.clear()
                if done:
                    break
            episode += 1
            self.rewardlist.append(reward_sum)
            print('Episode: {} \t step: {}/{} \t Total reward: {}'.format(episode, steps, MAX_STEPS, reward_sum))
            if episode % 500 == 0:
                torch.save({'actor': self.agent.actor.state_dict(), 'critic': self.agent.critic.state_dict()},
                           "PPO/model/PPO_Ant_{}.pt".format(episode))

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
    trainer = Trainer(env, agent)
    trainer.train()
    trainer.plot_reward()
