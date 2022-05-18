import gym
import math
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import itertools

device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_episode_num = 10000
BUFFER_SIZE = int(1e6)  # replay buffer size
ALPHA = 0.05  # initial temperature for SAC
TAU = 0.005  # soft update parameter
NUM_LEARN = 1  # number of learning
NUM_TIME_STEP = 1  # every NUM_TIME_STEP do update
REWARD_SCALE = 1        # reward scale
LR_ACTOR = 3e-4  # learning rate of the actor
LR_CRITIC = 3e-4  # learning rate of the critic
RANDOM_STEP = 10000  # number of random step


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed, device):
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# continuous action space
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, action_bound, hidden_dim=[256, 256], gamma=0.99, lr=1e-4):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.device = device
        self.action_bound = action_bound

        self.k = (action_bound[1] - action_bound[0]) / 2

        self.intermediate_dim_list = [state_dim] + hidden_dim

        self.layer_intermediate = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in
             zip(self.intermediate_dim_list[:-1], self.intermediate_dim_list[1:])]
        )

        self.mu_log_std_layer = nn.Linear(self.intermediate_dim_list[-1], 2 * self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.apply(self.weights_init_)

    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = state

        for linear_layer in self.layer_intermediate:
            x = self.relu(linear_layer(x))

        x = self.mu_log_std_layer(x)
        mu, log_std = x[:, :self.action_dim], torch.clamp(x[:, self.action_dim:], -20, 2)
        std = torch.exp(log_std)

        return mu, std

    def get_action_log_prob(self, state, stochstic=True):
        mu, std = self.forward(state)  # mu = (batch, num_action), std = (batch, num_action)

        ######### log_prob => see the appendix of paper
        var = std ** 2
        u = mu + std * torch.normal(mean=0, std=1, size=mu.shape).to(self.device).float()
        action = self.k * torch.tanh(u)
        gaussian_log_prob = -0.5 * ((u - mu) ** 2 / (var + 1e-6) + 2 * torch.log(std + 1e-6)).sum(dim=-1,
                                                                                                  keepdim=True) - 0.5 * \
                            mu.shape[-1] * math.log(2 * math.pi)

        log_prob = gaussian_log_prob - torch.log(self.k * (1 - (action / self.k) ** 2 + 1e-6)).sum(dim=-1,
                                                                                                   keepdim=True)  # (batch,)

        if not stochstic:
            action = mu.detach().cpu().numpy() * self.k

        return action, log_prob

    def get_action(self, state, stochastic=True):
        if not stochastic:
            self.eval()

        mu, std = self.forward(state)  # mu = (batch, num_action), std = (batch, num_action)

        ######### log_prob => see the appendix of paper
        var = std ** 2
        u = mu + std * torch.normal(mean=0., std=1., size=mu.shape).to(self.device).float()
        action = self.k * torch.tanh(u)

        if not stochastic:
            action = mu.detach().cpu().numpy() * self.k

        return action

    def learn(self, log_probs, Q_min, alpha):

        loss = -torch.mean(Q_min - alpha * log_probs)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=[256, 256], gamma=0.99, lr=1e-4):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.device = device

        self.dim_list = hidden_dim + [1]

        self.first_layer = nn.Linear(in_features=state_dim + action_dim, out_features=hidden_dim[0])

        self.layer_module = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(self.dim_list[:-1], self.dim_list[1:])]
        )

        self.activation = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.apply(self.weights_init_)

    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.first_layer(x))

        for layer in self.layer_module[:-1]:  # not include out layer
            x = self.activation(layer(x))

        x = self.layer_module[-1](x)

        return x

    def learn(self, states, actions, td_target_values):

        current_value = self.forward(states, actions)
        loss = torch.mean((td_target_values - current_value) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


# continuous action space
class SACAgent(nn.Module):
    def __init__(self, env, device, buffer_size=BUFFER_SIZE, reward_scale=REWARD_SCALE, batch_size=256, gamma=0.99,
                 lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, print_period=20, save_period=1000000):
        super(SACAgent, self).__init__()

        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.batch_size = batch_size
        self.print_period = print_period
        self.action_bound = [env.action_space.low[0], env.action_space.high[0]]
        self.tau = TAU
        self.reward_scale = reward_scale
        self.total_step = 0
        self.max_episode_time = env._max_episode_steps  # maximum episode time for the given environment
        self.log_file = 'sac_log2.txt'

        self.save_model_path = 'model/'
        self.save_period = save_period

        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=0, device=self.device)

        self.actor = Actor(self.state_dim, self.action_dim, self.device, self.action_bound, lr=self.lr_actor).to(
            self.device)

        self.local_critic_1 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        self.local_critic_2 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        self.target_critic_1 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        self.target_critic_2 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        iterator = itertools.chain(self.local_critic_1.parameters(), self.local_critic_2.parameters())
        self.critic_optimizer = optim.Adam(iterator, lr=self.lr_critic)

        self.H_bar = torch.tensor([-self.action_dim]).to(self.device).float()  # minimum entropy
        self.alpha = ALPHA
        self.log_alpha = torch.tensor([1.0], requires_grad=True, device=self.device).float()
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_actor)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def my_print(self, content):
        with open(self.log_file, 'a') as writer:
            print(content)
            writer.write(content + '\n')

    def save_model(self):
        save_path = self.save_model_path + 'SAC_Ant_{}.pth'.format(str(self.total_step))
        torch.save(self.state_dict(), save_path)

    def load_model(self, path=None):
        if path is None:
            self.load_state_dict(torch.load(self.save_model_path + 'SAC_Ant_1000000.pth'))
        else:
            self.load_state_dict(torch.load(path))

    def train(self, max_episode_num=1000, max_time=1000):
        self.my_print('######################### Start train #########################')
        self.episode_rewards = []
        self.critic_loss_list = []
        self.actor_loss_list = []
        self.tot_steps = []

        # copy parameters to target
        self.soft_update(self.local_critic_1, self.target_critic_1, 1.0)
        self.soft_update(self.local_critic_2, self.target_critic_2, 1.0)

        ts = []

        for episode_idx in range(max_episode_num):
            state = self.env.reset()
            episode_reward = 0.
            temp_critic_loss_list = []
            temp_actor_loss_list = []
            for t in range(1, max_time + 1):
                if self.total_step < RANDOM_STEP:
                    action = self.env.action_space.sample()
                else:
                    action = self.actor.get_action(torch.tensor([state]).to(self.device).float())
                    action = action.squeeze(dim=0).detach().cpu().numpy()
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                masked_done = False if t == self.max_episode_time else done

                self.memory.add(state, action, reward, next_state, masked_done)

                if len(self.memory) < self.batch_size or t % NUM_TIME_STEP != 0: continue

                for _ in range(NUM_LEARN):
                    # Randomly sample a batch of trainsitions from D
                    states, actions, rewards, next_states, dones = self.memory.sample()

                    # Compute targets for the Q functions
                    # next_actions, next_log_probs = torch.tensor(self.actor.get_action_log_prob(next_states)).float().to(self.device)
                    with torch.no_grad():
                        sampled_next_actions, next_log_probs = self.actor.get_action_log_prob(next_states)
                        Q_target_1 = self.target_critic_1.forward(next_states, sampled_next_actions).detach()
                        Q_target_2 = self.target_critic_2.forward(next_states, sampled_next_actions).detach()
                        y = self.reward_scale * rewards + self.gamma * (1 - dones) * (
                                    torch.min(Q_target_1, Q_target_2) - self.alpha * next_log_probs)

                    # Update Q-functions by one step of gradient descent
                    Q_1_current_value = self.local_critic_1.forward(states, actions)
                    Q_loss_1 = torch.mean((y - Q_1_current_value) ** 2)
                    Q_2_current_value = self.local_critic_2.forward(states, actions)
                    Q_loss_2 = torch.mean((y - Q_2_current_value) ** 2)
                    Q_loss = Q_loss_1 + Q_loss_2
                    self.critic_optimizer.zero_grad()
                    Q_loss.backward()
                    self.critic_optimizer.step()
                    temp_critic_loss_list.append(Q_loss.item())

                    # Update policy by one step of gradient ascent
                    sampled_actions, log_probs = self.actor.get_action_log_prob(states)
                    Q_min = torch.min(self.local_critic_1.forward(states, sampled_actions),
                                      self.local_critic_2.forward(states, sampled_actions))
                    policy_loss = self.actor.learn(log_probs, Q_min, self.alpha)
                    temp_actor_loss_list.append(policy_loss)

                    # Adjust temperature
                    loss_log_alpha = self.log_alpha * (-log_probs.detach() - self.H_bar).mean()
                    self.log_alpha_optimizer.zero_grad()
                    loss_log_alpha.backward()
                    self.log_alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().detach()

                    # Update target networks
                    self.soft_update(self.local_critic_1, self.target_critic_1, self.tau)
                    self.soft_update(self.local_critic_2, self.target_critic_2, self.tau)

                state = next_state

                self.total_step += 1

                if self.total_step % self.save_period == 0:
                    self.save_model()

                if done: break

            ts.append(t)
            self.tot_steps.append(self.total_step)
            self.episode_rewards.append(episode_reward)
            self.critic_loss_list.append(np.mean(temp_critic_loss_list))
            self.actor_loss_list.append(np.mean(temp_actor_loss_list))
            if (episode_idx + 1) % self.print_period == 0:
                content = 'Tot_step: {0:<6} \t | Episode: {1:<4} \t | Time: {2:5.2f} \t | Reward : {3:5.3f} \t | actor_loss : {4:5.3f} \t | critic_loss : {5:5.3f}'.format(
                    self.total_step, episode_idx + 1, np.mean(ts), np.mean(self.episode_rewards[-self.print_period:]),
                    np.mean(self.actor_loss_list[-self.print_period:]),
                    np.mean(self.critic_loss_list[-self.print_period:]))
                self.my_print(content)

                ts = []


class Trainer:
    def __init__(self, env, agent: SACAgent):
        self.env = env
        self.agent = agent

    def train(self):
        self.agent.train()

    def plot_reward(self):
        plt.plot(self.agent.episode_rewards)
        plt.xlabel("episode")
        plt.ylabel("episode_reward")
        plt.title('SAC Ant Reward')
        plt.show()


def train_sac():
    env = gym.make('Ant-v2')
    agent = SACAgent(env, device)
    agent.train(max_episode_num=max_episode_num, max_time=5000)


train_sac()
