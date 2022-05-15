import numpy as np
import torch
import torch.optim as optim
from PPO.models import Actor, Critic

GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 64
actor_lr = 0.0003
critic_lr = 0.0003
l2_rate = 0.001
CLIP_EPISILON = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_gae(rewards, masks, values):
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + GAMMA * running_returns * masks[t]
        running_tderror = rewards[t] + GAMMA * previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + GAMMA * LAMDA * running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def get_action(mu, sigma):
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample().cpu().numpy()
    return action


class PpoAgent:
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

        # ----------------------------
        # step 1: get returns and GAEs and log probability of old policy
        returns, advants = get_gae(rewards, masks, values)
        mu, sigma, log_sigma = self.actor(states)

        pi = torch.distributions.Normal(mu, sigma)
        old_policy = pi.log_prob(actions).sum(1, keepdim=True)

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)

        # ----------------------------
        # step 2: get value loss and actor loss and update actor & critic
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
