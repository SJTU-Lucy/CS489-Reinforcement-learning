import torch
import torch.nn.functional as F
from torch.optim import Adam
from SAC.models import GaussianPolicy, QNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'

GAMMA = 0.99
TAU = 0.005
lr = 3e-4
HIDDEN_SIZE = 256


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SAC(object):
    def __init__(self, input_size, action_space, alpha=0.2, auto_e_tune=False):
        self.alpha = alpha
        self.target_update_interval = 1
        self.automatic_entropy_tuning = auto_e_tune

        self.critic = QNetwork(input_size, action_space.shape[0], HIDDEN_SIZE).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(input_size, action_space.shape[0], HIDDEN_SIZE).to(device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A)
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self.policy = GaussianPolicy(input_size, action_space.shape[0], HIDDEN_SIZE, action_space).to(device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * GAMMA * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)  # f(εt;st)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, TAU)

        return self.alpha

    def save_model(self, env_name, model_path=None):
        if model_path is None:
            model_path = 'trained_models/' + env_name + '_model.pth.tar'

        torch.save({'actor': self.policy.state_dict(), 'critic': self.critic.state_dict()}, model_path)

    def load_model(self, model_path):
        self.policy.load_state_dict(torch.load(model_path)['actor'])
        self.critic.load_state_dict(torch.load(model_path)['critic'])
