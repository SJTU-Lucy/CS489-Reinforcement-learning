import time
import gym
from tqdm import tqdm
import torch
import numpy as np
from PPO.train_ppo import Actor, Critic, get_action


def test_ppo(n_episodes=100, model_path=None, render=True):
    env = gym.make('Ant-v2')
    input_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = Actor(input_size, action_size)
    critic = Critic(input_size)

    model_path = 'PPO/model/PPO_Ant_1000.pt'
    pretrained_model = torch.load(model_path)
    actor.load_state_dict(pretrained_model['actor'])
    critic.load_state_dict(pretrained_model['critic'])
    actor.eval()
    critic.eval()

    reward_list = []
    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        reward_sum = 0
        while True:
            if render:
                env.render()
                time.sleep(0.02)
            mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
            action = get_action(mu, std)[0]
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = next_state
            if done:
                break
        reward_list.append(reward_sum)
    env.close()
    reward_list = np.array(reward_list)
    avg_r = np.mean(reward_list)
    std_r = np.std(reward_list)
    print("Reward sum avg: ", avg_r, "std: ", std_r)
