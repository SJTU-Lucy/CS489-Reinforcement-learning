import gym
import numpy as np
import torch
from SAC.train_sac import SACAgent


def test_sac(n_episodes=100, model_path=None, render=False):
    env = gym.make('Ant-v2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = SACAgent(env, device)
    agent.load_model(agent.save_model_path + 'SAC_Ant_1000000.pth')
    num_test = 5
    episode_rewards = []
    for _ in range(num_test):
        state = env.reset()
        episode_reward = 0.
        for _ in range(10000):
            action = agent.actor.get_action(torch.tensor([state]).to(device).float(), stochastic=False)
            env.render()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        episode_rewards.append(episode_reward)
    print('avg_episode_reward : ', np.mean(episode_rewards))
