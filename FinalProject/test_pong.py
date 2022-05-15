import time
import torch
import numpy as np
import gym
from DQN import DQN
from wrapper import make_atari

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def test_pong():
    # model_path = 'model/DQN_Pong_best.pt'
    model_path = 'model/DQN_Pong_episode_850.pt'
    env = make_atari("PongNoFrameskip-v4", max_episode_steps=10000)
    agent = DQN(in_channels=env.observation_space.shape[2], action_space=env.action_space)
    agent.net.load_state_dict(torch.load(model_path))
    obs = env.reset()
    state = get_state(obs)
    reward_sum = 0
    while True:
        env.render()
        time.sleep(0.01)
        action = agent.select_action(state)
        obs, reward, done, _ = env.step(action)
        next_state = get_state(obs)
        reward_sum += reward
        state = next_state
        if done:
            break
    print("total reward = ", reward_sum)
    env.close()
