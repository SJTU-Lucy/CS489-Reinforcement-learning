import gym
import time
import numpy as np
from tqdm import tqdm
from SAC.sac import SAC


def test_sac(env_name, n_episodes=100, model_path=None, render=False):
    if model_path is None:
        model_path = 'trained_models/' + env_name + '_model.pth.tar'

    env = gym.make(env_name)
    input_size = env.observation_space.shape[0]
    action_space = env.action_space

    agent = SAC(input_size, action_space)
    agent.load_model(model_path)
    agent.policy.eval()
    agent.critic.eval()
    agent.critic_target.eval()

    reward_list = []
    for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        reward_sum = 0

        while True:
            if render:
                env.render()
                time.sleep(0.02)

            action = agent.select_action(state, evaluate=True)

            next_state, reward, done, _ = env.step(action)
            state = next_state
            reward_sum += reward

            if done:
                break

        reward_list.append(reward_sum)

    env.close()
    reward_list = np.array(reward_list)
    avg_r = np.mean(reward_list)
    std_r = np.std(reward_list)
    print("Reward sum avg: ", avg_r, "std: ", std_r)
