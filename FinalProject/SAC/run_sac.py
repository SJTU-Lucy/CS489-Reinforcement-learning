import gym
import numpy as np
from SAC.sac import SAC
from SAC.replay_memory import ReplayMemory

BATCH_SIZE = 256
MEMORY_SIZE = 1e6
START_STEPS = 10000
MAX_STEPS = 1e6


def run_sac(env_name, alpha=0.2, auto_e_tune=False, model_path=None):
    env = gym.make(env_name)
    input_size = env.observation_space.shape[0]
    action_space = env.action_space

    agent = SAC(input_size, action_space, alpha, auto_e_tune)
    memory = ReplayMemory(MEMORY_SIZE)

    steps = 0
    i_episode = 0
    updates = 0
    reward_list = []
    alpha_list = []

    while steps < MAX_STEPS:
        reward_sum = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if steps < START_STEPS:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > BATCH_SIZE:
                alpha = agent.update_parameters(memory, BATCH_SIZE, updates)
                updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            steps += 1
            reward_sum += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            state = next_state
            alpha_list.append(alpha)

        reward_list.append(reward_sum)
        i_episode += 1

        if i_episode % 100 == 0:
            print("Total steps:{}, Episode: {} Reward: {:.2f}".format(steps, i_episode, reward_sum))

        if i_episode % 1000 == 0:
            agent.save_model(env_name)

    env.close()
    agent.save_model(env_name, model_path)
    np.save(env_name + '_train', np.array(reward_list))
    np.save(env_name + '_alpha', np.array(alpha_list))

