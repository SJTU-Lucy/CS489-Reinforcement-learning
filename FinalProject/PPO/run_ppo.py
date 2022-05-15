import gym
from collections import deque
from PPO.ppo import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_STEPS = 3e6
UPDATE_STEPS = 2048


def run_ppo(env_name, model_path=None):
    if model_path is None:
        model_path = 'trained_models/' + env_name + '_model.pth.tar'

    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    ppo = PpoAgent(state_size, action_size)

    steps = 0
    episode = 0
    reward_list = []

    memory = deque()
    while steps < MAX_STEPS:
        # episode start
        state = env.reset()
        reward_sum = 0
        for _ in range(1000):  # episode max steps
            steps += 1
            mu, std, _ = ppo.actor(torch.Tensor(state).unsqueeze(0).to(device))
            action = get_action(mu, std)[0]
            next_state, reward, done, _ = env.step(action)

            if done:
                mask = 0
            else:
                mask = 1

            memory.append([state, action, reward, mask])

            reward_sum += reward
            state = next_state

            if steps % UPDATE_STEPS == 0:
                ppo.train(memory)
                memory.clear()

            if done:
                break

        episode += 1
        reward_list.append(reward_sum)

        if episode % 100 == 0:
            print('Total steps:{}, Episode: {} Reward: {.2f}'.format(steps, episode, reward_sum))
            torch.save({'actor': ppo.actor.state_dict(), 'critic': ppo.critic.state_dict()}, model_path)

    torch.save({'actor': ppo.actor.state_dict(), 'critic': ppo.critic.state_dict()}, model_path)
    np.save(env_name + '_train', np.array(reward_list))
