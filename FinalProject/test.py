import gym

# pong
# action_space:6 observation_space: (210, 160, 3)
env = gym.make("PongNoFrameskip-v4")
action_space = env.action_space
state_sapce = env.observation_space
print(action_space.n)
print(state_sapce.shape)