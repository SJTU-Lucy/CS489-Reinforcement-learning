import os
import argparse
from test_pong import test_pong
from PPO.test_ppo import test_ppo

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the environment to run')
parser.add_argument('--method', type=str, default='PPO', help='Method to use in MuJoCo environments, PPO or SAC')
args = parser.parse_args()

if args.env_name == 'PongNoFrameskip-v4':
    print('Environment: ' + args.env_name + ' Method: DQN')
    test_pong()
elif args.env_name == 'Ant-v2':
    print('Environment: ' + args.env_name + ' Method: PPO')
    test_ppo(args.env_name)