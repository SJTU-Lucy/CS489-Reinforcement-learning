import os
import argparse
from test_pong import test_pong

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the environment to run')
args = parser.parse_args()

if args.env_name == 'PongNoFrameskip-v4':
    print('Environment: ' + args.env_name + ' Method: DQN')
    test_pong()
else:
    print("Wrong environment name!")
