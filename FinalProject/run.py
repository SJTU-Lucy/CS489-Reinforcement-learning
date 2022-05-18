import argparse
from DQN.test_pong import test_pong
from DQN.train_pong import train_pong
from PPO.test_ppo import test_ppo
from PPO.train_ppo import train_ppo
from SAC.test_sac import test_sac
from SAC.train_sac import train_sac

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the environment')
parser.add_argument('--method', type=str, default='PPO', help='method for MuJoCo, PPO or SAC')
parser.add_argument('--action', type=str, default='test', help='run train or test')
args = parser.parse_args()

if args.env_name == 'PongNoFrameskip-v4':
    print('Environment: ' + args.env_name + ' Method: DQN')
    if args.action == 'test':
        test_pong()
    elif args.action == 'train':
        train_pong()

elif args.env_name == 'Ant-v2':
    print('Environment: ' + args.env_name + ' Method: PPO')
    if args.method == 'PPO':
        if args.action == 'test':
            test_ppo(render=True)
        elif args.action == 'train':
            train_ppo()
    elif args.method == 'SAC':
        if args.action == 'test':
            test_sac()
        elif args.action == 'train':
            train_sac()
