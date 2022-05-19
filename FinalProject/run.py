import argparse
from DQN.test_dqn import test_dqn
from DQN.train_dqn import train_dqn
from DDQN.test_ddqn import test_ddqn
from DDQN.train_ddqn import train_ddqn
from PPO.test_ppo import test_ppo
from PPO.train_ppo import train_ppo
from SAC.test_sac import test_sac
from SAC.train_sac import train_sac

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the environment')
parser.add_argument('--method', type=str, default='PPO', help='method for Atari and MuJoCo')
parser.add_argument('--action', type=str, default='test', help='run train or test')
args = parser.parse_args()

if args.env_name == 'PongNoFrameskip-v4':
    if args.method == 'DQN':
        print('Environment: ' + args.env_name + ' Method: DQN')
        if args.action == 'test':
            test_dqn()
        elif args.action == 'train':
            train_dqn()
    elif args.method == 'DDQN':
        print('Environment: ' + args.env_name + ' Method: DDQN')
        if args.action == 'test':
            test_ddqn()
        elif args.action == 'train':
            train_ddqn()


elif args.env_name == 'Ant-v2':
    if args.method == 'PPO':
        print('Environment: ' + args.env_name + ' Method: PPO')
        if args.action == 'test':
            test_ppo(render=True)
        elif args.action == 'train':
            train_ppo()
    elif args.method == 'SAC':
        print('Environment: ' + args.env_name + ' Method: SAC')
        if args.action == 'test':
            test_sac()
        elif args.action == 'train':
            train_sac()
