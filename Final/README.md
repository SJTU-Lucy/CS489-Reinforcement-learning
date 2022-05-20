# README



To run this program, you needs to provide three arguments.

1. **--env_name**: the name of environment, either **PongNoFrameskip-v4** or Ant-v2.
2. **--method**: the algorithm used. **DQN** and **DDQN** for Pong, **PPO** and **SAC** for Ant-v2.
3. **--action**: whether to **train** or **test** model.

So if you'd like to **test** **PPO** on **Ant-v2**, the launch code should be like:

```
python run.py --env_name=Ant-v2 --method=PPO --action=test
```

