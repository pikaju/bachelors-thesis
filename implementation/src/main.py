import gym
from baselines.a2c import a2c
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.bench import Monitor

import scenario.register

# from muzero.train import train as train_muzero


def main():
    log_dir = './logs/a2c'
    a2c.learn('mlp', SubprocVecEnv(
        [lambda: Monitor(gym.make('BachelorThesis-v0'), log_dir) for _ in range(1)]))


if __name__ == '__main__':
    main()
