import tensorflow as tf
import gym

from gym import envs

from baselines.deepq.replay_buffer import ReplayBuffer


def main():
    env = gym.make('CartPole-v0')
    env.reset()


if __name__ == '__main__':
    main()
