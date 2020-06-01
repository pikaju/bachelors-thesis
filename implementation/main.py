import tensorflow as tf
import gym
import baselines

from gym import envs


def main():
    print(tf.add(tf.constant(2), tf.constant(3)))
    env = gym.make('CartPole-v0')
    print(env)


if __name__ == '__main__':
    main()
