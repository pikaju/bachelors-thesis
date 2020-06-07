import tensorflow as tf
import gym

from gym import envs

from baselines.deepq.replay_buffer import ReplayBuffer


def main():
    env = gym.make('CartPole-v0')
    print('Observation space:', env.observation_space)
    print('Action space:', env.action_space)
    while True:
        obs_t = env.reset()
        while True:
            env.render()
            action = env.action_space.sample()
            obs_tp1, reward, done, info = env.step(action)

            if done:
                break
            obs_t = obs_tp1


if __name__ == '__main__':
    main()
