import tensorflow as tf
import numpy as np
import gym

from gym import envs

from baselines.deepq.replay_buffer import ReplayBuffer
from muzero import MuZeroPSO


state_shape = 4


def define_representation(env):
    obs_shape = env.observation_space.shape
    representation = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation=tf.nn.relu, input_shape=obs_shape),
        tf.keras.layers.Dense(state_shape, activation=tf.nn.relu),
    ], name='representation')
    return representation


def define_model(env):
    representation = define_representation(env)

    action_shape = env.action_space.n
    print(action_shape)

    dynamics_trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(state_shape, activation=tf.nn.relu,
                              input_shape=(state_shape + action_shape,)),
    ])
    dynamics_reward_head = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
    ])
    dynamics_state_head = tf.keras.Sequential([
        tf.keras.layers.Dense(state_shape, activation=tf.nn.relu),
    ])
    dynamics_reward_path = tf.keras.Sequential(
        [dynamics_trunk, dynamics_reward_head], name='dynamics_reward')
    dynamics_state_path = tf.keras.Sequential(
        [dynamics_trunk, dynamics_state_head], name='dynamics_state')

    def dynamics(state, action):
        action = tf.one_hot(action, action_shape)
        state_action = np.hstack((state, action))
        return (dynamics_reward_path(state_action), dynamics_state_path(state_action))

    prediction_trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(state_shape, activation=tf.nn.relu,
                              input_shape=(state_shape,)),
    ])
    prediction_policy_head = tf.keras.Sequential([
        tf.keras.layers.Dense(action_shape, activation=tf.nn.softmax)
    ])
    prediction_value_head = tf.keras.Sequential([
        tf.keras.layers.Dense(action_shape)
    ])
    prediction_policy_path = tf.keras.Sequential(
        [prediction_trunk, prediction_policy_head], name='prediction_policy')
    prediction_value_path = tf.keras.Sequential(
        [prediction_trunk, prediction_value_head], name='prediction_value')

    def prediction(state):
        return (prediction_policy_path(state), tf.reshape(prediction_value_path(state), [-1]))

    def action_sampler(policy):
        return tf.argmax(tf.random.categorical(policy, num_samples=1), axis=-1)

    return representation, dynamics, prediction, action_sampler


def main():
    env = gym.make('CartPole-v0')
    print('Observation space:', env.observation_space)
    print('Action space:', env.action_space)

    replay_buffer = ReplayBuffer(4096)

    representation, dynamics, prediction, action_sampler = define_model(env)
    muzero = MuZeroPSO(representation, dynamics, prediction)

    while True:
        obs_t = env.reset()
        while True:
            env.render()

            action = muzero.plan(obs_t, action_sampler)[0][0]

            obs_tp1, reward, done, _ = env.step(action.numpy())
            replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            if done:
                break
            obs_t = obs_tp1


if __name__ == '__main__':
    main()
