import tensorflow as tf
import numpy as np
import gym

from gym import envs

from replay_buffer import ReplayBuffer
from muzero import MuZeroPSO

import random


state_shape = 4


def define_representation(env):
    obs_shape = env.observation_space.shape
    representation = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation=tf.nn.tanh, input_shape=obs_shape),
        tf.keras.layers.Dense(state_shape, activation=tf.nn.tanh),
    ], name='representation')
    return representation, representation.trainable_variables


def define_model(env):
    representation, representation_variables = define_representation(env)

    action_shape = env.action_space.n

    dynamics_trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(state_shape, activation=tf.nn.tanh,
                              input_shape=(state_shape + action_shape,)),
    ])
    dynamics_reward_head = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
    ])
    dynamics_state_head = tf.keras.Sequential([
        tf.keras.layers.Dense(state_shape),
    ])
    dynamics_reward_path = tf.keras.Sequential(
        [dynamics_trunk, dynamics_reward_head], name='dynamics_reward')
    dynamics_state_path = tf.keras.Sequential(
        [dynamics_trunk, dynamics_state_head], name='dynamics_state')

    def dynamics(state, action):
        action = tf.one_hot(action, action_shape)
        state_action = tf.concat((state, action), axis=1)
        return (tf.reshape(dynamics_reward_path(state_action), [-1]), dynamics_state_path(state_action))

    prediction_trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(state_shape, activation=tf.nn.tanh,
                              input_shape=(state_shape,)),
    ])
    prediction_policy_head = tf.keras.Sequential([
        tf.keras.layers.Dense(action_shape)
    ])
    prediction_value_head = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])
    prediction_policy_path = tf.keras.Sequential(
        [prediction_trunk, prediction_policy_head], name='prediction_policy')
    prediction_value_path = tf.keras.Sequential(
        [prediction_trunk, prediction_value_head], name='prediction_value')

    def prediction(state):
        return (prediction_policy_path(state), tf.reshape(prediction_value_path(state), [-1]))

    def action_sampler(policy):
        return tf.reshape(tf.random.categorical(policy, num_samples=1), [-1])

    variables = [
        *representation_variables,
        *dynamics_reward_path.trainable_variables,
        *dynamics_state_path.trainable_variables,
        *prediction_policy_path.trainable_variables,
        *prediction_value_path.trainable_variables,
    ]

    return representation, dynamics, prediction, action_sampler, variables


def define_losses(variables):
    def loss_r(true, pred):
        return tf.losses.MSE(true, pred)

    def loss_v(true, pred):
        return tf.losses.MSE(true, pred)

    def loss_p(action, logits):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(action, logits)) * 0.01

    return loss_r, loss_v, loss_p


def main():
    env = gym.make('CartPole-v0')
    discount_factor = 0.8
    print('Observation space:', env.observation_space)
    print('Action space:', env.action_space)

    replay_buffer = ReplayBuffer(1024)

    representation, dynamics, prediction, action_sampler, variables = define_model(
        env)
    loss_r, loss_v, loss_p = define_losses(variables)
    muzero = MuZeroPSO(representation, dynamics, prediction)

    optimizer = tf.optimizers.Adam()

    attempt = 0
    while True:
        obs_t = env.reset()
        while True:
            env.render()

            action = muzero.plan(obs_t, action_sampler,
                                 num_particles=64, depth=2)[0][0].numpy()
            if attempt < 32:
                action = env.action_space.sample()

            obs_tp1, reward, done, _ = env.step(action)
            replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            if done:
                break
            obs_t = obs_tp1
        attempt += 1

        # Training phase
        losses = []
        batch = replay_buffer.sample(16, 3)
        for sample in batch:
            obs, actions, rewards, obs_tp1, dones = zip(*sample)
            with tf.GradientTape() as tape:
                loss = muzero.loss(obs, actions, rewards, obs_tp1,
                                   dones[-1], discount_factor, loss_r, loss_v, loss_p)
                losses.append(loss)
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
        print('Loss:', tf.reduce_mean(losses).numpy(),
              '(attempt:', str(attempt) + ')')
        print('Replay buffer size:', len(replay_buffer))


if __name__ == '__main__':
    main()
