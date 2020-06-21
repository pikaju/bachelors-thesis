import tensorflow as tf
import numpy as np
import gym

from gym import envs

from replay_buffer import ReplayBuffer
from muzero import MuZeroPSO

import random


state_shape = 8
activation = tf.nn.relu


def define_representation(env):
    obs_shape = env.observation_space.shape
    representation = tf.keras.Sequential([
        tf.keras.layers.Dense(
            8, activation=activation, input_shape=obs_shape),
        tf.keras.layers.Dense(state_shape, activation=activation),
    ], name='representation')
    return representation, representation.trainable_variables


def define_dynamics(env):
    if isinstance(env.action_space, gym.spaces.Box):
        action_shape = env.action_space.shape[0]
    else:
        action_shape = env.action_space.n

    dynamics_trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(state_shape, activation=activation,
                              input_shape=(state_shape + action_shape,)),
        tf.keras.layers.Dense(state_shape, activation=activation),
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
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = tf.one_hot(action, action_shape, axis=-1)
        state_action = tf.concat((state, action), axis=1)
        return (tf.reshape(dynamics_reward_path(state_action), [-1]), dynamics_state_path(state_action))

    return dynamics, [*dynamics_reward_path.trainable_variables, *dynamics_state_path.trainable_variables]


def define_prediction(env):
    if isinstance(env.action_space, gym.spaces.Box):
        action_shape = env.action_space.shape[0]
    else:
        action_shape = env.action_space.n

    prediction_trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(state_shape, activation=activation,
                              input_shape=(state_shape,)),
        tf.keras.layers.Dense(action_shape * 2, activation=activation),
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
        if isinstance(env.action_space, gym.spaces.Discrete):
            return tf.reshape(tf.random.categorical(policy, num_samples=1), [-1])
        else:
            return tf.random.normal(policy.shape, mean=policy)

    return prediction, action_sampler, [*prediction_policy_path.trainable_variables, *prediction_value_path.trainable_variables]


def define_model(env):
    representation, representation_variables = define_representation(env)
    dynamics, dynamics_variables = define_dynamics(env)
    prediction, action_sampler, prediction_variables = define_prediction(env)

    variables = [
        *representation_variables,
        *dynamics_variables,
        *prediction_variables,
    ]

    return representation, dynamics, prediction, action_sampler, variables


def define_losses(env, variables):
    def loss_r(true, pred):
        return tf.losses.MSE(true, pred)

    def loss_v(true, pred):
        return tf.losses.MSE(true, pred)

    def loss_p(action, policy):
        if isinstance(env.action_space, gym.spaces.Discrete):
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(action, policy))
        else:
            return tf.losses.MSE(action, policy)

    def regularization():
        return tf.add_n([tf.nn.l2_loss(variable) for variable in variables]) * 0.001

    return loss_r, loss_v, loss_p, regularization


def main():
    writer = tf.summary.create_file_writer('logs/')

    env = gym.make('LunarLanderContinuous-v2')
    discount_factor = 0.8
    print('Observation space:', env.observation_space)
    print('Action space:', env.action_space)

    replay_buffer = ReplayBuffer(2048)

    representation, dynamics, prediction, action_sampler, variables = define_model(
        env)
    loss_r, loss_v, loss_p, regularization = define_losses(env, variables)
    muzero = MuZeroPSO(representation, dynamics, prediction)

    optimizer = tf.optimizers.Adam(0.003)

    attempt = 0
    while True:
        obs_t = env.reset()
        total_reward = 0
        while True:
            env.render()

            action = muzero.plan(obs_t, action_sampler, discount_factor,
                                 num_particles=32, depth=4)[0][0].numpy()

            obs_tp1, reward, done, _ = env.step(action)
            total_reward += reward
            reward /= 16.0
            replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            if done:
                break
            obs_t = obs_tp1
        attempt += 1

        with writer.as_default():
            tf.summary.scalar('total_reward', total_reward, step=attempt)

        # Training phase
        for _ in range(16):
            batch = replay_buffer.sample(128, 5)

            obs, actions, rewards, obs_tp1, dones = zip(
                *[zip(*entry) for entry in batch])
            obs = tf.constant(list(zip(*obs)))
            actions = tf.constant(list(zip(*actions)))
            rewards = tf.constant(list(zip(*rewards)), dtype=tf.float32)
            obs_tp1 = tf.constant(list(zip(*obs_tp1)))
            dones = tf.constant(list(zip(*dones)))

            with tf.GradientTape() as tape:
                loss = muzero.loss(obs, actions, rewards, obs_tp1,
                                   dones, discount_factor, loss_r, loss_v, loss_p, regularization)
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            with writer.as_default():
                tf.summary.scalar('loss', loss, step=attempt)
                print('Loss:', loss.numpy(),
                      '(attempt:', str(attempt) + ')')

        with writer.as_default():
            tf.summary.scalar('replay buffer size', len(
                replay_buffer), step=attempt)
            print('Replay buffer size:', len(replay_buffer))


if __name__ == '__main__':
    main()
