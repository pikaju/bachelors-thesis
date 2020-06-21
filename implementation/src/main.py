import tensorflow as tf
import numpy as np
import gym

from gym import envs

from model import define_model, define_losses
from replay_buffer import ReplayBuffer
from muzero import MuZeroPSO

import random


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
