import tensorflow as tf
import numpy as np
import gym

from gym import envs

from model import define_model, define_losses
from replay_buffer import ReplayBuffer
from muzero import MuZeroPSO

import random

from multiprocessing import Pool


def run_env(env_name,
            discount_factor=0.8,
            replay_buffer_size=1024,
            learning_rate=0.005,
            batch_size=512,
            max_epochs=250,
            render=True):
    writer = tf.summary.create_file_writer(
        'logs/run-{}-{}-{}-{}'.format(discount_factor, replay_buffer_size, learning_rate, batch_size))

    env = gym.make(env_name)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    representation, dynamics, prediction, action_sampler, variables = define_model(
        env)
    loss_r, loss_v, loss_p, regularization = define_losses(env, variables)
    muzero = MuZeroPSO(representation, dynamics, prediction)

    optimizer = tf.optimizers.Adam(learning_rate)

    epoch = 0
    while max_epochs is None or epoch < max_epochs:
        obs_t = env.reset()
        total_reward = 0
        while True:
            if render:
                env.render()

            action = muzero.plan(obs_t, action_sampler, discount_factor,
                                 num_particles=32, depth=4)[0][0].numpy()

            obs_tp1, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            if done:
                break
            obs_t = obs_tp1
        epoch += 1

        with writer.as_default():
            tf.summary.scalar('total_reward', total_reward, step=epoch)

        # Training phase
        for _ in range(16):
            batch = replay_buffer.sample(batch_size, 5)

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
                tf.summary.scalar('loss', loss, step=epoch)


def benchmark():
    pool = Pool(6)
    for learning_rate in [0.001, 0.005, 0.0005, 0.01]:
        for replay_buffer_size in [1024, 1400]:
            for discount_factor in [0.99, 0.8, 0.95]:
                for batch_size in [256, 512]:
                    pool.apply_async(run_env, kwds={
                        'env_name': 'CartPole-v1',
                        'learning_rate': learning_rate,
                        'replay_buffer_size': replay_buffer_size,
                        'discount_factor': discount_factor,
                        'batch_size': batch_size,
                        'render': False,
                    })

    import time
    while True:
        try:
            time.sleep(100)
        except KeyboardInterrupt:
            return


def test():
    run_env(
        env_name='CartPole-v1',
        learning_rate=0.005,
        replay_buffer_size=1024,
        discount_factor=0.99,
        batch_size=256,
        max_epochs=None,
    )


def main():
    test()


if __name__ == '__main__':
    main()
