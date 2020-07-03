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
            reward_lr=1.0,
            value_lr=1.0,
            policy_lr=1.0,
            regularization_lr=0.001,
            epsilon=0.1,
            batch_size=512,
            max_epochs=250,
            render=True):
    writer = tf.summary.create_file_writer(
        'logs/run-{}df-{}rbs-{}lr-{}rlr-{}-vlr-{}plr-{}reglr-{}eps-{}bs'.format(
            discount_factor,
            replay_buffer_size,
            learning_rate,
            reward_lr,
            value_lr,
            policy_lr,
            regularization_lr,
            epsilon,
            batch_size
        )
    )

    env = gym.make(env_name)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    representation, dynamics, prediction, action_sampler, variables = define_model(
        env)

    loss_r, loss_v, loss_p, regularization = define_losses(
        env,
        variables,
        reward_lr,
        value_lr,
        policy_lr,
        regularization_lr,
    )

    muzero = MuZeroPSO(representation, dynamics, prediction)

    optimizer = tf.optimizers.Adam(learning_rate)

    epoch = 0
    while max_epochs is None or epoch < max_epochs:
        obs_t = env.reset()
        total_reward = 0
        while True:
            if render:
                env.render()

            if random.uniform(0, 1) > epsilon:
                action, _ = [x.numpy() for x in muzero.plan(
                    obs=obs_t,
                    action_sampler=action_sampler,
                    discount_factor=tf.constant(discount_factor, tf.float32),
                    num_particles=tf.constant(64, tf.int32),
                    depth=4
                )]
            else:
                action = env.action_space.sample()

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
            obs = [tf.constant(x, tf.float32) for x in zip(*obs)]
            actions = [tf.constant(x) for x in zip(*actions)]
            rewards = [tf.constant(x, tf.float32) for x in zip(*rewards)]
            obs_tp1 = [tf.constant(x, tf.float32) for x in zip(*obs_tp1)]
            dones = [tf.constant(x, tf.bool) for x in zip(*dones)]

            with tf.GradientTape() as tape:
                losses = muzero.loss(
                    obs,
                    actions,
                    rewards,
                    obs_tp1,
                    dones,
                    tf.constant(discount_factor, tf.float32),
                    loss_r,
                    loss_v,
                    loss_p,
                    regularization,
                )
                loss = tf.reduce_mean(losses)
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            with writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch)
                tf.summary.scalar('loss_r', losses[0], step=epoch)
                tf.summary.scalar('loss_v', losses[1], step=epoch)
                tf.summary.scalar('loss_p', losses[2], step=epoch)
                tf.summary.scalar('loss_reg', losses[3], step=epoch)


def benchmark():
    pool = Pool(16)
    for batch_size in [512]:
        for discount_factor in [0.99, 0.995, 0.999]:
            for replay_buffer_size in [8192, 4096, 2048]:
                for learning_rate in [0.001, 0.005, 0.0005]:
                    for epsilon in [0.1]:
                        pool.apply_async(run_env, kwds={
                            'env_name': 'LunarLander-v2',
                            'learning_rate': learning_rate,
                            'epsilon': epsilon,
                            'replay_buffer_size': replay_buffer_size,
                            'discount_factor': discount_factor,
                            'batch_size': batch_size,
                            'render': False,
                            'max_epochs': None,
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
        epsilon=0.05,
        replay_buffer_size=1024,
        discount_factor=0.95,
        batch_size=512,
        max_epochs=None,
    )


def main():
    test()


if __name__ == '__main__':
    main()
