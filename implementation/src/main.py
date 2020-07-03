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
            num_particles=32,
            search_depth=4,
            discount_factor=0.8,
            replay_buffer_size=1024,
            learning_rate=0.005,
            reward_lr=1.0,
            value_lr=1.0,
            policy_lr=1.0,
            regularization_lr=0.001,
            training_iterations=16,
            epsilon=0.1,
            batch_size=512,
            max_episodes=256,
            render=True):
    writer = tf.summary.create_file_writer(
        'logs/run-{}np-{}sd-{}df-{}rbs-{}lr-{}rlr-{}-vlr-{}plr-{}reglr-{}iter-{}eps-{}bs'.format(
            num_particles,
            search_depth,
            discount_factor,
            replay_buffer_size,
            learning_rate,
            reward_lr,
            value_lr,
            policy_lr,
            regularization_lr,
            training_iterations,
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

    episode = 0
    while max_episodes is None or episode < max_episodes:
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
                    num_particles=tf.constant(num_particles, tf.int32),
                    depth=search_depth
                )]
            else:
                action = env.action_space.sample()

            obs_tp1, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            if done:
                break
            obs_t = obs_tp1

        with writer.as_default():
            tf.summary.scalar('total_reward', total_reward, step=episode)

        # Training phase
        for _ in range(training_iterations):
            batch = replay_buffer.sample(batch_size, search_depth + 1)

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
                tf.summary.scalar('loss', loss, step=episode)
                tf.summary.scalar('loss_r', losses[0], step=episode)
                tf.summary.scalar('loss_v', losses[1], step=episode)
                tf.summary.scalar('loss_p', losses[2], step=episode)
                tf.summary.scalar('loss_reg', losses[3], step=episode)

        episode += 1


def benchmark():
    import time

    pool = Pool(6)
    tasks = []
    while True:
        for task in tasks:
            if task.ready():
                tasks.remove(task)

        if len(tasks) < 12:
            params = {
                'env_name': 'LunarLander-v2',
                'num_particles': random.randrange(8, 128),
                'search_depth': random.randrange(1, 16),
                'learning_rate': random.uniform(0.03, 0.0001),
                'reward_lr': random.uniform(10.0, 0.1),
                'value_lr': random.uniform(10.0, 0.1),
                'policy_lr': random.uniform(10.0, 0.1),
                'regularization_lr': random.uniform(10.0, 0.1),
                'training_iterations': random.randrange(8, 128),
                'replay_buffer_size': random.randrange(1024, 32000),
                'discount_factor': random.uniform(0.9, 0.9999),
                'batch_size': random.randrange(128, 512),
                'max_episodes': 256,
                'render': False,
            }
            tasks.append(pool.apply_async(run_env, kwds=params))

        try:
            time.sleep(0.1)
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
        max_episodes=None,
    )


def main():
    benchmark()


if __name__ == '__main__':
    main()
