import tensorflow as tf
import numpy as np
import gym

from gym import envs

from config import *
from model import define_model, define_losses
from replay_buffer import PrioritizedReplayBuffer
from muzero import MuZeroMCTS, MuZeroPSO

import random
import time

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool


def run_env(config: Config):
    env_spec = gym.make(config.environment_name)
    replay_buffer = PrioritizedReplayBuffer(config.replay_buffer)
    (representation,
     dynamics,
     prediction,
     action_sampler,
     policy_to_probabilities,
     variables) = define_model(env_spec)

    loss_r, loss_v, loss_p, regularization = define_losses(
        env_spec,
        variables,
        config.training.reward_learning_rate,
        config.training.value_learning_rate,
        config.training.policy_learning_rate,
        config.training.regularization_learning_rate,
    )
    optimizer = tf.optimizers.Adam(config.training.learning_rate)

    muzero = MuZeroMCTS(representation, dynamics, prediction)

    env = gym.make(config.environment_name)
    while True:
        obs_t = env.reset()
        replay_candidate = []
        while True:
            if config.render:
                env.render()

            action, value = [x.numpy() for x in muzero.plan(
                obs=obs_t,
                num_actions=env_spec.action_space.n,
                policy_to_probabilities=policy_to_probabilities,
                # action_sampler=action_sampler,
                discount_factor=tf.constant(
                    config.discount_factor, tf.float32),
                # num_particles=tf.constant(num_particles, tf.int32),
                # depth=search_depth
            )]

            obs_tp1, reward, done, _ = env.step(action)

            replay_candidate.append(
                (128.0, (obs_t, value, action, reward, obs_tp1, done)))
            if len(replay_candidate) > config.training.unroll_steps:
                replay_buffer.add(replay_candidate[0][0], [
                    m[1] for m in replay_candidate])
                replay_candidate.pop(0)

            if done:
                break
            obs_t = obs_tp1

        # Training phase
        for _ in range(config.training.iterations):
            batch = replay_buffer.sample(config.training.batch_size)

            obs, values, actions, rewards, obs_tp1, dones = zip(
                *[zip(*entry[-1]) for entry in batch])
            obs = [tf.constant(x, tf.float32) for x in zip(*obs)]
            values = [tf.constant(x, tf.float32) for x in zip(*values)]
            actions = [tf.constant(x) for x in zip(*actions)]
            rewards = [tf.constant(x, tf.float32) for x in zip(*rewards)]
            obs_tp1 = [tf.constant(x, tf.float32) for x in zip(*obs_tp1)]
            dones = [tf.constant(x, tf.bool) for x in zip(*dones)]

            importance_weights = tf.constant(
                [e[2] for e in batch], tf.float32)

            with tf.GradientTape() as tape:
                losses, priorities = muzero.loss(
                    obs,
                    values,
                    actions,
                    rewards,
                    obs_tp1,
                    dones,
                    tf.constant(config.discount_factor, tf.float32),
                    loss_r,
                    loss_v,
                    loss_p,
                    regularization,
                )
                weighted_losses = [
                    tf.reduce_sum(loss * importance_weights) for loss in losses
                ]
                total_loss = tf.reduce_sum(weighted_losses)

            # Update replay buffer priorities
            for element, priority in zip(batch, priorities.numpy()):
                replay_buffer.update(element[0], priority)

            gradients = tape.gradient(total_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))


def test():
    config = Config(
        environment_name="CartPole-v1",
        discount_factor=0.95,
        render=True,
        training=TrainingConfig(
        ),
        replay_buffer=ReplayBufferConfig(
        ),
        muzero=MuZeroConfig(
        )
    )
    run_env(config)


def main():
    test()


if __name__ == '__main__':
    main()
