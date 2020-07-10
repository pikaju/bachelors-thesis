import tensorflow as tf
import numpy as np
import gym

from gym import envs

from config import *
from replay_buffer import PrioritizedReplayBuffer
from model import Model
from muzero import MuZeroMCTS, MuZeroPSO

import random
import time

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool


def run_env(config: Config):
    writer = tf.summary.create_file_writer(config.summary_directory)
    env = gym.make(config.environment_name)

    replay_buffer = PrioritizedReplayBuffer(config.replay_buffer)
    model = Model(config.model, env.observation_space, env.action_space)
    muzero = MuZeroMCTS(model.representation, model.dynamics, model.prediction)

    optimizer = tf.optimizers.Adam(config.training.learning_rate)

    episode = 0
    while True:
        obs_t = env.reset()
        replay_candidate = []
        step = 0
        total_reward = 0.0
        while True:
            if config.render:
                env.render()
            action, value = [x.numpy() for x in muzero.plan(
                obs=obs_t,
                num_actions=env.action_space.n,
                policy_to_probabilities=model.policy_to_probabilities,
                # action_sampler=model.action_sampler,
                discount_factor=tf.constant(
                    config.discount_factor, tf.float32),
                config=config.muzero
            )]
            print(value)
            obs_tp1, reward, done, _ = env.step(action)
            total_reward += reward

            replay_candidate.append(
                (128.0, (obs_t, value, action, reward, done)))
            if len(replay_candidate) >= config.training.unroll_steps:
                replay_buffer.add(replay_candidate[0][0], [
                    m[1] for m in replay_candidate])
                replay_candidate.pop(0)

            step += 1
            if done:
                break
            obs_t = obs_tp1

        with writer.as_default():
            tf.summary.scalar('environment_steps', step, step=episode)
            tf.summary.scalar('reward/total', total_reward, step=episode)
            tf.summary.scalar('reward/mean', total_reward / step, step=episode)

        # Training phase
        for _ in range(config.training.iterations):
            batch = replay_buffer.sample(config.training.batch_size)

            obs, values, actions, rewards, dones = zip(
                *[zip(*entry[-1]) for entry in batch])
            obs = tf.constant(list(zip(*obs)), tf.float32)
            values = tf.constant(list(zip(*values)), tf.float32)
            actions = tf.constant(list(zip(*actions)))
            rewards = tf.constant(list(zip(*rewards)), tf.float32)
            dones = tf.constant(list(zip(*dones)), tf.bool)

            importance_weights = tf.constant(
                [e[2] for e in batch], tf.float32)

            with tf.GradientTape() as tape:
                losses, priorities = muzero.loss(
                    obs,
                    values,
                    actions,
                    rewards,
                    dones,
                    tf.constant(config.discount_factor, tf.float32),
                    model.loss_reward,
                    model.loss_value,
                    model.loss_policy,
                    model.loss_regularization,
                )
                learning_rates = [
                    config.training.reward_learning_rate,
                    config.training.value_learning_rate,
                    config.training.policy_learning_rate,
                    config.training.regularization_learning_rate,
                ]
                weighted_losses = []
                for loss, lr in zip(losses, learning_rates):
                    weighted_losses.append(tf.reduce_sum(
                        loss * importance_weights * lr))
                total_loss = tf.reduce_sum(weighted_losses)

            # Update replay buffer priorities
            for element, priority in zip(batch, priorities.numpy()):
                replay_buffer.update(element[0], priority)

            variables = model.trainable_variables
            gradients = tape.gradient(total_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            with writer.as_default():
                wl = weighted_losses
                tf.summary.scalar('loss/total', total_loss, step=episode)
                tf.summary.scalar('loss/reward', wl[0], step=episode)
                tf.summary.scalar('loss/value', wl[1], step=episode)
                tf.summary.scalar('loss/policy', wl[2], step=episode)
                tf.summary.scalar('loss/regularization',
                                  wl[3], step=episode)

        episode += 1


def test():
    config = Config(
        summary_directory='./logs/mcts',
        environment_name='CartPole-v1',
        discount_factor=0.95,
        render=True,
        training=TrainingConfig(
        ),
        replay_buffer=ReplayBufferConfig(
        ),
        model=ModelConfig(
        ),
        muzero=MuZeroConfig(
        ),
    )
    run_env(config)


def main():
    test()


if __name__ == '__main__':
    main()
