import random
import time
from typing import Callable
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

import tensorflow as tf
import numpy as np
import gym

from gym import envs

from config import *
from replay_buffer import PrioritizedReplayBuffer
from model import Model
from muzero import MuZeroMCTS, MuZeroPSO


def run_env(config_generator: Callable[[int], Config]):
    config = config_generator(0)

    def replay_candidate_to_sample(replay_candidate):
        rc = replay_candidate
        sample = []
        bv = rc[-1][2] if rc[-1][-1] else rc[-1][4]
        for i in range(config.training.unroll_steps):
            z = 0.0
            for j in range(i, len(rc) - 1):
                z += config.discount_factor ** (j - i) * rc[j][2]
            z += config.discount_factor ** (len(rc) - i) * bv
            sample.append((rc[i][0], rc[i][1], rc[i][2], rc[i][3], z))
        priority = abs(rc[0][4] - sample[0][-1])
        return priority, sample

    writer = tf.summary.create_file_writer(config.summary_directory)
    env = gym.make(config.environment_name)

    replay_buffer = PrioritizedReplayBuffer(config.replay_buffer)
    model = Model(config.model, env.observation_space, env.action_space)
    muzero = MuZeroMCTS(model.representation, model.dynamics, model.prediction)

    optimizer = tf.optimizers.Adam(lambda: config.training.learning_rate)

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
            # if random.uniform(0, 1) < 0.2:
            #     action = env.action_space.sample()
            obs_tp1, reward, done, _ = env.step(action)
            total_reward += reward
            reward *= config.training.reward_factor

            replay_candidate.append((
                obs_t,
                action,
                reward,
                obs_tp1,
                value,
                done,
            ))

            while len(replay_candidate) >= config.training.n or (done and len(replay_candidate) >= config.training.unroll_steps):
                priority, sample = replay_candidate_to_sample(replay_candidate)
                replay_buffer.add(priority, sample)
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

            obses, actions, rewards, obses_tp1, zs = zip(
                *[zip(*entry[-1]) for entry in batch])
            obses = tf.constant(list(zip(*obses)), tf.float32)
            actions = tf.constant(list(zip(*actions)))
            rewards = tf.constant(list(zip(*rewards)), tf.float32)
            obses_tp1 = tf.constant(list(zip(*obses_tp1)), tf.float32)
            zs = tf.constant(list(zip(*zs)), tf.float32)

            importance_weights = tf.constant(
                [e[2] for e in batch], tf.float32)

            with tf.GradientTape() as tape:
                losses = muzero.loss(
                    obses,
                    actions,
                    rewards,
                    obses_tp1,
                    zs,
                    model.loss_reward,
                    model.loss_value,
                    model.loss_policy,
                    model.loss_state,
                    model.loss_regularization,
                )
                learning_rates = [
                    config.training.reward_learning_rate,
                    config.training.value_learning_rate,
                    config.training.policy_learning_rate,
                    config.training.state_learning_rate,
                    config.training.regularization_learning_rate,
                ]
                weighted_losses = []
                for loss, lr in zip(losses, learning_rates):
                    weighted_losses.append(tf.reduce_sum(
                        loss * importance_weights * lr))
                total_loss = tf.reduce_sum(weighted_losses)

            variables = model.trainable_variables
            gradients = tape.gradient(total_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            with writer.as_default():
                wl = weighted_losses
                tf.summary.scalar('loss/total', total_loss, step=episode)
                tf.summary.scalar('loss/reward', wl[0], step=episode)
                tf.summary.scalar('loss/value', wl[1], step=episode)
                tf.summary.scalar('loss/policy', wl[2], step=episode)
                tf.summary.scalar('loss/state', wl[3], step=episode)
                tf.summary.scalar('loss/regularization',
                                  wl[3], step=episode)

        episode += 1
        config = config_generator(episode)


def test():
    def generate_config(episode: int):
        return Config(
            summary_directory='./logs/with_0.4/test2',
            environment_name='CartPole-v1',
            discount_factor=0.97,
            render=False,
            training=TrainingConfig(
                reward_factor=1.0,
                learning_rate=0.002 * 0.99 ** episode,
                state_learning_rate=1.0,
                batch_size=512,
                iterations=32,
            ),
            replay_buffer=ReplayBufferConfig(
                size=8192,
            ),
            model=ModelConfig(
                state_size=16,
            ),
            muzero=MuZeroConfig(
                num_simulations=16,
                temperature=1.0 * 0.99 ** episode,
            ),
        )
    run_env(generate_config)


def main():
    test()


if __name__ == '__main__':
    main()
