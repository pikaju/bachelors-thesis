import gym
import tensorflow as tf
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from a2c.model import Model, ModelConfig


def train():
    def env_fn(): return gym.make('CartPole-v1')
    env = env_fn()
    model = Model(
        config=ModelConfig(),
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    env = SubprocVecEnv([env_fn for _ in range(8)])

    tmax = 5

    optimizer = tf.optimizers.Adam(0.001)
    gamma = 0.95

    @tf.function
    def calculate_losses(obs, action, value):
        policy_pred, value_pred = model.forward(obs)
        advantage = tf.stop_gradient(value - value_pred)
        return (
            tf.reduce_mean(model.loss_policy(action, policy_pred) * advantage),
            tf.reduce_mean(model.loss_value(value, value_pred)),
            tf.reduce_mean(model.loss_entropy(policy_pred)),
        )

    def candidate_to_batch(replay_candidate):
        obses = []
        actions = []
        values = []

        _, bootstrapped_value = model.forward(tf.constant(replay_candidate[-1][3]))

        value = bootstrapped_value
        for obs_t, action, reward, _, done in reversed(replay_candidate):
            value = value * gamma * (1.0 - tf.cast(done, tf.float32)) + reward
            obses.append(obs_t)
            actions.append(action)
            values.append(value)

        return [tf.concat(l, 0) for l in [obses, actions, values]]

    episode = 0
    while True:
        obs_t = env.reset()
        replay_candidate = []
        step = 0
        total_reward = np.zeros([env.num_envs])
        while True:
            # env.render()

            policy, value = model.forward(tf.constant(obs_t))
            action = model.action_sampler(policy).numpy()
            obs_tp1, reward, done, _ = env.step(action)

            replay_candidate.append((obs_t, action, reward, obs_tp1, done))

            if len(replay_candidate) >= tmax:
                obs, action, value = candidate_to_batch(replay_candidate)
                replay_candidate.clear()

                with tf.GradientTape() as tape:
                    loss_policy, loss_value, loss_entropy = calculate_losses(obs, action, value)
                    total_loss = loss_policy + loss_value * 0.5 + loss_entropy * 0.01

                variables = model.trainable_variables
                gradients = tape.gradient(total_loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))

            total_reward += reward
            for i in range(env.num_envs):
                if done[i]:
                    print('Total reward:', total_reward[i])
                    total_reward[i] = 0.0

            obs_t = obs_tp1
