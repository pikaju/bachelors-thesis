import gym
import tensorflow as tf
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from a2c.model import Model, ModelConfig


def train():
    def env_fn(): return gym.make('BachelorThesis-v0')
    env = env_fn()
    model = Model(
        config=ModelConfig(),
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    env = SubprocVecEnv([env_fn for _ in range(4)])

    tmax = 5

    optimizer = tf.optimizers.Adam(0.002)
    gamma = 0.95

    episode = 0
    while True:
        obs_t = env.reset()
        replay_candidate = []
        step = 0
        total_reward = 0.0
        while True:
            env.render()

            policy, value = model.forward(tf.constant(obs_t))
            action = model.action_sampler(policy).numpy()
            obs_tp1, reward, done, _ = env.step(action)

            replay_candidate.append((obs_t, action, reward, obs_tp1, done))

            if not False in done or len(replay_candidate) >= tmax:
                with tf.GradientTape() as tape:
                    loss_policy = 0.0
                    loss_value = 0.0
                    loss_entropy = 0.0
                    done = replay_candidate[-1][-1]
                    last_obs_tp1 = replay_candidate[-1][3]
                    bootstrapped_value = tf.cast(done, tf.float32) * tf.stop_gradient(
                        model.forward(tf.constant(last_obs_tp1))[1])

                    value_target = bootstrapped_value
                    for obs, action, reward, obs_tp1, done in reversed(replay_candidate):
                        value_target = value_target * gamma + reward
                        policy_pred, value_pred = model.forward(
                            tf.constant(obs))

                        advantage = tf.stop_gradient(value_target - value_pred)

                        loss_policy += advantage * model.loss_policy(
                            tf.constant(action), policy_pred)
                        loss_value += model.loss_value(
                            value_target, value_pred)
                        dist = tf.nn.softmax(policy)
                        entropy = - \
                            tf.reduce_sum(tf.reduce_mean(dist)
                                          * tf.math.log(dist))
                        loss_entropy += entropy
                    total_loss = loss_policy + loss_value * 0.5 + loss_entropy * 0.02
                    total_loss /= len(replay_candidate)

                variables = model.trainable_variables
                print(total_loss)
                gradients = tape.gradient(total_loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))

                replay_candidate.clear()

            total_reward += reward
            if not False in done:
                print('Total reward:', total_reward)
                break

            obs_t = obs_tp1
