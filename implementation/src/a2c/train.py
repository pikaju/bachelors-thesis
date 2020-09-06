import gym
import tensorflow as tf
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

    env = SubprocVecEnv([env_fn for _ in range(4)])

    tmax = 5

    optimizer = tf.optimizers.Adam(0.001)
    gamma = 0.95

    @tf.function
    def calculate_losses(obses_t, actions, rewards, obses_tp1, dones):
        loss_policy = tf.zeros([1])
        loss_value = tf.zeros([1])
        loss_entropy = tf.zeros([1])
        last_obs_tp1 = obses_tp1[-1]
        bootstrapped_value = tf.stop_gradient(model.forward(last_obs_tp1)[1])

        value_target = bootstrapped_value
        for i in tf.range(len(obses_t) - 1, -1, -1):
            value_target = tf.cast(dones[i], tf.float32) * value_target * gamma + rewards[i]
            policy_pred, value_pred = model.forward(obses_t[i])

            advantage = tf.stop_gradient(value_target - value_pred)

            loss_policy += tf.reduce_mean(advantage * model.loss_policy(actions[i], policy_pred))
            loss_value += tf.reduce_mean(model.loss_value(value_target, value_pred))
            dist = tf.nn.softmax(policy_pred)
            entropy = -tf.reduce_sum(tf.reduce_mean(dist) * tf.math.log(dist))
            loss_entropy += entropy
        return [loss / len(obses_t) for loss in [loss_policy, loss_value, loss_entropy]]

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
            print(value)

            replay_candidate.append((obs_t, action, reward, obs_tp1, done))

            if len(replay_candidate) >= tmax:
                obses_t = tf.constant([e[0] for e in replay_candidate], tf.float32)
                actions = tf.constant([e[1] for e in replay_candidate], tf.int32)
                rewards = tf.constant([e[2] for e in replay_candidate], tf.float32)
                obses_tp1 = tf.constant([e[3] for e in replay_candidate], tf.float32)
                dones = tf.constant([e[4] for e in replay_candidate], tf.bool)

                with tf.GradientTape() as tape:
                    loss_policy, loss_value, loss_entropy = calculate_losses(
                        obses_t=obses_t,
                        actions=actions,
                        rewards=rewards,
                        obses_tp1=obses_tp1,
                        dones=dones,
                    )
                    total_loss = loss_policy + loss_value * 0.5 + loss_entropy * 0.02

                variables = model.trainable_variables
                gradients = tape.gradient(total_loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))

                replay_candidate.clear()

            total_reward += reward
            if not (False in done):
                print('Total reward:', total_reward)
                break

            obs_t = obs_tp1
