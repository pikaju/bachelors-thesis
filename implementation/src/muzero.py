import tensorflow as tf


class MuZeroBase:
    def __init__(self, representation, dynamics, prediction):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction

    def predict(self, obs, actions):
        state = self.representation(obs)
        for action in actions:
            reward, state = self.dynamics(state, action)
        policy, value = self.prediction(state)
        return policy, value, reward

    def loss(self, obs_t, actions, rewards, obs_tp1, dones, discount, loss_r, loss_v, loss_p, regularization):
        losses = []
        state = self.representation(obs_t[0])

        bootstrapped_value = (1 - tf.cast(dones[-1], tf.float32)) * tf.stop_gradient(self.prediction(
            self.representation(obs_tp1[-1]))[-1])
        z = []
        for i in range(len(rewards)):
            z_i = 0
            for j in range(i, len(rewards)):
                z_i += discount ** (j - i) * rewards[j]
            z_i += discount ** (len(rewards) - i) * bootstrapped_value
            z.append(z_i)

        for _, action, true_reward, _, z_k in zip(obs_t, actions, rewards, obs_tp1, z):
            policy, value = self.prediction(state)
            reward, state = self.dynamics(state, action)

            losses.append(loss_r(true_reward, reward) +
                          loss_v(z_k, value) + loss_p(action, policy) + regularization())

        return tf.reduce_mean(losses)


class MuZeroPSO(MuZeroBase):
    def __init__(self, representation, dynamics, prediction):
        super().__init__(representation, dynamics, prediction)

    def plan(self, obs, action_sampler, num_particles=4, depth=4):
        obs = tf.expand_dims(obs, 0)
        initial_state = self.representation(obs)
        state = tf.repeat(initial_state, repeats=[num_particles], axis=0)

        actions = []
        total_reward = 0
        for _ in range(depth):
            policy, value = self.prediction(state)
            action = action_sampler(policy)
            actions.append(action)
            reward, state = self.dynamics(state, action)
            total_reward += reward

        _, value = self.prediction(state)
        best_index = tf.argmax(total_reward + value)
        import numpy as np
        print(np.array([action[best_index] for action in actions]))

        return [action[best_index] for action in actions], value[best_index]
