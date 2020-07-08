import tensorflow as tf
import numpy as np

from config import MuZeroConfig
import mcts


@tf.function
def scale_gradient(tensor):
    """Scales the gradient for the backward pass."""
    gradient_scale = 0.5
    return tensor * gradient_scale + tf.stop_gradient(tensor) * (1 - gradient_scale)


class MuZeroBase:
    def __init__(self, representation, dynamics, prediction):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction

    @tf.function
    def predict(self, obs, actions):
        state = self.representation(obs)
        for action in actions:
            reward, state = self.dynamics(state, action)
        policy, value = self.prediction(state)
        return policy, value, reward

    @tf.function
    def loss(
        self,
        obs_t,
        values,
        actions,
        rewards,
        obs_tp1,
        dones,
        discount,
        loss_r,
        loss_v,
        loss_p,
        regularization,
    ):
        rollout_size = len(obs_t)
        batch_size = obs_t[0].shape[0]

        r_losses, v_losses, p_losses, reg_losses = 0.0, 0.0, 0.0, 0.0
        state = self.representation(obs_t[0])

        bootstrapped_value = (1 - tf.cast(dones[-1], tf.float32)) * values[-1]

        z = []
        for i in range(rollout_size):
            z_i = 0
            for j in range(i, rollout_size):
                z_i += discount ** (j - i) * rewards[j]
            z_i += discount ** (rollout_size - i) * bootstrapped_value
            z.append(z_i)

        for _, action, true_reward, _, z_k in zip(obs_t, actions, rewards, obs_tp1, z):
            policy, value = self.prediction(state)
            reward, state = self.dynamics(scale_gradient(state), action)

            r_losses += loss_r(true_reward, reward) / rollout_size
            v_losses += loss_v(z_k, value) / rollout_size
            p_losses += loss_p(action, policy) / rollout_size
            reg_losses += tf.repeat(regularization(),
                                    repeats=[batch_size]) / rollout_size

        priorities = tf.abs(z[0] - values[0])
        return [r_losses, v_losses, p_losses, reg_losses], priorities


class MuZeroMCTS(MuZeroBase):
    def __init__(self, representation, dynamics, prediction):
        super().__init__(representation, dynamics, prediction)

    def plan(
        self,
        obs,
        num_actions,
        policy_to_probabilities,
        discount_factor,
        config: MuZeroConfig,
    ):
        obs = tf.expand_dims(obs, 0)
        initial_state = self.representation(obs)
        policy, _ = self.prediction(initial_state)
        probabilities = policy_to_probabilities(policy)[0].numpy()
        root = mcts.Node(initial_state, probabilities, num_actions)

        for _ in range(config.num_simulations):
            node = root
            while True:
                # Maximize pUCT
                ucbs = [node.puct(action, config.puct_c1, config.puct_c2)
                        for action in range(num_actions)]
                action = np.argmax(ucbs)

                next_node = node.children[action]
                if next_node is None:
                    reward, state = self.dynamics(
                        node.state, tf.expand_dims(action, 0))
                    policy, value = self.prediction(state)
                    probabilities = policy_to_probabilities(policy)[0].numpy()
                    node.expand(action, reward[0].numpy(),
                                state, probabilities, num_actions)
                    node.backup(action, value, discount_factor)
                    break
                node = next_node

        count = tf.expand_dims(root.visit_count, 0)
        powed_count = tf.math.pow(
            tf.cast(count, tf.float32), 1.0 / config.temperature)
        search_policy = powed_count / tf.reduce_sum(powed_count)
        search_action = tf.random.categorical(
            tf.math.log(search_policy), 1)[0][0]
        search_value = tf.constant(root.q[search_action.numpy()])
        return search_action, search_value


class MuZeroPSO(MuZeroBase):
    def __init__(self, representation, dynamics, prediction):
        super().__init__(representation, dynamics, prediction)

    @tf.function
    def plan(
        self,
        obs,
        action_sampler,
        discount_factor,
        config: MuZeroConfig,
    ):
        obs = tf.expand_dims(obs, 0)
        initial_state = self.representation(obs)
        state = tf.repeat(initial_state, repeats=[
                          config.num_particles], axis=0)

        actions = []
        total_reward = tf.zeros([config.num_particles])
        for i in range(config.particle_depth):
            policy, value = self.prediction(state)
            action = action_sampler(policy)
            actions.append(action)
            reward, state = self.dynamics(state, action)
            discounted_reward = reward * (discount_factor ** i)
            total_reward += discounted_reward

        _, final_value = self.prediction(state)
        value = final_value * (discount_factor **
                               config.particle_depth) + total_reward

        best_index = tf.argmax(value)

        return actions[0][best_index], value[best_index]
