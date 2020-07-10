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
        obses,
        rewards,
        zs,
        actions,
        loss_r,
        loss_v,
        loss_p,
        regularization,
    ):
        unroll_steps = obses.shape[0]
        batch_size = obses.shape[1]

        r_losses, v_losses, p_losses, reg_losses = [
            tf.zeros([batch_size]) for _ in range(4)]
        state = self.representation(obses[0])

        for i in tf.range(unroll_steps):
            # Get model predictions.
            policy, value = self.prediction(state)
            reward, state = self.dynamics(scale_gradient(state), actions[i])

            r_losses += loss_r(rewards[i], reward) / unroll_steps
            v_losses += loss_v(zs[i], value) / unroll_steps
            p_losses += loss_p(actions[i], policy) / unroll_steps
            reg_losses += tf.repeat(regularization(),
                                    repeats=[batch_size]) / unroll_steps

        return [r_losses, v_losses, p_losses, reg_losses]


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
                # Maximize pUCT value based on action.
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
        search_value = tf.reduce_sum(
            (root.q * discount_factor + root.reward) * search_policy)
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

        policy, value = self.prediction(state)
        action = first_action = action_sampler(policy)
        reward, state = self.dynamics(state, action)
        total_reward = reward

        for i in tf.range(1, config.particle_depth, dtype=tf.float32):
            policy, value = self.prediction(state)
            action = action_sampler(policy)
            reward, state = self.dynamics(state, action)
            discounted_reward = reward * (discount_factor ** i)
            total_reward += discounted_reward

        _, final_value = self.prediction(state)
        value = final_value * (discount_factor **
                               config.particle_depth) + total_reward

        best_index = tf.argmax(value)
        return first_action[best_index], value[best_index]
