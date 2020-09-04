import tensorflow as tf

from a2c.model import Model


class A2CAgent:
    def __init__(self, model: Model):
        self.model = model

    def act(self, obs):
        policy, _ = self.model.forward(obs)
        action = self.model.sample_action(policy)
        return action

    @tf.function
    def loss(
        self,
        obses,
        rewards,
        actions,
    ):
        unroll_steps = obses.shape[0]
        batch_size = obses.shape[1]

        bootstrapped_value = tf.zeroes([batch_size])

        for i in tf.range(0, unroll_steps, -1):
            bootstrapped_value += bootstrapped_value *
