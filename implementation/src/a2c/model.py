import tensorflow as tf
import gym


class ModelConfig:
    def __init__(
        self,
        activation=tf.nn.relu,
        hidden_shared_layers=[16],
        hidden_policy_layers=[16],
        hidden_value_layers=[8],
    ):
        self.__dict__.update(locals())


class Model:
    def __init__(
        self,
        config: ModelConfig,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        action_size = action_space.shape[0] if isinstance(
            action_space, gym.spaces.Box) else action_space.n

        assert isinstance(observation_space, gym.spaces.Box)

        self.shared_base = tf.keras.Sequential(name='shared_base')
        for i, units in enumerate(config.hidden_shared_layers):
            self.shared_base.add(tf.keras.layers.Dense(
                units,
                input_shape=None if i != 0 else observation_space.shape,
                activation=config.activation,
                kernel_initializer='he_normal',
            ))

        self.policy_head = tf.keras.Sequential([
            *[
                tf.keras.layers.Dense(
                    units,
                    activation=config.activation,
                    kernel_initializer='he_normal',
                ) for units in config.hidden_policy_layers
            ],
            tf.keras.layers.Dense(
                action_size,
                activation=config.activation,
                kernel_initializer='he_normal',
            ),
        ], name='policy_head')

        self.value_head = tf.keras.Sequential([
            *[
                tf.keras.layers.Dense(
                    units,
                    kernel_initializer='he_normal',
                ) for units in config.hidden_value_layers
            ],
            tf.keras.layers.Dense(
                1,
                kernel_initializer='he_normal',
            ),
        ], name='value_head')

    @property
    def trainable_variables(self):
        return [
            *self.shared_base.trainable_variables,
            *self.policy_head.trainable_variables,
            *self.value_head.trainable_variables,
        ]

    @tf.function
    def forward(self, obs):
        base_output = self.shared_base(obs)
        policy_output = self.policy_head(base_output)
        value_output = tf.reshape(self.value_head(base_output), [-1])
        return policy_output, value_output

    @tf.function
    def action_sampler(self, policy):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return tf.random.categorical(policy, num_samples=1)[:, 0]
        else:
            return tf.map_fn(lambda x: tf.random.normal([1], mean=x), policy)

    @tf.function
    def policy_to_probabilities(self, policy):
        assert isinstance(self.action_space, gym.spaces.Discrete)
        return tf.nn.softmax(policy)

    @tf.function
    def loss_policy(self, true, pred):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return tf.losses.sparse_categorical_crossentropy(true, pred, from_logits=True)
        else:
            return tf.losses.MSE(true, pred)

    @tf.function
    def loss_value(self, true, pred):
        return tf.losses.MSE(true, pred)

    @tf.function
    def loss_entropy(self, pred):
        dist = tf.nn.softmax(pred)
        return tf.reduce_sum(dist * tf.math.log(dist), 1)
