import tensorflow as tf
import gym

from config import ModelConfig


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

        self.representation_path = tf.keras.Sequential([
            tf.keras.layers.Dense(
                16,
                activation=config.activation,
                input_shape=observation_space.shape,
                kernel_initializer='he_normal',
            ),
            tf.keras.layers.Dense(
                config.state_size,
                kernel_initializer='he_normal',
            ),
        ], name='representation')

        dynamics_trunk = tf.keras.Sequential([
            tf.keras.layers.Dense(
                config.state_size,
                activation=config.activation,
                input_shape=[config.state_size + action_size],
                kernel_initializer='he_normal',
            ),
        ])
        dynamics_reward_head = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1,
                kernel_initializer='he_normal',
            ),
        ])
        dynamics_state_head = tf.keras.Sequential([
            tf.keras.layers.Dense(
                config.state_size,
                kernel_initializer='he_normal',
            ),
        ])
        self.dynamics_reward_path = tf.keras.Sequential(
            [dynamics_trunk, dynamics_reward_head], name='dynamics_reward')
        self.dynamics_state_path = tf.keras.Sequential(
            [dynamics_trunk, dynamics_state_head], name='dynamics_state')

        prediction_trunk = tf.keras.Sequential([
            tf.keras.layers.Dense(
                config.state_size,
                activation=config.activation,
                input_shape=[config.state_size],
                kernel_initializer='he_normal',
            ),
        ])
        prediction_policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(
                action_size,
                kernel_initializer='he_normal',
            )
        ])
        prediction_value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1,
                kernel_initializer='he_normal',
            )
        ])
        self.prediction_policy_path = tf.keras.Sequential(
            [prediction_trunk, prediction_policy_head], name='prediction_policy')
        self.prediction_value_path = tf.keras.Sequential(
            [prediction_trunk, prediction_value_head], name='prediction_value')

    @tf.function
    def _scale_state(self, state):
        state_min = tf.reduce_min(state, -1, True)
        state_max = tf.reduce_max(state, -1, True)
        return (state - state_min) / (state_max - state_min)

    @tf.function
    def _action_to_repr(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = tf.one_hot(action, self.action_space.n, 1.0, 0.0, axis=-1)
        return action

    @property
    def trainable_variables(self):
        return [
            *self.representation_path.trainable_variables,
            *self.dynamics_reward_path.trainable_variables,
            *self.dynamics_state_path.trainable_variables,
            *self.prediction_policy_path.trainable_variables,
            *self.prediction_value_path.trainable_variables,
        ]

    @ tf.function
    def representation(self, observation):
        raw_state = self.representation_path(observation)
        return self._scale_state(raw_state)

    @tf.function
    def dynamics(self, state, action):
        state_action = tf.concat((state, self._action_to_repr(action)), -1)
        reward = tf.reshape(self.dynamics_reward_path(state_action), [-1])
        state = self.dynamics_state_path(state_action)
        return (reward, self._scale_state(state))

    @tf.function
    def prediction(self, state):
        return (self.prediction_policy_path(state),
                tf.reshape(self.prediction_value_path(state), [-1]))

    @tf.function
    def action_sampler(self, policy):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return tf.reshape(tf.random.categorical(policy, num_samples=1), [-1])
        else:
            return tf.map_fn(lambda x: tf.random.normal([1], mean=x), policy)

    @tf.function
    def policy_to_probabilities(self, policy):
        assert isinstance(self.action_space, gym.spaces.Discrete)
        return tf.nn.softmax(policy)

    @tf.function
    def loss_reward(self, true, pred):
        true = tf.expand_dims(true, 1)
        pred = tf.expand_dims(pred, 1)
        return tf.losses.MSE(true, pred)

    @tf.function
    def loss_value(self, true, pred):
        true = tf.expand_dims(true, 1)
        pred = tf.expand_dims(pred, 1)
        return tf.losses.MSE(true, pred)

    @tf.function
    def loss_policy(self, true, pred):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return tf.losses.sparse_categorical_crossentropy(true, pred, from_logits=True)
        else:
            return tf.losses.MSE(true, pred)

    @tf.function
    def loss_regularization(self):
        variables = self.trainable_variables
        return tf.add_n([tf.nn.l2_loss(variable) for variable in variables])
