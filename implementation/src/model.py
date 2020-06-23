import tensorflow as tf
import gym


state_shape = 8
activation = tf.nn.relu


def define_representation(env):
    obs_shape = env.observation_space.shape
    representation_path = tf.keras.Sequential([
        tf.keras.layers.Dense(
            8,
            activation=activation,
            input_shape=obs_shape,
            kernel_initializer='he_normal',
        ),
        tf.keras.layers.Dense(
            state_shape,
            activation=activation,
            kernel_initializer='he_normal',
        ),
    ], name='representation')

    def no_inf(x):
        return 1.0 if x > 10.0**20 else (-1.0 if x < -10.0**20 else x)
    high = env.observation_space.high
    low = env.observation_space.low
    high = tf.expand_dims(tf.cast([no_inf(x) for x in high], tf.float32), 0)
    low = tf.expand_dims(tf.cast([no_inf(x) for x in low], tf.float32), 0)

    @tf.function
    def representation(obs):
        if isinstance(env.observation_space, gym.spaces.Box):
            return representation_path((tf.cast(obs, tf.float32) - low) / (high - low) * 2.0 - 1.0)

    return representation, representation_path.trainable_variables


def define_dynamics(env):
    if isinstance(env.action_space, gym.spaces.Box):
        action_shape = env.action_space.shape[0]
    else:
        action_shape = env.action_space.n

    dynamics_trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(
            state_shape,
            activation=activation,
            input_shape=(state_shape + action_shape,),
            kernel_initializer='he_normal',
        ),
        tf.keras.layers.Dense(
            state_shape,
            activation=activation,
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
            state_shape,
            kernel_initializer='he_normal',
        ),
    ])
    dynamics_reward_path = tf.keras.Sequential(
        [dynamics_trunk, dynamics_reward_head],
        name='dynamics_reward'
    )
    dynamics_state_path = tf.keras.Sequential(
        [dynamics_trunk, dynamics_state_head],
        name='dynamics_state'
    )

    @tf.function
    def dynamics(state, action):
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = tf.one_hot(action, action_shape, axis=-1)
        state_action = tf.concat((state, action), axis=1)
        return (tf.reshape(dynamics_reward_path(state_action), [-1]), dynamics_state_path(state_action))

    return dynamics, [*dynamics_reward_path.trainable_variables, *dynamics_state_path.trainable_variables]


def define_prediction(env):
    if isinstance(env.action_space, gym.spaces.Box):
        action_shape = env.action_space.shape[0]
    else:
        action_shape = env.action_space.n

    prediction_trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(
            state_shape,
            activation=activation,
            input_shape=(state_shape,),
            kernel_initializer='he_normal',
        ),
        tf.keras.layers.Dense(
            action_shape * 2,
            activation=activation,
            kernel_initializer='he_normal',
        ),
    ])
    prediction_policy_head = tf.keras.Sequential([
        tf.keras.layers.Dense(
            action_shape,
            kernel_initializer='he_normal',
        )
    ])
    prediction_value_head = tf.keras.Sequential([
        tf.keras.layers.Dense(
            1,
            kernel_initializer='he_normal',
        )
    ])
    prediction_policy_path = tf.keras.Sequential(
        [prediction_trunk, prediction_policy_head],
        name='prediction_policy'
    )
    prediction_value_path = tf.keras.Sequential(
        [prediction_trunk, prediction_value_head],
        name='prediction_value'
    )

    @tf.function
    def prediction(state):
        return (prediction_policy_path(state), tf.reshape(prediction_value_path(state), [-1]))

    @tf.function
    def action_sampler(policy):
        if isinstance(env.action_space, gym.spaces.Discrete):
            return tf.reshape(tf.random.categorical(policy, num_samples=1), [-1])
        else:
            return tf.random.normal(policy.shape, mean=policy)

    return prediction, action_sampler, [*prediction_policy_path.trainable_variables, *prediction_value_path.trainable_variables]


def define_model(env):
    representation, representation_variables = define_representation(env)
    dynamics, dynamics_variables = define_dynamics(env)
    prediction, action_sampler, prediction_variables = define_prediction(env)

    variables = [
        *representation_variables,
        *dynamics_variables,
        *prediction_variables,
    ]

    return representation, dynamics, prediction, action_sampler, variables


def define_losses(env, variables):
    @tf.function
    def loss_r(true, pred):
        return tf.losses.MSE(true, pred)

    @tf.function
    def loss_v(true, pred):
        return tf.losses.MSE(true, pred)

    @tf.function
    def loss_p(action, policy):
        if isinstance(env.action_space, gym.spaces.Discrete):
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(action, policy))
        else:
            return tf.losses.MSE(action, policy)

    @tf.function
    def regularization():
        return tf.add_n([tf.nn.l2_loss(variable) for variable in variables]) * 0.001

    return loss_r, loss_v, loss_p, regularization
