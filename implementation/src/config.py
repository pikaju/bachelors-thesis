import tensorflow as tf


class TrainingConfig:
    def __init__(
        self,
        learning_rate=0.005,
        reward_learning_rate=1.0,
        value_learning_rate=1.0,
        policy_learning_rate=1.0,
        regularization_learning_rate=1.0,
        batch_size=256,
        iterations=16,
        unroll_steps=5,
    ):
        self.__dict__.update(locals())


class ReplayBufferConfig:
    def __init__(
        self,
        size=1024,
        alpha=1.0,
        beta=1.0,
    ):
        self.__dict__.update(locals())


class ModelConfig:
    def __init__(
        self,
        activation=tf.nn.relu,
        state_size=16,
    ):
        self.__dict__.update(locals())


class MuZeroConfig:
    def __init__(
        self,
        num_simulations=16,
        num_particles=32,
        particle_depth=4,
        dirichlet_alpha=1.0,
        dirichlet_x=0.75,
        puct_c1=1.25,
        puct_c2=5000,
        temperature=1.0,
    ):
        self.__dict__.update(locals())


class Config:
    def __init__(
        self,
        environment_name="",
        discount_factor=1.0,
        render=True,
        training=TrainingConfig(),
        replay_buffer=ReplayBufferConfig(),
        model=ModelConfig(),
        muzero=MuZeroConfig(),
    ):
        self.__dict__.update(locals())
