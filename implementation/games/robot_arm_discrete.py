import os
import datetime

import torch
import numpy as np

from muzero.games.abstract_game import AbstractGame
from scenario.scenario import Scenario


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        # Fix the maximum number of GPUs to use. By default muzero uses every GPUs available
        self.max_num_gpus = None

        # Game
        # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (1, 1, 8 * 3 + 4 + 1)
        # Fixed list of all possible actions. You should only edit the length
        self.action_space = list(range(10))
        # List of players. You should only edit the length
        self.players = list(range(1))
        # Number of previous observations and previous actions to add to the current observation
        self.stacked_observations = 0

        # Evaluate
        # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.muzero_player = 0
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        # Self-Play
        # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_workers = 8
        self.selfplay_on_gpu = False
        self.max_moves = 256  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.99  # Chronological discount of the reward
        # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size
        self.support_size = 10

        # Residual Network
        # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.downsample = False
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = []
        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = []
        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = []

        # Fully Connected Network
        self.encoding_size = 10
        # Define the hidden layers in the representation network
        self.fc_representation_layers = []
        # Define the hidden layers in the dynamics network
        self.fc_dynamics_layers = [64]
        # Define the hidden layers in the reward network
        self.fc_reward_layers = [64]
        # Define the hidden layers in the value network
        self.fc_value_layers = [64]
        # Define the hidden layers in the policy network
        self.fc_policy_layers = [64]

        # Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[
                                         :-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_weights = True  # Save the weights in results_path as model.weights
        # Total number of training steps (ie weights update according to a batch)
        self.training_steps = 200000
        self.batch_size = 64  # Number of parts of games to train on at each training step
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = 10
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 1
        self.train_on_gpu = True if torch.cuda.is_available(
        ) else False  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        # Replay Buffer
        # Number of self-play games to keep in the replay buffer
        self.replay_buffer_size = 2000
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 30
        # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER = True
        # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = 0.5

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = False
        self.reanalyse_on_gpu = False

        # Best known ratio for deterministic version: 0.8 --> 0.4 in 250 self played game (self_play_delay = 25 on GTX 1050Ti Max-Q).
        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 0.35


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Scenario()
        self._suction_cup_state = False

    def step(self, action):
        mapped_action = None
        if action == 8:
            self._suction_cup_state = True
            mapped_action = [*[0.0 for _ in range(4)], 1.0]
        elif action == 9:
            self._suction_cup_state = False
            mapped_action = [*[0.0 for _ in range(4)], -1.0]
        else:
            mapped_action = [*[0.0 for _ in range(4)],
                             1.0 if self._suction_cup_state else -1.0]
            mapped_action[action // 2] = 1.0 if action % 2 == 0 else -1.0

        observation, reward, done = self.env.step(mapped_action)
        return np.array([[[*observation, 1.0 if self._suction_cup_state else 0.0]]]), reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(10))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        self._suction_cup_state = False
        return np.array([[[*self.env.reset(), 0.0]]])

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        # self.env.render()
        # input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return str(action_number)
