import tensorflow as tf
import numpy as np


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

    def loss(self, obs_t, actions, rewards, obs_tp1, done, discount, loss_r, loss_v, loss_p):
        loss = 0
        state = self.representation(obs_t[0])

        _, bootstrapped_value = self.prediction(
            self.representation(obs_tp1[-1]))
        z = []
        for i in range(len(rewards)):
            z_i = 0
            for j in range(i, len(rewards)):
                z_i += discount ** j * rewards[i]
            z_i += discount ** len(rewards) * bootstrapped_value * (1 - done)
            z.append(z_i)

        for _, action, true_reward, _, z_k in zip(obs_t, actions, rewards, obs_tp1, z):
            reward, state = self.dynamics(state, action)
            policy, value = self.prediction(state)

            loss += loss_r(true_reward, reward) + \
                loss_v(z_k, value) + loss_p(action, policy)


class MuZeroPSO(MuZeroBase):
    def __init__(self, representation, dynamics, prediction):
        super().__init__(representation, dynamics, prediction)

    def plan(self, obs, action_sampler, num_particles=4, depth=4):
        obs = np.array([obs])
        initial_state = self.representation(obs)
        best_action_sequence, best_value = None, None
        for _ in range(num_particles):
            state = initial_state
            action_sequence = []
            for _ in range(depth):
                policy, value = self.prediction(state)
                action = action_sampler(policy)
                action_sequence.append(action[0])
                _, state = self.dynamics(state, action)
            if best_value is None or best_value < value[0]:
                best_action_sequence = action_sequence
                best_value = value[0]
        return best_action_sequence, best_value
