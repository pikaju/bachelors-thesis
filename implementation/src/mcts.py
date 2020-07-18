import numpy as np
import math


class Node:
    def __init__(self, state, policy, num_actions):
        self.parent = None
        self.parent_action = None
        self._min_q = 0.0
        self._max_q = 0.0

        self.state = state
        self.policy = policy
        self.num_actions = num_actions

        self.visit_count = np.zeros([num_actions], int)
        self.q = np.zeros([num_actions])
        self.reward = np.zeros([num_actions])
        self.children = [None for _ in range(num_actions)]

    def puct(self, action, c1, c2):
        # Get minimum and maximum Q values in the tree.
        min_q, max_q = self.min_q, self.max_q
        # Prevent division by zero.
        if min_q == 0.0 and max_q == 0.0:
            max_q = 1.0
        # Normalize Q value.
        normalized_q = (self.q[action] - min_q) / (max_q - min_q)
        tvc = np.sum(self.visit_count)
        rvc = math.sqrt(tvc) / (1 + self.visit_count[action])
        il = math.log((tvc + c2 + 1) / c2)
        return normalized_q + self.policy[action] * rvc * (c1 + il)

    def expand(self, action, reward, state, policy, num_actions):
        self.reward[action] = reward
        self.children[action] = Node(state, policy, num_actions)
        self.children[action].parent = self
        self.children[action].parent_action = action

    def backup(self, action, value, discount_factor):
        self.q[action] = (self.visit_count[action] *
                          self.q[action] + value) / (self.visit_count[action] + 1)
        self.visit_count[action] += 1
        self._propagate_q(self.q[action])

        if self.parent is not None:
            self.parent.backup(self.parent_action, self.reward[action] +
                               discount_factor * value, discount_factor)

    @property
    def min_q(self):
        return self._min_q if self.parent is None else self.parent.min_q

    @property
    def max_q(self):
        return self._max_q if self.parent is None else self.parent.max_q

    def _propagate_q(self, q):
        self._min_q = min(self._min_q, q)
        self._max_q = max(self._max_q, q)
        if self.parent is not None:
            self.parent._propagate_q(q)
