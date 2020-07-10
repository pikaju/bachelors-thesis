import numpy as np
import random

from config import ReplayBufferConfig
from sum_tree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self._tree = SumTree(config.size)
        self._current_size = 0

    def add(self, priority, transition):
        self._tree.add(priority ** self.config.alpha, transition)
        self._current_size += 1
        if self._current_size > self._tree.capacity:
            self._current_size = self._tree.capacity

    def update(self, index, priority):
        self._tree.update(index, priority ** self.config.alpha)

    def sample(self, batch_size):
        # If the replay buffer is empty, return an empty batch.
        if len(self) == 0:
            return []

        batch = []
        while len(batch) < batch_size:
            pointer = random.uniform(0, self._tree.total())
            index, probability, data = self._tree.get(pointer)
            weight = ((1.0 / batch_size) *
                      (1.0 / probability)) ** self.config.beta
            batch.append((index, probability, weight, data))
        return batch

    def __len__(self):
        return self._current_size
