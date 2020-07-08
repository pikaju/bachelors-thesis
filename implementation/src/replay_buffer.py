import numpy as np
import random
from sum_tree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=1.0, beta=1.0):
        self._tree = SumTree(size)
        self.alpha = alpha
        self.beta = beta

    def add(self, priority, transition):
        self._tree.add(priority ** self.alpha, transition)

    def update(self, index, priority):
        self._tree.update(index, priority ** self.alpha)

    def sample(self, batch_size, rollout_size):
        batch = []

        while len(batch) < batch_size:
            pointer = random.uniform(0, self._tree.total())
            index, probability, data = self._tree.get(pointer)
            if isinstance(data, int):
                print('Replay sampling failed, skipping.')
                continue
            weight = ((1.0 / batch_size) *
                      (1.0 / probability)) ** self.beta
            batch.append((index, probability, weight, data))

        return batch
