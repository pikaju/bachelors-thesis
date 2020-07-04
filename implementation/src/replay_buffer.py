import numpy as np
import random
from sum_tree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.7, beta=0.7):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self._tree = SumTree(self.size)

    def add(self, priority, transition):
        self._tree.add(priority ** self.alpha, transition)

    def update(self, index, priority):
        self._tree.update(index, priority ** self.alpha)

    def sample(self, batch_size, rollout_size):
        batch = []

        def unroll(index):
            unrolled = np.zeros([rollout_size], dtype=object)
            for offset in range(rollout_size):
                storage_index = (index + offset) % self.size
                # The write head breaks experience chains.
                if storage_index == self._tree.write:
                    return None
                if self._tree.data[storage_index] == 0:
                    return None
                # Don't allow done = True in the middle of a rollout.
                if offset < rollout_size - 1 and self._tree.data[storage_index][-1]:
                    return None
                unrolled[offset] = self._tree.data[storage_index]
            return unrolled

        while len(batch) < batch_size:
            pointer = random.uniform(0, self._tree.total())
            index, probability, _ = self._tree.get(pointer)
            unrolled = unroll(index)
            if unrolled is not None:
                weight = ((1.0 / batch_size) *
                          (1.0 / probability)) ** self.beta
                batch.append((index, probability, weight, unrolled))

        return batch
