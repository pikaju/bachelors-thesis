import numpy as np
import random
from sum_tree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.7, beta=0.7):
        self._tree = SumTree(size)
        self.alpha = alpha
        self.beta = beta

    def add(self, priority, transition):
        self._tree.add(priority ** self.alpha, transition)

    def update(self, index, priority):
        self._tree.update(index, priority ** self.alpha)

    def sample(self, batch_size, rollout_size):
        batch = []

        def unroll(index):
            unrolled = np.zeros([rollout_size], dtype=object)
            data_index = index - self._tree.capacity + 1

            for offset in range(rollout_size):
                # The write head breaks experience chains.
                if data_index == self._tree.write:
                    return None
                data = self._tree.data[data_index]
                # Don't allow done = True in the middle of a rollout.
                if offset < rollout_size - 1 and data[-1]:
                    return None
                unrolled[offset] = data
                data_index = (data_index + 1) % self._tree.capacity
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
