import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.beta = 1.0
        self._storage = np.zeros([size], dtype=object)
        self._priorities = np.zeros([size])
        self._write_head = 0

    def add(self, priority, transition):
        self._priorities[self._write_head] = priority
        self._storage[self._write_head] = transition
        self._write_head = (self._write_head + 1) % self.size

    def update(self, index, priority):
        self._priorities[index] = priority

    def _try_sample_single(self, batch_size, unroll):
        probabilities = self._priorities / np.sum(self._priorities)
        index = np.random.choice(probabilities.shape[0], p=probabilities)

        unrolled = np.zeros([unroll], dtype=object)
        for offset in range(unroll):
            storage_index = (index + offset) % self.size
            # The write head breaks experience chains.
            if storage_index == self._write_head:
                return None
            # Don't allow done = True in the middle of a rollout.
            if offset < unroll - 1 and self._storage[storage_index][-1]:
                return None
            unrolled[offset] = self._storage[storage_index]

        probability = probabilities[index]
        weight = ((1.0 / batch_size) * (1.0 / probability)) ** self.beta
        return index, probability, weight, unrolled

    def _sample_single(self, batch_size, unroll):
        while True:
            sample = self._try_sample_single(batch_size, unroll)
            if sample != None:
                return sample

    def sample(self, batch_size, unroll):
        return [self._sample_single(batch_size, unroll) for _ in range(batch_size)]
