import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.7, beta=0.7):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self._storage = np.zeros([size], dtype=object)
        self._priorities = np.zeros([size])
        self._write_head = 0

    def add(self, priority, transition):
        self._priorities[self._write_head] = priority
        self._storage[self._write_head] = transition
        self._write_head = (self._write_head + 1) % self.size

    def update(self, index, priority):
        self._priorities[index] = priority

    def sample(self, batch_size, rollout_size):
        batch = []
        scaled_priorities = np.power(self._priorities, self.alpha)
        probabilities = scaled_priorities / np.sum(scaled_priorities)

        def unroll(index):
            unrolled = np.zeros([rollout_size], dtype=object)
            for offset in range(rollout_size):
                storage_index = (index + offset) % self.size
                # The write head breaks experience chains.
                if storage_index == self._write_head:
                    return None
                # Don't allow done = True in the middle of a rollout.
                if offset < rollout_size - 1 and self._storage[storage_index][-1]:
                    return None
                unrolled[offset] = self._storage[storage_index]
            return unrolled

        while len(batch) < batch_size:
            indices = np.random.choice(
                self.size, batch_size - len(batch), p=probabilities)

            for index in indices:
                unrolled = unroll(index)
                if unrolled is not None:
                    probability = probabilities[index]
                    weight = ((1.0 / batch_size) *
                              (1.0 / probability)) ** self.beta
                    batch.append((index, probability, weight, unrolled))

        return batch
