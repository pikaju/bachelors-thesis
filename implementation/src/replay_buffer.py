import random


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self._storage = []

    def add(self, obs_t, action, reward, obs_tp1, done):
        self._storage.append((
            obs_t,
            action,
            reward,
            obs_tp1,
            done
        ))
        if len(self._storage) > self.size:
            self._storage.pop(0)

    def _sample_single(self, max_iterations):
        start = random.randrange(0, len(self._storage))
        stop = start + 1
        while not self._storage[stop - 1][-1] and stop < len(self._storage) and (stop - start) < max_iterations:
            stop += 1
        return self._storage[start:stop]

    def sample(self, batch_size, max_iterations):
        return [self._sample_single(max_iterations) for _ in range(batch_size)]

    def __len__(self):
        return len(self._storage)
