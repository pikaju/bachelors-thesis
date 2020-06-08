import random


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.episodes = []

    def begin_episode(self):
        print(len(self.episodes))
        self.current_episode = []
        self.episodes.append(self.current_episode)
        if len(self.episodes) > self.size:
            self.episodes = self.episodes[1:]

    def add(self, obs_t, action, reward, obs_tp1, done):
        self.current_episode.append((
            obs_t,
            action,
            reward,
            obs_tp1,
            done
        ))

    def _sample_from(self, episode, iterations):
        start = random.randrange(0, len(episode) - iterations + 1)
        return episode[start:start+iterations]

    def sample(self, batch_size, iterations):
        return [self._sample_from(random.choice(self.episodes), iterations) for _ in range(batch_size)]
