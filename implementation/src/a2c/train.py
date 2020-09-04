from a2c.model import Model

episode = 0
while True:
    obs_t = env.reset()
    replay_candidate = []
    step = 0
    total_reward = 0.0
    while True:
        if config.render:
            env.render()


def train():
    model = Model()
