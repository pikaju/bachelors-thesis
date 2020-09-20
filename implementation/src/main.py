import os

import ray

from muzero import MuZero


def config_generator():
    for run in range(256):
        for param in [0.0, 0.5, 1.0]:
            config = {
                "game_name": "cartpole",
                "reconstruction_loss_weight": 0,
                "consistency_loss_weight": 1.0,
                "results_path": os.path.join("logs", "lr{}".format(param), "run{}".format(run)),
            }
            yield config


def main():
    cg = config_generator()
    running_experiments = []
    num_parallel = 3
    try:
        for config in cg:
            muzero = MuZero(config["game_name"], config, num_parallel)
            muzero.train()
    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero):
                experiment.terminate_workers()


if __name__ == "__main__":
    main()
