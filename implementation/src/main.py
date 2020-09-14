import os

import ray

from muzero import MuZero


def config_generator():
    for run in range(256):
        for param in [0.0, 0.5, 1.0]:
            config = {
                "reconstruction_loss_weight": param,
                "results_path": os.path.join("logs", "lr{}".format(param), "run{}".format(run)),
            }
            yield config


def main():
    cg = config_generator()
    running_experiments = []
    num_parallel = 3
    try:
        while True:
            for i in range(len(running_experiments)):
                experiment = running_experiments[i]
                if experiment.shared_storage_worker and experiment.config.training_steps <= ray.get(
                    experiment.shared_storage_worker.get_info.remote("training_step")
                ):
                    experiment.terminate_workers()
                    del running_experiments[i]
                    i -= 1

            if len(running_experiments) < num_parallel:
                muzero = MuZero("cartpole", next(cg), num_parallel)
                muzero.train()
                # running_experiments.append(muzero)
    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero):
                experiment.terminate_workers()


if __name__ == "__main__":
    main()
