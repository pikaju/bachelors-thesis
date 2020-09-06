import scenario.register

from muzero.train import train as train_muzero
from a2c.train import train as train_a2c


def main():
    train_muzero()


if __name__ == '__main__':
    main()
