import argparse

from config.settings import create_settings
from dagger_training import Trainer


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trajectory tracker.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='test')
    trainer = Trainer(settings)
    trainer.perform_testing()


if __name__ == "__main__":
    main()
