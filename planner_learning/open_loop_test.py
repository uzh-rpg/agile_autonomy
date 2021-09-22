#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append("./src/PlannerLearning/models")
import time

from plan_learner import PlanLearner

from config.settings import create_settings


def main():
    parser = argparse.ArgumentParser(description='Train Planning Network')
    parser.add_argument('--settings_file',
                        help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='openloop')

    learner = PlanLearner(settings=settings)
    learner.test()


if __name__ == "__main__":
    main()
