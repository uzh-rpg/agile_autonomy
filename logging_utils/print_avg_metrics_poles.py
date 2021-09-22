import numpy as np
import argparse
import os
import json


def main(parser):
    args = parser.parse_args()
    data_dir = args.data_dir
    log_file = os.path.join(data_dir, "experiment_metrics.json")

    with open(log_file) as f:
        # this is a list of dicts
        data = json.load(f)

    print("Number of loaded rollouts is {}".format(len(data)))
    metrics = data[0].keys()
    combined_dict = {}
    for name in metrics:
        combined_dict[name] = []
    for rollout_res in data:
        for key, value in rollout_res.items():
            combined_dict[key].append(value)
    # Print some metrics
    for key, result_list in combined_dict.items():
        print("Average {} is {}".format(key, np.mean(result_list)))

    # Print Success Rate
    print("Success Rate is {}".format(np.mean(np.array(combined_dict['number_crashes']) == 0)))

    # Poles business
    good_rollouts = np.array(combined_dict['number_crashes']) == 0
    distances = np.array(combined_dict['closest_distance'])[good_rollouts]
    print("Avg. closest distances from poles when success is {}".format(np.mean(distances)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logger')
    parser.add_argument('--data_dir',
                        help='Path to data', required=True)
    main(parser)





