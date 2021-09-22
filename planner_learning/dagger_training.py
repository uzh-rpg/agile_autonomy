#!/usr/bin/env python3

import argparse
import os
import sys
import time
import numpy as np
import rospy
import shutil
from PlannerLearning import PlannerLearning
from std_msgs.msg import Bool, Empty
from common import setup_sim, place_quad_at_start, MessageHandler
import json
import random

from config.settings import create_settings

MAX_TIME_EXP = 500  # in second, if it takes more the process needs to be killed


class Trainer():
    def __init__(self, settings):
        rospy.init_node('iterative_learning_node', anonymous=False)
        self.settings = settings
        np.random.seed(self.settings.random_seed)
        self.expert_done = False
        self.label_sub = rospy.Subscriber("/hummingbird/labelling_completed", Bool,
                                          self.callback_expert, queue_size=1)  # Expert is done, decide what to do.
        self.msg_handler = MessageHandler()

    def callback_expert(self, data):
        # Will receive a true bool when expert is done
        print("Expert done with data labelling")
        self.expert_done = data.data

    def start_experiment(self, rollout_idx):
        self.msg_handler.publish_reset()
        place_quad_at_start(self.msg_handler)
        print("Doing experiment {}".format(rollout_idx))
        # Save point_cloud
        if self.settings.execute_nw_predictions:
            self.msg_handler.publish_save_pc()
        else:
            # We use expert to collect data
            print("Using expert to collect data")
            msg = Bool()
            msg.data = True
            self.learner.expert_pub.publish(msg)

    def perform_training(self):
        self.learner = PlannerLearning.PlanLearning(
            self.settings, mode="iterative")
        rollout_idx = 0
        # Wipe out expert dir to avoid problems
        removable_rollout_folders = os.listdir(self.settings.expert_folder)
        if len(removable_rollout_folders) > 0:
            removable_rollout_folders = [os.path.join(self.settings.expert_folder, d) \
                                         for d in removable_rollout_folders]
            removable_rollout_folders = [d for d in removable_rollout_folders if os.path.isdir(d)]
            for d in removable_rollout_folders:
                string = "rm -rf {}".format(d)
                os.system(string)
        while rollout_idx < self.settings.max_rollouts:
            if len(os.listdir(self.settings.expert_folder)) > 0:
                rollout_dir = os.path.join(self.settings.expert_folder,
                                           sorted(os.listdir(self.settings.expert_folder))[-1])
                rm_string = "rm -rf {}".format(rollout_dir)
                os.system(rm_string)
            self.learner.maneuver_complete = False  # Just to be sure
            self.expert_done = False  # Re-init to be sure
            spacing = random.choice(self.settings.tree_spacings)
            # whatever is not in the environment will be ignored
            self.msg_handler.publish_tree_spacing(spacing)
            self.msg_handler.publish_obj_spacing(spacing)
            unity_start_pos = setup_sim(self.msg_handler, config=self.settings)
            self.start_experiment(rollout_idx)
            start = time.time()
            exp_failed = False
            while not self.learner.maneuver_complete:
                time.sleep(0.1)
                duration = time.time() - start
                if duration > MAX_TIME_EXP:
                    exp_failed = True
                    self.learner.publish_stop_recording_msg()
                    break
            if (exp_failed or self.learner.exp_failed):
                print("Current experiment failed, will not save data")
                if len(os.listdir(self.settings.expert_folder)) > 0:
                    rollout_dir = os.path.join(self.settings.expert_folder,
                                               sorted(os.listdir(self.settings.expert_folder))[-1])
                    rm_string = "rm -rf {}".format(rollout_dir)
                    os.system(rm_string)
            else:  # Experiment Worked: label it and save it
                # final logging if experiment worked
                metrics_experiment = self.learner.experiment_report()
                for name, value in metrics_experiment.items():
                    print("{} is {:.3f}".format(name, value))
                rollout_idx += 1
                # Wait for expert to be done labelling (block gazebo meanwhile)
                os.system("rosservice call /gazebo/pause_physics")
                # Send message to get expert running
                self.learner.run_mppi_expert()
                while not self.expert_done:
                    time.sleep(1)
                # Mv data to train folder the labelled data
                rollout_dir = os.path.join(self.settings.expert_folder,
                                           sorted(os.listdir(self.settings.expert_folder))[-1])
                move_string = "mv {} {}".format(
                    rollout_dir, self.settings.train_dir)
                os.system(move_string)
                if rollout_idx % self.settings.train_every_n_rollouts == 0:
                    self.learner.train()
                if rollout_idx % self.settings.increase_net_usage_every_n_rollouts == 0:
                    self.settings.fallback_radius_expert = \
                        np.minimum(
                            self.settings.fallback_radius_expert + 0.5, 50.0)
                    print("Setting threshold to {}".format(
                        self.settings.fallback_radius_expert))
                os.system("rosservice call /gazebo/unpause_physics")

    def perform_testing(self):
        self.learner = PlannerLearning.PlanLearning(
            self.settings, mode="testing")
        tree_spacings = self.settings.tree_spacings
        removable_rollout_folders = os.listdir(self.settings.expert_folder)
        if len(removable_rollout_folders) > 0:
            removable_rollout_folders = [os.path.join(self.settings.expert_folder, d) \
                                         for d in removable_rollout_folders]
            removable_rollout_folders = [d for d in removable_rollout_folders if os.path.isdir(d)]
            for d in removable_rollout_folders:
                string = "rm -rf {}".format(d)
                os.system(string)
        for spacing in tree_spacings:
            self.msg_handler.publish_tree_spacing(spacing)
            self.msg_handler.publish_obj_spacing(spacing)
            exp_log_dir = os.path.join(self.settings.log_dir, "tree_{}_obj_{}".format(spacing,spacing))
            os.makedirs(exp_log_dir)
            # Start Experiment
            rollout_idx = 0
            report_buffer = []
            while rollout_idx < self.settings.max_rollouts:
                self.learner.maneuver_complete = False  # Just to be sure
                unity_start_pos = setup_sim(self.msg_handler, config=self.settings)
                self.start_experiment(rollout_idx)
                output_file_buffer = os.path.join(exp_log_dir,
                                                "experiment_metrics.json")
                start = time.time()
                exp_failed = False
                self.expert_done = False  # Re-init to be sure
                while not self.learner.maneuver_complete:
                    time.sleep(0.1)
                    duration = time.time() - start
                    if duration > MAX_TIME_EXP:
                        print("Current experiment failed. Will try again")
                        exp_failed = True
                        break
                if ((not exp_failed) and (self.learner.planner_succed)):
                    # final logging
                    metrics_experiment = self.learner.experiment_report()
                    report_buffer.append(metrics_experiment)
                    print("------- {} Rollout ------------".format(rollout_idx+1))
                    for name, value in metrics_experiment.items():
                        print("{} is {:.3f}".format(name, value))
                    print("-------------------------------")
                    rollout_idx += 1
                    rollout_dir = os.path.join(self.settings.expert_folder,
                                                   sorted(os.listdir(self.settings.expert_folder))[-1])
                    # Wait one second to stop recording
                    time.sleep(1)
                    if self.settings.verbose:
                        # Mv data record to log folder
                        move_string = "mv {} {}".format(
                            rollout_dir, exp_log_dir)
                        os.system(move_string)
                    else:
                        print("Rollout dir is {}".format(rollout_dir))
                        shutil.rmtree(rollout_dir)
                    # Save latest version of report buffer
                    with open(output_file_buffer, 'w') as fout:
                        json.dump(report_buffer, fout)
                else:
                    # Wait one second to stop recording
                    time.sleep(1)
                    # remove folder
                    rollout_dir = os.path.join(self.settings.expert_folder,
                                               sorted(os.listdir(self.settings.expert_folder))[-1])
                    rm_string = "rm -rf {}".format(rollout_dir)
                    os.system(rm_string)

def main():
    parser = argparse.ArgumentParser(description='Train Planning network.')
    parser.add_argument('--settings_file',
                        help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='dagger')
    trainer = Trainer(settings)
    trainer.perform_training()


if __name__ == "__main__":
    main()
