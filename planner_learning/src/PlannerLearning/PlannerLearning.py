#!/usr/bin/env python3
import os
import random
import open3d as o3d
import tensorflow as tf
import copy

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_msgs.msg import Empty
import scipy

from .PlannerBase import PlanBase


class PlanLearning(PlanBase):
    def __init__(self, config, mode):
        self.recorded_samples = 0
        self.pcd = None
        self.pcd_tree = None
        self.pc_min = None
        self.pc_max = None
        self.counter = 1000
        self.crashed = False
        self.exp_failed = False
        self.planner_succed = True
        if config.track_global_traj:
            # Eliminate slow-down maneuver, gives ambigous labels.
            self.end_ref_percentage = 0.95
        else:
            self.end_ref_percentage = 0.8
        self.data_pub = rospy.Publisher("/hummingbird/agile_autonomy/start_flying", Bool,
                                        queue_size=1)  # Stop upon some condition
        self.planner_succed_sub = rospy.Subscriber("/test_primitive/completed_planning",
                                                  Bool, self.planner_succed_callback, queue_size=1)
        self.label_data_pub = rospy.Publisher("/hummingbird/start_label", Bool,
                                              queue_size=1)  # Stop upon some condition
        self.success_subs = rospy.Subscriber("success_reset", Empty,
                                             self.callback_success_reset,
                                             queue_size=1)
        self.reset_metrics()
        # Check at 20Hz the collision
        self.timer_check = rospy.Timer(
            rospy.Duration(1. / 20.),
            self.check_task_progress)

        super(PlanLearning, self).__init__(config, mode)

    def publish_stop_recording_msg(self):
        # Send message to cpp side to stop recording data
        self.maneuver_complete = True
        self.use_network = False
        print("Giving a stop from python")
        msg = Bool()
        msg.data = False
        self.data_pub.publish(msg)

    def run_mppi_expert(self):
        label_data_msg = Bool()
        label_data_msg.data = True
        self.label_data_pub.publish(label_data_msg)

    def planner_succed_callback(self, data):
        self.planner_succed = data.data

    def callback_fly(self, data):
        # If self.use_network is true, then trajectory is already loaded
        if data.data and (not self.use_network):
            # Load pointcloud and make kdtree out of it
            rollout_dir = os.path.join(self.config.expert_folder,
                                       sorted(os.listdir(self.config.expert_folder))[-1])
            pointcloud_fname = os.path.join(
                rollout_dir, "pointcloud-unity.ply")
            print("Reading pointcloud from %s" % pointcloud_fname)
            self.pcd = o3d.io.read_point_cloud(pointcloud_fname)
            self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
            # get dimensions prom point cloud

            self.pc_max = self.pcd.get_max_bound()
            self.pc_min = self.pcd.get_min_bound()
            print("min max pointcloud")
            print(self.pc_max)
            print(self.pc_min)

            # Load the reference trajectory
            if self.config.track_global_traj:
                traj_fname = os.path.join(rollout_dir, "ellipsoid_trajectory.csv")
            else:
                traj_fname = os.path.join(rollout_dir, "reference_trajectory.csv")
            print("Reading Trajectory from %s" % traj_fname)
            self.load_trajectory(traj_fname)
            self.reference_initialized = True
            # only enable network when KDTree and trajectory are ready

        # Might be here if you crash in less than a second.
        if self.maneuver_complete:
            return
        # If true, network should fly.
        # If false, maneuver is finished and network is off.
        self.use_network = data.data and self.config.execute_nw_predictions
        if (not data.data):
            self.maneuver_complete = True
            self.use_network = False

    def train(self):
        self.is_training = True
        self.learner.train()
        self.is_training = False
        self.use_network = False

    def experiment_report(self):
        metrics_report = super(PlanLearning, self).experiment_report()
        metrics_report.update(copy.deepcopy(self.metrics))
        return metrics_report

    def reset_metrics(self):
        self.metrics = {'number_crashes': 0,
                        'travelled_dist': 0,
                        'closest_distance': 1000}

    def callback_success_reset(self, data):
        print("Received call to Clear Buffer and Restart Experiment")
        os.system("rosservice call /gazebo/pause_physics")
        self.rollout_idx += 1
        self.use_network = False
        self.pcd = None
        self.reference_initialized = False
        self.maneuver_complete = False
        self.counter = 1000
        self.pcd_tree = None
        self.start_time = None
        # Learning phase to test
        tf.keras.backend.set_learning_phase(0)
        self.n_times_expert = 0.000
        self.n_times_net = 0.001
        self.crashed = False
        self.exp_failed = False
        self.planner_succed = True
        self.reset_queue()
        self.reset_metrics()
        print("Resetting experiment")
        os.system("rosservice call /gazebo/unpause_physics")
        print('Done Reset')

    def check_task_progress(self, _timer):
        # go here if there are problems with the generation of traj
        # No need to check anymore
        if self.maneuver_complete:
            return
        if not self.planner_succed:
            print("Stopping experiment because planner failed!")
            self.publish_stop_recording_msg()
            self.exp_failed = True
            return

        # check if pointcloud is ready
        if self.pcd is None or (self.pc_min is None):
            return
        # check if reference is ready
        if not self.reference_initialized:
            return

        if (self.reference_progress / (self.reference_len)) > self.end_ref_percentage:
            print("It worked well. (Arrived at %d / %d)" % (self.reference_progress, self.reference_len))
            self.publish_stop_recording_msg()

        # check if crashed into something
        quad_position = [self.odometry.pose.pose.position.x,
                         self.odometry.pose.pose.position.y,
                         self.odometry.pose.pose.position.z]

        # Check if crashed into ground or outside a box (check in z, x, y)
        if (quad_position[0] < self.pc_min[0]) or (quad_position[0] > self.pc_max[0]) or \
           (quad_position[1] < self.pc_min[1]) or (quad_position[1] > self.pc_max[1]) or \
           (quad_position[2] < self.pc_min[2]) or (quad_position[2] > self.pc_max[2]):
            print("Stopping experiment because quadrotor outside allowed range!")
            print(quad_position)
            self.publish_stop_recording_msg()
            return
        if self.reference_progress > 50: # first second used to warm up
            self.update_metrics(quad_position)

    def update_metrics(self, quad_position):
        # Meters until crash
        if self.metrics['number_crashes'] == 0:
            current_velocity = np.array([self.odometry.twist.twist.linear.x,
                                         self.odometry.twist.twist.linear.y,
                                         self.odometry.twist.twist.linear.z]).reshape((3, 1))
            travelled_dist = current_velocity * 1. / 20.  # frequency of update
            travelled_dist = np.linalg.norm(travelled_dist)  # integrate
            self.metrics['travelled_dist'] += travelled_dist

        if self.metrics['travelled_dist'] < 5.0:
            # no recording in the first 5 m due to transient
            return
        # Number of crashes per maneuver
        [_, __, dist_squared] = self.pcd_tree.search_knn_vector_3d(quad_position, 1)
        closest_distance = np.sqrt(dist_squared)[0]

        if self.metrics['closest_distance'] > closest_distance and self.metrics['number_crashes'] == 0:
            self.metrics['closest_distance'] = closest_distance

        if closest_distance < self.config.crashed_thr and (not self.crashed):
            # it crashed into something, stop recording. Will not consider a condition to break the experiment now
            print("Crashing into something!")
            self.metrics['number_crashes'] += 1
            self.crashed = True
            # uncomment if you want to stop after crash
            # self.publish_stop_recording_msg()
        # make sure to not count double crashes
        if self.crashed and closest_distance > 1.5 * self.config.crashed_thr:
            self.crashed = False

    def evaluate_dagger_condition(self):
        if self.reference_progress < 50:
            # At the beginning always use expert (otherwise gives gazebo problems)
            print("Starting up!")
            return False
        if self.mode == 'testing':
            return True

        if self.counter < self.config.network_frequency and self.mode == 'iterative':
            self.counter += 1
            return False

        current_position = np.array([self.odometry.pose.pose.position.x,
                                     self.odometry.pose.pose.position.y,
                                     self.odometry.pose.pose.position.z]).reshape((3, 1))

        # check that you are not at the boundary of the pointcloud
        safety_th = 0.5 #m

        if current_position[0] < (self.pc_min[0] + safety_th) or current_position[0] > (self.pc_max[0] - safety_th):
            self.counter = 0
            return False
        if current_position[1] < (self.pc_min[1] + safety_th) or current_position[1] > (self.pc_max[1] - safety_th):
            self.counter = 0
            return False
        if current_position[2] < (self.pc_min[2] + safety_th) or current_position[2] > (self.pc_max[2] - safety_th):
            self.counter = 0
            return False


        reference_point = self.full_reference[self.reference_progress]


        reference_position = np.array([reference_point.pose.position.x,
                                      reference_point.pose.position.y,
                                      reference_point.pose.position.z]).reshape((3, 1))

        distance = np.linalg.norm(current_position - reference_position)

        # check that you are not too far from the reference trajectory
        if distance > self.config.fallback_radius_expert:
            net_in_control = False
            self.counter = 0
        else:
            net_in_control = True
        return net_in_control
