#pragma once

#include <fstream>
#include <random>
#include "quadrotor_common/trajectory.h"
#include "ros/ros.h"
#include "trajectory_generation_helper/acrobatic_sequence.h"

void fuseTrajectories(
    const std::list<quadrotor_common::Trajectory>& maneuver_list,
    quadrotor_common::Trajectory* acrobatic_trajectory) {
  ros::Duration time_wrapover = ros::Duration(0.0);
  for (auto trajectory : maneuver_list) {
    for (auto point : trajectory.points) {
      // save trajectory point
      point.time_from_start += time_wrapover;
      acrobatic_trajectory->points.push_back(point);
    }
    time_wrapover = time_wrapover + trajectory.points.back().time_from_start;
  }
}

void computeReferenceTrajectoryPosBased(
    const Eigen::Vector3d& curr_position,
    const quadrotor_common::Trajectory& full_reference,
    const double desired_duration,
    quadrotor_common::Trajectory* reference_trajectory,
    int* reference_progress) {
  quadrotor_common::TrajectoryPoint reference_state;

  // New trajectory where we fill in our lookahead horizon.
  *reference_trajectory = quadrotor_common::Trajectory();
  reference_trajectory->trajectory_type =
      quadrotor_common::Trajectory::TrajectoryType::GENERAL;

  // first, iterate N points forward and advance progress
  double min_dist = std::numeric_limits<double>::max();
  int point_idx(0);
  for (auto point : full_reference.points) {
    if (point_idx >= *reference_progress) {
      double curr_dist = (point.position - curr_position).norm();
      if (curr_dist <= min_dist) {
        min_dist = curr_dist;
      } else {
        *reference_progress = point_idx - 1;
        break;
      }
    }
    point_idx++;
  }

  int added_points = 0;
  point_idx = 0;
  for (auto point : full_reference.points) {
    // Add a point if the time corresponds to a sample on the lookahead.
    if (point_idx >= *reference_progress) {
      reference_trajectory->points.push_back(point);
      if ((reference_trajectory->points.back().time_from_start -
           reference_trajectory->points.front().time_from_start)
              .toSec() > desired_duration) {
        break;
      }
      added_points++;
    }
    point_idx++;
  }

  if (reference_trajectory->points.front().velocity.norm() < 0.1) {
    reference_trajectory->points.front().velocity +=
        0.2 * Eigen::Vector3d::UnitX();
  }

  // handle case of empty reference_trajectory
  if (reference_trajectory->points.empty()) {
    ROS_WARN("Empty reference trajectory!");
    *reference_trajectory = quadrotor_common::Trajectory(reference_state);
  }
}

void loadReferenceTrajectory(quadrotor_common::Trajectory* reference_trajectory,
                             const std::string filename,
                             const bool verbose = false) {
  reference_trajectory->points.clear();
  // open file
  std::ifstream csvFile;
  csvFile.open(filename.c_str());

  if (!csvFile.is_open()) {
    std::cout << "Path Wrong!!!!" << std::endl;
    std::cout << filename << std::endl;
    exit(EXIT_FAILURE);
  } else {
    if (verbose) {
      std::cout << "Loading reference trajectory from " << filename
                << std::endl;
    }
  }

  std::string line;
  // skip header line
  getline(csvFile, line);
  int idx = 0;
  while (getline(csvFile, line)) {
    std::istringstream iss(line);
    std::string lineStream;
    std::string::size_type sz;

    std::vector<double> row;

    while (getline(iss, lineStream, ',')) {
      row.push_back(stold(lineStream, &sz));  // convert to double
    }

    quadrotor_common::TrajectoryPoint point;
    point.time_from_start = ros::Duration(row[0]);
    // position
    point.position.x() = row[1];
    point.position.y() = row[2];
    point.position.z() = row[3];
    // linear velocity
    point.velocity.x() = row[4];
    point.velocity.y() = row[5];
    point.velocity.z() = row[6];
    // acceleration
    point.acceleration.x() = row[7];
    point.acceleration.y() = row[8];
    point.acceleration.z() = row[9];
    // jerk
    point.jerk.x() = row[10];
    point.jerk.y() = row[11];
    point.jerk.z() = row[12];
    // snap
    point.snap.x() = row[13];
    point.snap.y() = row[14];
    point.snap.z() = row[15];
    // attitude
    point.orientation.w() = row[16];
    point.orientation.x() = row[17];
    point.orientation.y() = row[18];
    point.orientation.z() = row[19];
    // bodyrates
    point.bodyrates.x() = row[20];
    point.bodyrates.y() = row[21];
    point.bodyrates.z() = row[22];
    // angular acceleration
    point.angular_acceleration.x() = row[23];
    point.angular_acceleration.y() = row[24];
    point.angular_acceleration.z() = row[25];
    // angular jerk
    point.angular_jerk.x() = row[26];
    point.angular_jerk.y() = row[27];
    point.angular_jerk.z() = row[28];
    // angular snap
    point.angular_snap.x() = row[29];
    point.angular_snap.y() = row[30];
    point.angular_snap.z() = row[31];
    // heading (should be absorbed in the quaternion & bodyrates here)
    point.heading = row[32];
    point.heading_rate = row[33];
    point.heading_acceleration = row[34];

    reference_trajectory->points.push_back(point);
  }
}

void loadReferenceFromFile(quadrotor_common::Trajectory* reference_trajectory,
                           const double& traj_dt, const std::string filename,
                           const bool verbose) {
  reference_trajectory->points.clear();
  // open file
  std::ifstream csvFile;
  csvFile.open(filename.c_str());

  if (!csvFile.is_open()) {
    std::cout << "Path Wrong!!!!" << std::endl;
    std::cout << filename << std::endl;
    exit(EXIT_FAILURE);
  } else {
    if (verbose) {
      std::cout << "Loading initial guess trajectory from " << filename
                << std::endl;
    }
  }

  std::string line;
  // skip header line
  getline(csvFile, line);
  int idx = 0;
  while (getline(csvFile, line)) {
    std::istringstream iss(line);
    std::string lineStream;
    std::string::size_type sz;

    std::vector<double> row;

    while (getline(iss, lineStream, ',')) {
      row.push_back(stold(lineStream, &sz));  // convert to double
    }
    int n_points = (static_cast<int>(row.size()) - 1) / 17;
    for (int i = 0; i < n_points; i++) {
      quadrotor_common::TrajectoryPoint point;
      point.time_from_start = ros::Duration(i * traj_dt);
      // position
      point.position.x() = row[17 * i + 0];
      point.position.y() = row[17 * i + 1];
      point.position.z() = row[17 * i + 2];
      // linear velocity
      point.velocity.x() = row[17 * i + 3];
      point.velocity.y() = row[17 * i + 4];
      point.velocity.z() = row[17 * i + 5];

      // acceleration
      point.acceleration.x() = row[17 * i + 6];
      point.acceleration.y() = row[17 * i + 7];
      point.acceleration.z() = row[17 * i + 8];

      // jerk
      point.jerk.x() = 0.0;
      point.jerk.y() = 0.0;
      point.jerk.z() = 0.0;
      // snap
      point.snap.x() = 0.0;
      point.snap.y() = 0.0;
      point.snap.z() = 0.0;
      // attitude
      point.orientation.w() = row[17 * i + 9];
      point.orientation.x() = row[17 * i + 10];
      point.orientation.y() = row[17 * i + 11];
      point.orientation.z() = row[17 * i + 12];
      // bodyrates
      point.bodyrates.x() = 0.0;
      point.bodyrates.y() = 0.0;
      point.bodyrates.z() = 0.0;
      // angular acceleration
      point.angular_acceleration.x() = 0.0;
      point.angular_acceleration.y() = 0.0;
      point.angular_acceleration.z() = 0.0;
      // angular jerk
      point.angular_jerk.x() = 0.0;
      point.angular_jerk.y() = 0.0;
      point.angular_jerk.z() = 0.0;
      // angular snap
      point.angular_snap.x() = 0.0;
      point.angular_snap.y() = 0.0;
      point.angular_snap.z() = 0.0;
      // heading (should be absorbed in the quaternion & bodyrates here)
      point.heading = 0.0;
      point.heading_rate = 0.0;
      point.heading_acceleration = 0.0;

      reference_trajectory->points.push_back(point);
    }
  }

  if (verbose) {
    std::cout << "Successfully loaded initial guess trajectory from "
              << filename << std::endl;
  }
}

void smoothTrajectory(quadrotor_common::Trajectory* trajectory,
                      const double& max_speed) {
  printf("Smoothing trajectory of length [%lu]\n", trajectory->points.size());

  std::vector<quadrotor_common::TrajectoryPoint> smooth_trajectory;
  double desired_spacing = 0.001;  // distance between points in meter
  quadrotor_common::TrajectoryPoint curr_point, prev_point, next_point;
  curr_point.position = trajectory->points.front().position;
  smooth_trajectory.push_back(curr_point);
  double t_iter = 0.0;
  double dt = 0.0001;
  while ((curr_point.position - trajectory->points.back().position).norm() >
         0.2) {
    prev_point = trajectory->getStateAtTime(ros::Duration(t_iter - dt));
    curr_point = trajectory->getStateAtTime(ros::Duration(t_iter));
    next_point = trajectory->getStateAtTime(ros::Duration(t_iter + dt));
    curr_point.acceleration =
        (next_point.velocity - prev_point.velocity) / (2.0 * dt);

    Eigen::Vector3d thrust =
        curr_point.acceleration + 9.81 * Eigen::Vector3d::UnitZ();
    Eigen::Quaterniond q_pitch_roll =
        Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), thrust);
    curr_point.orientation = q_pitch_roll;
    //    printf("curr_point.position = [%.2f, %.2f, %.2f]\n",
    //           curr_point.position.x(), curr_point.position.y(),
    //           curr_point.position.z());
    if ((smooth_trajectory.back().position - curr_point.position).norm() >=
        desired_spacing) {
      smooth_trajectory.push_back(curr_point);
    }
    t_iter += dt;
  }

  printf("Smoothed trajectory has [%lu] points.\n", smooth_trajectory.size());

  double max_acc = 5.0;
  double curr_speed = 0.0;
  double sample_dt = 0.01;
  double curr_t = 0.0;
  int arclen_progress = 0;

  quadrotor_common::Trajectory resampled_trajectory;
  quadrotor_common::TrajectoryPoint resampled_point;
  resampled_point.time_from_start = ros::Duration(curr_t);
  resampled_point.position = smooth_trajectory.front().position;
  resampled_point.velocity = Eigen::Vector3d::Zero();
  resampled_point.orientation = smooth_trajectory.front().orientation;
  resampled_trajectory.points.push_back(resampled_point);

  // from now on I compute a velocity and search along the smooth trajectory for
  // the next position that matches the desired displacement
  bool arrived_at_end = false;
  // this implements a simple velocity rampup with a desired acceleration
  while ((resampled_trajectory.points.back().position -
          smooth_trajectory.back().position)
                 .norm() > 0.2 &&
         !arrived_at_end) {
    curr_speed = std::min(resampled_trajectory.points.back().velocity.norm() +
                              max_acc * sample_dt,
                          max_speed);
    curr_t += sample_dt;
    //    printf("curr_speed: %.3f\n", curr_speed);
    while (true) {
      double d_pos = (resampled_trajectory.points.back().position -
                      smooth_trajectory.at(arclen_progress).position)
                         .norm();
      //      printf("d_pos = %.5f\n", d_pos);
      if (d_pos >= curr_speed * sample_dt) {
        break;
      }
      arclen_progress++;
      if (arclen_progress >= smooth_trajectory.size() - 10) {
        arrived_at_end = true;
        break;
      }
    }

    resampled_point.time_from_start = ros::Duration(curr_t);
    resampled_point.position = smooth_trajectory.at(arclen_progress).position;
    Eigen::Vector3d vel_dir =
        (smooth_trajectory.at(arclen_progress + 1).position -
         smooth_trajectory.at(arclen_progress).position)
            .normalized();
    resampled_point.velocity = vel_dir * curr_speed;
    resampled_point.orientation =
        smooth_trajectory.at(arclen_progress).orientation;
    resampled_trajectory.points.push_back(resampled_point);
  }

  printf("resampled_trajectory has [%lu] points.\n",
         resampled_trajectory.points.size());

  // write the new trajectory back
  trajectory->points.clear();
  for (auto& point : resampled_trajectory.points) {
    trajectory->points.push_back(point);
  }
}
