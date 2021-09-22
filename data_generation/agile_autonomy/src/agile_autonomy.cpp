#include "agile_autonomy/agile_autonomy.h"

#include <glog/logging.h>
#include <stdio.h>
#include <experimental/filesystem>
#include <iomanip>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "agile_autonomy_utils/generate_reference.h"
#include "minimum_jerk_trajectories/RapidTrajectoryGenerator.h"
#include "quadrotor_common/parameter_helper.h"
#include "quadrotor_common/trajectory_point.h"
#include "std_msgs/Int32.h"
#include "tf/transform_listener.h"
#include "trajectory_generation_helper/acrobatic_sequence.h"

namespace agile_autonomy {

AgileAutonomy::AgileAutonomy(const ros::NodeHandle& nh,
                             const ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {
  if (!loadParameters()) {
    ROS_ERROR("[%s] Failed to load all parameters",
              ros::this_node::getName().c_str());
    ros::shutdown();
  }

  visualizer_ =
      std::make_shared<visualizer::Visualizer>(nh_, pnh_, "agile_autonomy");
  // Subscribers
  toggle_experiment_sub_ =
      nh_.subscribe("fpv_quad_looping/execute_trajectory", 1,
                    &AgileAutonomy::startExecutionCallback, this);
  odometry_sub_ = nh_.subscribe("ground_truth/odometry", 1,
                                &AgileAutonomy::odometryCallback, this,
                                ros::TransportHints().tcpNoDelay());
  setup_logging_sub_ =
      nh_.subscribe("save_pc", 1, &AgileAutonomy::setupLoggingCallback, this);
  traj_sub_ = nh_.subscribe("trajectory_predicted", 1,
                            &AgileAutonomy::trajectoryCallback, this);
  start_flying_sub_ = pnh_.subscribe("start_flying", 1,
                                     &AgileAutonomy::stopFlyingCallback, this);
  land_sub_ =
      nh_.subscribe("autopilot/land", 1, &AgileAutonomy::landCallback, this);
  off_sub_ =
      nh_.subscribe("autopilot/off", 1, &AgileAutonomy::offCallback, this);
  force_hover_sub_ = nh_.subscribe("autopilot/force_hover", 1,
                                   &AgileAutonomy::forceHoverCallback, this);
  completed_global_plan_sub_ =
      nh_.subscribe("completed_global_plan", 1,
                    &AgileAutonomy::completedGlobalPlanCallback, this);

  // Publishers
  control_command_pub_ = nh_.advertise<quadrotor_msgs::ControlCommand>(
      "autopilot/control_command_input", 1);
  start_flying_pub_ = pnh_.advertise<std_msgs::Bool>("start_flying", 1);
  ref_progress_pub_ = pnh_.advertise<std_msgs::Int32>("reference_progress", 1);
  setpoint_pub_ =
      pnh_.advertise<quadrotor_msgs::TrajectoryPoint>("setpoint", 1);
  compute_global_path_pub_ =
      pnh_.advertise<std_msgs::Float32>("compute_global_plan", 1);

  // Saving timer
  save_timer_ = nh_.createTimer(ros::Duration(1.0 / save_freq_),
                                &AgileAutonomy::saveLoop, this);
  sample_times_.clear();
  for (unsigned int i = 0; i <= traj_len_; i++) {
    sample_times_.push_back(traj_dt_ * i);
  }
  fine_sample_times_.clear();
  for (unsigned int i = 0; i <= static_cast<int>(traj_len_ * traj_dt_ * 50);
       i++) {
    fine_sample_times_.push_back(0.02 * i);
  }

  ROS_INFO("Connecting to unity...");
  flightmare_bridge_ =
      std::make_shared<flightmare_bridge::FlightmareBridge>(nh_, pnh_);
  unity_is_ready_ = true;
}

AgileAutonomy::~AgileAutonomy() { flightmare_bridge_->disconnect(); }

void AgileAutonomy::startExecutionCallback(const std_msgs::BoolConstPtr& msg) {
  ROS_INFO("Received startExecutionCallback message!");
  computeManeuver(msg->data);
}

void AgileAutonomy::setupLoggingCallback(const std_msgs::BoolConstPtr& msg) {
  if (msg->data) {
    ROS_INFO(
        "Initiated Logging, computing reference trajectory and generating "
        "point cloud!");
    computeManeuver(false);
  }
}

void AgileAutonomy::stopFlyingCallback(const std_msgs::BoolConstPtr& msg) {
  // we finished trajectory execution, close the log file
  if (!msg->data && state_machine_ != StateMachine::kComputeLabels) {
    logging_helper_.closeOdometryLog();
    frame_counter_ = 0;
    reference_progress_abs_ = 0;
    ROS_INFO("Switching to kComputeLabels");
    network_prediction_.clear();
    base_controller_.off();
    visualizer_->clearBuffers();
    state_machine_ = StateMachine::kComputeLabels;
  }
}

void AgileAutonomy::landCallback(const std_msgs::EmptyConstPtr& msg) {
  ROS_INFO("Received land command, stopping maneuver execution!");
  ROS_INFO("Switching to kOff");
  network_prediction_.clear();
  base_controller_.off();
  visualizer_->clearBuffers();
  state_machine_ = StateMachine::kOff;
}

void AgileAutonomy::offCallback(const std_msgs::EmptyConstPtr& msg) {
  ROS_INFO("Received off command, stopping maneuver execution!");
  ROS_INFO("Switching to kOff");
  reference_progress_abs_ = 0;
  network_prediction_.clear();
  base_controller_.off();
  visualizer_->clearBuffers();
  state_machine_ = StateMachine::kOff;
  setup_done_ = false;
}

void AgileAutonomy::forceHoverCallback(const std_msgs::EmptyConstPtr& msg) {
  ROS_INFO("Received force hover command, stopping maneuver execution!");
  ROS_INFO("Switching to kOff");
  state_machine_ = StateMachine::kOff;
}

void AgileAutonomy::computeManeuver(const bool only_expert) {
  ros::Time time_start_computation = ros::Time::now();
  ROS_INFO("Starting maneuver computation");
  reference_progress_abs_ = 0;
  quadrotor_common::TrajectoryPoint start_state;
  {
    std::lock_guard<std::mutex> guard(odom_mtx_);
    start_state.position = received_state_est_.position;
    start_state.orientation = received_state_est_.orientation;
    start_state.velocity =
        received_state_est_.velocity +
        start_state.orientation * Eigen::Vector3d::UnitX() * 0.2;
    start_state.acceleration = Eigen::Vector3d::Zero();
  }
  fpv_aggressive_trajectories::AcrobaticSequence acrobatic_sequence(
      start_state);

  bool success = true;

  Eigen::Quaterniond q_body_world = start_state.orientation;
  double end_yaw = yawFromQuaternion(q_body_world);
  Eigen::Vector3d end_position_1 =
      start_state.position +
      q_body_world * Eigen::Vector3d(maneuver_velocity_ / 2.0, 0.0, 0.0);
  Eigen::Vector3d end_velocity_1 =
      q_body_world * Eigen::Vector3d(maneuver_velocity_, 0.0, 0.0);
  Eigen::Vector3d end_position_2 =
      start_state.position +
      q_body_world * Eigen::Vector3d(
                         length_straight_ - maneuver_velocity_ / 2.0, 0.0, 0.0);
  Eigen::Vector3d end_velocity_2 =
      q_body_world * Eigen::Vector3d(maneuver_velocity_, 0.0, 0.0);

  Eigen::Vector3d end_position_3 =
      start_state.position +
      q_body_world * Eigen::Vector3d(length_straight_, 0.0, 0.0);
  acrobatic_sequence.appendStraight(end_position_1, end_velocity_1, end_yaw,
                                    1.1 * maneuver_velocity_,
                                    traj_sampling_freq_);
  acrobatic_sequence.appendStraight(end_position_2, end_velocity_2, end_yaw,
                                    1.1 * maneuver_velocity_,
                                    traj_sampling_freq_, false);
  acrobatic_sequence.appendStraight(end_position_3, Eigen::Vector3d::Zero(),
                                    end_yaw, 1.1 * maneuver_velocity_,
                                    traj_sampling_freq_);

  visualizer_->visualizeTrajectories(acrobatic_sequence.getManeuverList());

  acrobatic_trajectory_.points.clear();
  fuseTrajectories(acrobatic_sequence.getManeuverList(),
                   &acrobatic_trajectory_);

  if (success) {
    // create directory
    logging_helper_.createDirectories(data_dir_, &curr_data_dir_);
    // spawns both trees and objects, depending on the params set
    flightmare_bridge_->spawnObjects(start_state);
    //  save unity point cloud, adapt point cloud size to maneuver
    Eigen::Vector3d max_corner =
        Eigen::Vector3d::Ones() * std::numeric_limits<double>::min();
    Eigen::Vector3d min_corner =
        Eigen::Vector3d::Ones() * std::numeric_limits<double>::max();

    for (auto point : acrobatic_trajectory_.points) {
      min_corner[0] = std::min(min_corner[0], point.position.x());
      min_corner[1] = std::min(min_corner[1], point.position.y());
      min_corner[2] = std::min(min_corner[2], point.position.z());
      max_corner[0] = std::max(max_corner[0], point.position.x());
      max_corner[1] = std::max(max_corner[1], point.position.y());
      max_corner[2] = std::max(max_corner[2], point.position.z());
    }

    flightmare_bridge_->generatePointcloud(min_corner, max_corner,
                                           curr_data_dir_);

    // open log file
    logging_helper_.newOdometryLog(curr_data_dir_ + "/odometry.csv");
    logging_helper_.saveTrajectorytoCSV(
        curr_data_dir_ + "/reference_trajectory.csv", acrobatic_trajectory_);

    rollout_counter_ += 1;
    reference_progress_abs_ = 0;
    viz_id_ = 10;
    visualizer_->clearBuffers();
    std_msgs::Int32 progress_msg;
    progress_msg.data = reference_progress_abs_;
    ref_progress_pub_.publish(progress_msg);

    time_start_logging_ = ros::Time::now();

    received_network_prediction_ = false;
    network_prediction_.clear();
    reference_ready_ = false;

    if (perform_global_planning_) {
      ROS_INFO(
          "Not yet switching state machine, waiting for global reference to "
          "arrive...");
      setup_done_ = false;
      only_expert_ = only_expert;
      std_msgs::Float32 max_speed_msg;
      max_speed_msg.data = maneuver_velocity_;
      compute_global_path_pub_.publish(max_speed_msg);
    } else {
      setup_done_ = true;
      ROS_INFO("Gogogo!");
      if (only_expert) {
        // we only go to kExecuteTrajectory mode if we want to execute expert
        // only else we wait for the first trajectory from the network to arrive
        // before switch to kNetwork mode
        ROS_INFO("Switching to kExecuteExpert");
        state_machine_ = StateMachine::kExecuteExpert;
      } else {
        // setting the state machine to kAutopilot here reallows to switch to
        // network later
        ROS_INFO("Switching to kAutopilot");
        state_machine_ = StateMachine::kAutopilot;
      }
      for (int i = 0; i < 1; i++) {
        std_msgs::Bool bool_msg;
        bool_msg.data = true;
        start_flying_pub_.publish(bool_msg);
        ros::Duration(0.05).sleep();
      }
    }
    ROS_INFO("Maneuver computation successful!");
  } else {
    ROS_ERROR("Maneuver computation failed! Will not execute trajectory.");
  }
  ROS_INFO("Maneuver computation took %.4f seconds.",
           (ros::Time::now() - time_start_computation).toSec());
}

void AgileAutonomy::saveLoop(const ros::TimerEvent& time) {
  // if inputs are ok and unity is ready, we compute & publish depth
  if (unity_is_ready_ && received_state_est_.isValid()) {
    quadrotor_common::QuadStateEstimate temp_state_estimate;
    {
      std::lock_guard<std::mutex> guard(odom_mtx_);
      temp_state_estimate = received_state_est_;
    }
    quadrotor_common::TrajectoryPoint temp_reference;
    {
      std::lock_guard<std::mutex> guard(curr_ref_mtx_);
      temp_reference = curr_reference_;
    }
    std::ostringstream ss;
    ss << std::setw(8) << std::setfill('0') << frame_counter_;
    std::string s2(ss.str());

    // save data to disk
    if (state_machine_ == StateMachine::kExecuteExpert ||
        state_machine_ == StateMachine::kNetwork) {
      if (!logging_helper_.logOdometry(
              temp_state_estimate, temp_reference, time_start_logging_,
              reference_progress_abs_, cam_pitch_angle_)) {
        // log file not yet ready!
        flightmare_bridge_->getImages(temp_state_estimate, "", 0);
        return;
      }

      // we also save the network predictions to file
      if (save_network_trajectories_ && received_network_prediction_) {
        ros::WallTime t_start_log = ros::WallTime::now();
        std::string csv_filename_wf =
            curr_data_dir_ + "/trajectories/trajectories_nw_wf_" + s2 + ".csv";

        logging_helper_.save_nw_pred_to_csv(network_prediction_,
                                            csv_filename_wf);
      }

      flightmare_bridge_->getImages(temp_state_estimate, curr_data_dir_,
                                    frame_counter_);
      frame_counter_ += 1;
    } else {
      // we don't save data but would like to see the images in RViz
      if (!setup_done_) {
        flightmare_bridge_->getImages(temp_state_estimate, "", 0);
      }
    }
  }
}

double AgileAutonomy::yawFromQuaternion(const Eigen::Quaterniond& q) {
  Eigen::Vector3d body_x_world = q * Eigen::Vector3d::UnitX();
  body_x_world[2] = 0.0;
  body_x_world.normalize();
  double yaw = std::atan2(body_x_world.y(), body_x_world.x());
  return yaw;
}

bool AgileAutonomy::convertTrajectoriesToWorldFrame(
    const std::vector<quadrotor_common::Trajectory>& nw_trajectories,
    const rpg::Pose& T_W_S, const TrajectoryExt& prev_ref,
    std::vector<quadrotor_common::Trajectory>* world_trajectories,
    std::vector<double>* trajectory_costs) {
  int i = 1;
  for (auto trajectory : nw_trajectories) {
    world_trajectories->push_back(quadrotor_common::Trajectory());
    trajectory_costs->push_back(0.0);
    int point_idx = 0;
    for (auto point : trajectory.points) {
      // transform position to world frame
      rpg::Pose T_S_C =
          rpg::Pose(point.position, Eigen::Quaterniond::Identity());
      rpg::Pose T_W_C = T_W_S * T_S_C;

      // collect transformed trajectory points in reference trajectory
      quadrotor_common::TrajectoryPoint world_point;
      world_point.time_from_start = point.time_from_start;
      world_point.position = T_W_C.getPosition();
      world_point.position[2] =
          std::max(test_time_min_z_,
                   std::min(world_point.position[2], test_time_max_z_));
      world_trajectories->back().points.push_back(world_point);
      if (reference_ready_) {
        point_idx = std::min(point_idx,
                             static_cast<int>(prev_ref.getPoints().size() - 1));
        trajectory_costs->back() +=
            (world_point.position -
             prev_ref.getPoints()
                 .at(std::min(
                     point_idx,
                     static_cast<int>(prev_ref.getPoints().size()) - 1))
                 .position)
                .norm();
      }
      point_idx++;
    }
    visualizer_->visualizeTrajectory(world_trajectories->back(),
                                     "raw_nw_prediction", viz_id_ + i * 1000,
                                     1.0, 1.0, 1.0, 0.5);
    i++;
  }
  return true;
}

/// converts a stacked network prediction in bodyframe to a single selected
/// trajectory in world frame
bool AgileAutonomy::selectBestNetworkPrediction(
    const std::vector<quadrotor_common::Trajectory>& nw_trajectories,
    const Eigen::Vector3d& start_pos, const Eigen::Quaterniond& start_att,
    quadrotor_common::Trajectory* const selected_trajectory) {
  selected_trajectory->points.clear();

  rpg::Pose T_W_S;
  switch (nw_predictions_frame_id_) {
    case FrameID::Body: {
      T_W_S = rpg::Pose(start_pos, start_att);
      break;
    }
    default: {
      ROS_ERROR("Unsupported network reference frame.");
    }
  }

  // use the previous prediction to decide which NW trajectory to execute
  TrajectoryExt prev_ref;
  if (reference_ready_) {
    std::lock_guard<std::mutex> guard(curr_ref_mtx_);
    prev_ref = TrajectoryExt(prev_ref_traj_, FrameID::World,
                              prev_ref_traj_.points.front());
  }

  std::vector<quadrotor_common::Trajectory> world_trajectories;
  std::vector<double> trajectory_costs;
  convertTrajectoriesToWorldFrame(nw_trajectories, T_W_S, prev_ref,
                                  &world_trajectories, &trajectory_costs);

  // extract best trajectory
  int best_traj_idx = 0;
  if (!reference_ready_) {
    best_traj_idx = 0;
  } else {
    best_traj_idx =
        std::min_element(trajectory_costs.begin(), trajectory_costs.end()) -
        trajectory_costs.begin();
  }
  ROS_INFO("Selected trajectory #%d", best_traj_idx);
  if (best_traj_idx >= world_trajectories.size()) {
    ROS_ERROR("Selected trajectory is out of bounds! Selecting trajectory #0!");
    best_traj_idx = 0;
  }
  for (auto point : world_trajectories.at(best_traj_idx).points) {
    selected_trajectory->points.push_back(point);
  }

  return true;
}

void AgileAutonomy::trajectoryCallback(
    const agile_autonomy_msgs::MultiTrajectoryConstPtr& msg) {
  if (state_machine_ == StateMachine::kComputeLabels ||
      state_machine_ == StateMachine::kOff) {
    return;
  }
  pred_traj_idx_++;
  time_received_prediction_ = msg->header.stamp;
  std::vector<quadrotor_common::Trajectory> nw_trajectories;

  for (auto trajectory : msg->trajectories) {
    nw_trajectories.push_back(trajectory);
  }

  quadrotor_common::Trajectory traj_pred_world;
  traj_pred_world.trajectory_type =
      quadrotor_common::Trajectory::TrajectoryType::GENERAL;
  quadrotor_common::QuadStateEstimate temp_state_estimate;
  {
    std::lock_guard<std::mutex> guard(odom_mtx_);
    temp_state_estimate = received_state_est_;
  }

  quadrotor_common::QuadStateEstimate odom_at_inference;
  odom_at_inference.position =
      Eigen::Vector3d(msg->ref_pose.position.x, msg->ref_pose.position.y,
                      msg->ref_pose.position.z);
  odom_at_inference.orientation =
      Eigen::Quaterniond(
          msg->ref_pose.orientation.w, msg->ref_pose.orientation.x,
          msg->ref_pose.orientation.y, msg->ref_pose.orientation.z) *
      Eigen::Quaternion(std::cos(-cam_pitch_angle_ / 2.0), 0.0,
                        std::sin(-cam_pitch_angle_ / 2.0), 0.0);
  odom_at_inference.velocity = Eigen::Vector3d(
      msg->ref_vel.linear.x, msg->ref_vel.linear.y, msg->ref_vel.linear.z);

  bool execute_nw = msg->execute;
  selectBestNetworkPrediction(nw_trajectories, odom_at_inference.position,
                              odom_at_inference.orientation, &traj_pred_world);
  double red, green, blue;
  red = 0.0;
  blue = 1.0;
  green = 0.0;
  if (!execute_nw) {
    execute_nw = false;
    red = 1.0;
    blue = 0.0;
  }

  TrajectoryExt traj_world_ext(traj_pred_world, FrameID::World,
                                traj_pred_world.points.front());

  quadrotor_common::Trajectory network_traj;
  traj_world_ext.getTrajectory(&network_traj);
  visualizer_->visualizeTrajectory(network_traj, "selected_nw_prediction",
                                   viz_id_, red, green, blue, 1.0);

  TrajectoryExtPoint state_est_point;
  if (reference_ready_) {
    std::lock_guard<std::mutex> guard(curr_ref_mtx_);
    state_est_point.time_from_start = 0.0;
    state_est_point.position = odom_at_inference.position;
    state_est_point.velocity = temp_state_estimate.velocity;
    state_est_point.acceleration = curr_reference_.acceleration;
    state_est_point.jerk = curr_reference_.jerk;
  } else {
    state_est_point.time_from_start = 0.0;
    state_est_point.position = temp_state_estimate.position;
    state_est_point.velocity = temp_state_estimate.velocity;
    state_est_point.acceleration = Eigen::Vector3d::Zero();
  }
  traj_world_ext.replaceFirstPoint(state_est_point);
  traj_world_ext.fitPolynomialCoeffs(5, 3);
  traj_world_ext.enableYawing(enable_yawing_);
  traj_world_ext.resamplePointsFromPolyCoeffs();
  traj_world_ext.getTrajectory(&network_traj);
  visualizer_->visualizeTrajectory(network_traj, "fitted_nw_prediction",
                                   viz_id_, 0.5, 0.5, 0.0, 1.0);
  double desired_speed =
      std::min(temp_state_estimate.velocity.norm() + 1.0, test_time_velocity_);
  traj_world_ext.setConstantArcLengthSpeed(
      desired_speed, static_cast<int>(traj_len_), traj_dt_);
  traj_world_ext.getTrajectory(&network_traj);
  visualizer_->visualizeTrajectory(network_traj, "arclen_nw_prediction",
                                   viz_id_, red, 1.0, blue, 1.0);

  viz_id_ += 1;
  if ((viz_id_ - viz_id_start_) >= num_traj_viz_) {
    viz_id_ = viz_id_start_;
  }

  {
    std::lock_guard<std::mutex> guard(nw_pred_mtx_);
    network_prediction_ = traj_world_ext;
  }
  // go into network mode
  if (execute_nw) {
    state_machine_ = StateMachine::kNetwork;
  } else {
    state_machine_ = StateMachine::kExecuteExpert;
  }
}

void AgileAutonomy::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  odom_idx_++;
  if (odom_idx_ % process_every_nth_odometry_ == 0) {
    odom_idx_ = 0;
  } else {
    return;
  }
  visualizer_->displayQuadrotor();
  quadrotor_common::QuadStateEstimate state_estimate;
  {
    std::lock_guard<std::mutex> guard(odom_mtx_);
    received_state_est_ = *msg;
    if (!velocity_estimate_in_world_frame_) {
      received_state_est_.transformVelocityToWorldFrame();
    }
    state_estimate = received_state_est_;
  }
  // Push received state estimate into predictor
  state_predictor_.updateWithStateEstimate(state_estimate);
  ros::Time time_now = ros::Time::now();

  if (state_machine_ == StateMachine::kComputeLabels ||
      state_machine_ == StateMachine::kOff) {
    return;
  }

  quadrotor_common::Trajectory reference_trajectory;
  if (!acrobatic_trajectory_.points.empty()) {
    computeReferenceTrajectoryPosBased(
        state_estimate.position, acrobatic_trajectory_, traj_len_ * traj_dt_,
        &reference_trajectory, &reference_progress_abs_);

    // publish the progress to python
    std_msgs::Int32 progress_msg;
    progress_msg.data = reference_progress_abs_;
    ref_progress_pub_.publish(progress_msg);

    visualizer_->visualizeTrajectory(reference_trajectory, "nominal_reference",
                                     viz_id_, 1.0, 1.0, 1.0, 1.0);
  }

  if (state_machine_ == StateMachine::kExecuteExpert ||
      state_machine_ == StateMachine::kNetwork) {
    quadrotor_common::ControlCommand control_cmd;
    ros::Time cmd_execution_time = time_now + ros::Duration(ctrl_cmd_delay_);
    quadrotor_common::QuadStateEstimate predicted_state =
        getPredictedStateEstimate(cmd_execution_time, &state_predictor_);

    const ros::Time start_control_command_computation = ros::Time::now();

    // check if we reached the end of the maneuver
    if ((acrobatic_trajectory_.points.back().position - state_estimate.position)
                .norm() < 0.5 ||
        reference_progress_abs_ > (acrobatic_trajectory_.points.size() - 20)) {
      ROS_INFO("Reached end of trajectory.");
      reference_progress_abs_ = acrobatic_trajectory_.points.size() - 1;
      ROS_INFO("Switching to kOff");
      state_machine_ = StateMachine::kOff;
      // publish a message that maneuver is finished
      std_msgs::Bool false_msg;
      false_msg.data = false;
      start_flying_pub_.publish(false_msg);
    }

    if (state_machine_ == StateMachine::kExecuteExpert) {
      quadrotor_common::TrajectoryPoint curr_state;
      curr_state.position = predicted_state.position;
      curr_state.velocity = predicted_state.velocity;
      curr_state.orientation = predicted_state.orientation;

      TrajectoryExt rollout(reference_trajectory, FrameID::World, curr_state);
      rollout.truncateBack(traj_dt_ * traj_len_);
      rollout.enableYawing(enable_yawing_);
      rollout.convertToFrame(nw_predictions_frame_id_, curr_state.position,
                             curr_state.orientation);

      unsigned int poly_order = 5;
      unsigned int continuity_order = 2;
      rollout.fitPolynomialCoeffs(poly_order, continuity_order);
      rollout.resamplePointsFromPolyCoeffs();
      double desired_speed =
          std::min(state_estimate.velocity.norm() + 1.0, test_time_velocity_);
      rollout.convertToFrame(FrameID::World, curr_state.position,
                             curr_state.orientation);
      rollout.getTrajectory(&reference_trajectory);
      visualizer_->visualizeTrajectory(reference_trajectory, "fitted_reference",
                                       101, 1.0, 1.0, 0.0);
    } else if (state_machine_ == StateMachine::kNetwork &&
               !network_prediction_.getPoints().empty()) {
      double t_shift =
          std::max(0.0, (time_now - time_received_prediction_).toSec());
      quadrotor_common::Trajectory tmp_ref;
      {
        std::lock_guard<std::mutex> guard(nw_pred_mtx_);
        network_prediction_.getTrajectory(&tmp_ref);
      }
      reference_trajectory.points.clear();

      for (auto point : tmp_ref.points) {
        quadrotor_common::TrajectoryPoint tmp_point = tmp_ref.getStateAtTime(
            point.time_from_start + ros::Duration(t_shift));
        reference_trajectory.points.push_back(tmp_point);
      }
    }

    // feed the reference to the controller
    {
      std::lock_guard<std::mutex> guard(curr_ref_mtx_);
      curr_reference_ = reference_trajectory.points.front();
      prev_ref_traj_ = reference_trajectory;
      t_prev_ref_traj_ = time_now;
      reference_ready_ = true;
    }

    // visualize reference trajectory first
    visualizer_->visualizeTrajectory(reference_trajectory, "control_ref", 0,
                                     0.0, 1.0, 0.0);
    visualizer_->visualizeExecutedReference(
        reference_trajectory.points.front());
    visualizer_->visualizeExecutedTrajectory(*msg);
    setpoint_pub_.publish(reference_trajectory.points.front().toRosMessage());
    control_cmd = base_controller_.run(predicted_state, reference_trajectory,
                                       base_controller_params_);
    control_cmd.timestamp = time_now;
    control_cmd.expected_execution_time = cmd_execution_time;
    const ros::Duration control_computation_time =
        ros::Time::now() - start_control_command_computation;

    publishControlCommand(control_cmd);
  }
}

void AgileAutonomy::completedGlobalPlanCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg->data) {
    ROS_WARN(
        "Planner failed, not switching state and waiting for python to "
        "restart "
        "things...");
    return;
  }
  ROS_INFO(
      "Global Planner completed! Loading adapted reference trajectory now!");
  quadrotor_common::Trajectory adapted_reference;

  std::string adapted_ref_fname = curr_data_dir_ + "/ellipsoid_trajectory.csv";
  loadReferenceTrajectory(&adapted_reference, adapted_ref_fname, true);

  // check the trajectory
  smoothTrajectory(&adapted_reference, maneuver_velocity_);
  logging_helper_.saveTrajectorytoCSV(adapted_ref_fname, adapted_reference);

  for (auto& point : adapted_reference.points) {
    printf(
        "t: %.2f | Position: %.2f, %.2f, %.2f | Velocity: %.2f, %.2f, %.2f\n",
        point.time_from_start.toSec(), point.position.x(), point.position.y(),
        point.position.z(), point.velocity.x(), point.velocity.y(),
        point.velocity.z());
  }

  received_network_prediction_ = false;
  network_prediction_.clear();
  reference_ready_ = false;
  setup_done_ = true;

  acrobatic_trajectory_.points.clear();
  acrobatic_trajectory_ = adapted_reference;
  ROS_INFO("Gogogo!");
  if (only_expert_) {
    ROS_INFO("Switching to kExecuteExpert");
    state_machine_ = StateMachine::kExecuteExpert;
  } else {
    ROS_INFO("Switching to kAutopilot");
    state_machine_ = StateMachine::kAutopilot;
  }
  for (int i = 0; i < 1; i++) {
    std_msgs::Bool bool_msg;
    bool_msg.data = true;
    start_flying_pub_.publish(bool_msg);
    ros::Duration(0.05).sleep();
  }
}

quadrotor_common::QuadStateEstimate AgileAutonomy::getPredictedStateEstimate(
    const ros::Time& time, const state_predictor::StatePredictor* predictor) {
  return predictor->predictState(time);
}

void AgileAutonomy::publishControlCommand(
    const quadrotor_common::ControlCommand& control_command) {
  if (state_machine_ == StateMachine::kExecuteExpert ||
      state_machine_ == StateMachine::kNetwork) {
    quadrotor_msgs::ControlCommand control_cmd_msg;
    control_cmd_msg = control_command.toRosMessage();
    control_command_pub_.publish(control_cmd_msg);
    state_predictor_.pushCommandToQueue(control_command);
  }
}

bool AgileAutonomy::loadParameters() {
  ROS_INFO("Loading parameters...");

  if (!quadrotor_common::getParam("general/velocity_estimate_in_world_frame",
                                  velocity_estimate_in_world_frame_, false))
    return false;
  if (!quadrotor_common::getParam("general/control_command_delay",
                                  ctrl_cmd_delay_, 0.0))
    return false;
  if (!quadrotor_common::getParam("general/num_traj_viz", num_traj_viz_, 100))
    return false;
  if (!quadrotor_common::getParam("general/perform_global_planning",
                                  perform_global_planning_, false))
    return false;
  if (!quadrotor_common::getParam("general/test_time_velocity",
                                  test_time_velocity_, 1.0))
    return false;
  if (!quadrotor_common::getParam("general/test_time_max_z", test_time_max_z_,
                                  5.0))
    return false;
  if (!quadrotor_common::getParam("general/test_time_min_z", test_time_min_z_,
                                  1.0))
    return false;

  int nw_predictions_frame_id;
  if (!quadrotor_common::getParam("general/nw_predictions_frame_id",
                                  nw_predictions_frame_id, 0))
    return false;

  if (!quadrotor_common::getParam("general/process_every_nth_odometry",
                                  process_every_nth_odometry_, 1))
    return false;

  if (!quadrotor_common::getParam("maneuver/length_straight", length_straight_,
                                  35.0))
    return false;

  if (!quadrotor_common::getParam("maneuver/maneuver_velocity",
                                  maneuver_velocity_, 1.0))
    return false;
  int traj_len;
  if (!quadrotor_common::getParam("trajectory/traj_len", traj_len, 10))
    return false;
  if (!quadrotor_common::getParam("trajectory/traj_dt", traj_dt_, 0.1))
    return false;
  if (!quadrotor_common::getParam("trajectory/continuity_order",
                                  continuity_order_, 1))
    return false;
  if (!quadrotor_common::getParam("trajectory/enable_yawing", enable_yawing_,
                                  false))
    return false;
  if (!quadrotor_common::getParam("data_generation/save_freq", save_freq_,
                                  10.0))
    return false;

  double pitch_angle_deg;
  if (!quadrotor_common::getParam("camera/pitch_angle_deg", pitch_angle_deg,
                                  0.0))
    return false;
  cam_pitch_angle_ = pitch_angle_deg / 180.0 * M_PI;

  traj_len_ = static_cast<unsigned int>(traj_len);

  switch (nw_predictions_frame_id) {
    case 0: {
      nw_predictions_frame_id_ = FrameID::World;
      break;
    }
    case 1: {
      nw_predictions_frame_id_ = FrameID::Body;
      break;
    }
  }

  if (!pnh_.getParam("data_dir", data_dir_)) {
    return false;
  }

  ROS_INFO("Successfully loaded parameters!");
  return true;
}

}  // namespace agile_autonomy

int main(int argc, char** argv) {
  ros::init(argc, argv, "agile_autonomy");
  agile_autonomy::AgileAutonomy agile_autonomy;

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();

  return 0;
}
