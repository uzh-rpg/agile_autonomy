#pragma once

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include "nav_msgs/Odometry.h"
#include "quadrotor_common/trajectory.h"

namespace visualizer {
class Visualizer {
 public:
  Visualizer(const ros::NodeHandle& nh, const ros::NodeHandle& pnh,
             const std::string& identifier);

  Visualizer()
      : Visualizer(ros::NodeHandle(), ros::NodeHandle("~"), "unknown") {}

  virtual ~Visualizer();

  void visualizeTrajectories(
      const std::list<quadrotor_common::Trajectory>& maneuver_list);

  void create_vehicle_markers(int num_rotors, float arm_len, float body_width,
                              float body_height);

  void displayQuadrotor();

  void visualizeTrajectory(const quadrotor_common::Trajectory& trajectory,
                           std::string topic, const int& id, const double& red,
                           const double& green, const double& blue,
                           const double& alpha = 1.0);

  void visualizeExecutedTrajectory(const nav_msgs::Odometry& odometry);

  void visualizeExecutedReference(
      const quadrotor_common::TrajectoryPoint& reference);

  void clearBuffers();

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  std::string identifier_;
  ros::Publisher bodyrates_viz_pub_;
  ros::Publisher active_ref_pub_;
  ros::Publisher planned_traj_pub_;
  ros::Publisher network_pred_pub_;
  ros::Publisher vehicle_marker_pub_;
  ros::Publisher raw_network_pred_pub_;
  ros::Publisher fitted_network_pred_pub_;
  ros::Publisher expert_traj_pub_;

  std::map<std::string, ros::Publisher> publishers_;
  std::map<std::string, ros::Publisher> path_publishers_;

  std::vector<nav_msgs::Odometry> odometry_buffer_;
  std::vector<quadrotor_common::TrajectoryPoint> reference_buffer_;

  std::shared_ptr<visualization_msgs::MarkerArray> vehicle_marker_;
};
}  // namespace visualizer
