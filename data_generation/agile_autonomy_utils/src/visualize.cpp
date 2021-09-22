#include "agile_autonomy_utils/visualize.h"

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

namespace visualizer {

Visualizer::Visualizer(const ros::NodeHandle& nh, const ros::NodeHandle& pnh,
                       const std::string& identifier)
    : nh_(nh), pnh_(pnh), identifier_(identifier) {
  printf("Initiated visualizer with identifier [%s]\n", identifier_.c_str());

  int num_rotors = 4;
  float arm_len = 0.2;
  float body_width = 0.2;
  float body_height = 0.1;
  create_vehicle_markers(num_rotors, arm_len, body_width, body_height);

  planned_traj_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      identifier_ + "/planned_trajectory", 1);
  vehicle_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      identifier_ + "/quadrotor_viz", 1);
}

Visualizer::~Visualizer() {}

void Visualizer::visualizeTrajectory(
    const quadrotor_common::Trajectory& trajectory, std::string topic,
    const int& id, const double& red, const double& green, const double& blue,
    const double& alpha) {
  std::for_each(topic.begin(), topic.end(), [](char& c) { c = ::tolower(c); });
  if (publishers_.find(topic) == publishers_.end()) {
    ROS_INFO("Creating a new publisher for topic [%s].", topic.c_str());
    ros::Publisher curr_publisher, path_publisher;
    curr_publisher = nh_.advertise<visualization_msgs::MarkerArray>(
        identifier_ + "/trajectory/" + topic, 1);
    path_publisher =
        nh_.advertise<nav_msgs::Path>(identifier_ + "/path/" + topic, 1);
    publishers_.insert(

        std::pair<std::string, ros::Publisher>(topic, curr_publisher));
    path_publishers_.insert(
        std::pair<std::string, ros::Publisher>(topic, path_publisher));
  }

  // actual visualization logic...
  visualization_msgs::MarkerArray marker_msg_trajectory;
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "";
  marker.action = visualization_msgs::Marker::MODIFY;
  marker.lifetime = ros::Duration(0);
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.id = id;

  marker.scale.x = 0.015;
  marker.color.r = red;
  marker.color.g = green;
  marker.color.b = blue;
  marker.color.a = alpha;

  nav_msgs::Path path_msg;
  path_msg.header.stamp = ros::Time::now();
  path_msg.header.frame_id = "world";

  for (auto it = trajectory.points.begin(); it != trajectory.points.end();
       it++) {
    geometry_msgs::Point point;
    point.x = it->position.x();
    point.y = it->position.y();
    point.z = it->position.z();
    marker.points.push_back(point);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = "world";
    pose_stamped.header.stamp = ros::Time::now() + it->time_from_start;
    pose_stamped.pose.position = point;
    pose_stamped.pose.orientation.w = it->orientation.w();
    pose_stamped.pose.orientation.x = it->orientation.x();
    pose_stamped.pose.orientation.y = it->orientation.y();
    pose_stamped.pose.orientation.z = it->orientation.z();
    path_msg.poses.push_back(pose_stamped);
  }

  marker_msg_trajectory.markers.push_back(marker);
  publishers_.at(topic).publish(marker_msg_trajectory);
  path_publishers_.at(topic).publish(path_msg);
}

void Visualizer::visualizeTrajectories(
    const std::list<quadrotor_common::Trajectory>& maneuver_list) {
  visualization_msgs::MarkerArray markers;
  int marker_id = 0;
  double marker_lifetime = 300.0;

  for (auto trajectory : maneuver_list) {
    for (auto point : trajectory.points) {
      Eigen::Quaternion<double> q_pose =
          Eigen::Quaterniond(point.orientation.w(), point.orientation.x(),
                             point.orientation.y(), point.orientation.z());
      Eigen::Quaternion<double> q_heading =
          Eigen::Quaternion<double>(Eigen::AngleAxis<double>(
              point.heading, Eigen::Matrix<double, 3, 1>::UnitZ()));
      Eigen::Quaternion<double> q_orientation = q_pose * q_heading;

      // publish marker of reference at this point
      visualization_msgs::Marker z_axis;
      z_axis.header.frame_id = "world";
      z_axis.ns = "";
      z_axis.lifetime = ros::Duration(marker_lifetime);
      z_axis.header.stamp = ros::Time();
      z_axis.type = visualization_msgs::Marker::Type::CYLINDER;
      z_axis.id = marker_id;
      Eigen::Vector3d offset_body = Eigen::Vector3d(0.0, 0.0, 0.1);
      Eigen::Vector3d offset_world = q_orientation * offset_body;
      Eigen::Vector3d pos_corr = point.position + offset_world;
      // set off position by half the marker size
      z_axis.pose.position.x = pos_corr.x();
      z_axis.pose.position.y = pos_corr.y();
      z_axis.pose.position.z = pos_corr.z();
      z_axis.pose.orientation.w = q_orientation.w();
      z_axis.pose.orientation.x = q_orientation.x();
      z_axis.pose.orientation.y = q_orientation.y();
      z_axis.pose.orientation.z = q_orientation.z();
      z_axis.scale.x = 0.05;
      z_axis.scale.y = 0.05;
      z_axis.scale.z = 0.2;
      z_axis.color.a = 1.0;
      z_axis.color.r = 0.0;
      z_axis.color.g = 0.0;
      z_axis.color.b = 1.0;
      markers.markers.push_back(z_axis);

      // rotate to get x-axis
      Eigen::Quaternion<double> q_z_x =
          Eigen::Quaternion<double>(Eigen::AngleAxis<double>(
              M_PI / 2.0, Eigen::Matrix<double, 3, 1>::UnitY()));
      Eigen::Quaternion<double> q_x = q_orientation * q_z_x;
      visualization_msgs::Marker x_axis;
      x_axis.header.frame_id = "world";
      x_axis.ns = "";
      x_axis.lifetime = ros::Duration(marker_lifetime);
      x_axis.header.stamp = ros::Time();
      x_axis.type = visualization_msgs::Marker::Type::CYLINDER;
      x_axis.id = marker_id + 1;
      offset_body = Eigen::Vector3d(0.0, 0.0, 0.1);
      offset_world = q_x * offset_body;
      pos_corr = point.position + offset_world;
      x_axis.pose.position.x = pos_corr.x();
      x_axis.pose.position.y = pos_corr.y();
      x_axis.pose.position.z = pos_corr.z();
      x_axis.pose.orientation.w = q_x.w();
      x_axis.pose.orientation.x = q_x.x();
      x_axis.pose.orientation.y = q_x.y();
      x_axis.pose.orientation.z = q_x.z();
      x_axis.scale.x = 0.05;
      x_axis.scale.y = 0.05;
      x_axis.scale.z = 0.2;
      x_axis.color.a = 1.0;
      x_axis.color.r = 1.0;
      x_axis.color.g = 0.0;
      x_axis.color.b = 0.0;
      markers.markers.push_back(x_axis);

      // rotate to get y-axis
      Eigen::Quaternion<double> q_z_y =
          Eigen::Quaternion<double>(Eigen::AngleAxis<double>(
              -M_PI / 2.0, Eigen::Matrix<double, 3, 1>::UnitX()));
      Eigen::Quaternion<double> q_y = q_orientation * q_z_y;
      visualization_msgs::Marker y_axis;
      y_axis.header.frame_id = "world";
      y_axis.ns = "";
      y_axis.lifetime = ros::Duration(marker_lifetime);
      y_axis.header.stamp = ros::Time();
      y_axis.type = visualization_msgs::Marker::Type::CYLINDER;
      y_axis.id = marker_id + 2;
      offset_body = Eigen::Vector3d(0.0, 0.0, 0.1);
      offset_world = q_y * offset_body;
      pos_corr = point.position + offset_world;
      y_axis.pose.position.x = pos_corr.x();
      y_axis.pose.position.y = pos_corr.y();
      y_axis.pose.position.z = pos_corr.z();
      y_axis.pose.orientation.w = q_y.w();
      y_axis.pose.orientation.x = q_y.x();
      y_axis.pose.orientation.y = q_y.y();
      y_axis.pose.orientation.z = q_y.z();
      y_axis.scale.x = 0.05;
      y_axis.scale.y = 0.05;
      y_axis.scale.z = 0.2;
      y_axis.color.a = 1.0;
      y_axis.color.r = 0.0;
      y_axis.color.g = 1.0;
      y_axis.color.b = 0.0;
      markers.markers.push_back(y_axis);

      marker_id += 3;
    }
  }
  planned_traj_pub_.publish(markers);
}

void Visualizer::visualizeExecutedTrajectory(
    const nav_msgs::Odometry& odometry) {
  std::string topic = "executed_trajectory";
  std::for_each(topic.begin(), topic.end(), [](char& c) { c = ::tolower(c); });
  if (publishers_.find(topic) == publishers_.end()) {
    ROS_INFO("Creating a new publisher for topic [%s].", topic.c_str());
    ros::Publisher curr_publisher, path_publisher;
    curr_publisher = nh_.advertise<visualization_msgs::MarkerArray>(
        identifier_ + "/trajectory/" + topic, 1);
    path_publisher =
        nh_.advertise<nav_msgs::Path>(identifier_ + "/path/" + topic, 1);
    publishers_.insert(

        std::pair<std::string, ros::Publisher>(topic, curr_publisher));
    path_publishers_.insert(
        std::pair<std::string, ros::Publisher>(topic, path_publisher));
  }

  odometry_buffer_.push_back(odometry);

  nav_msgs::Path path_msg;
  path_msg.header.stamp = ros::Time::now();
  path_msg.header.frame_id = "world";

  for (auto curr_odom : odometry_buffer_) {
    geometry_msgs::PoseStamped curr_pose;
    curr_pose.header.stamp = curr_odom.header.stamp;
    curr_pose.pose = curr_odom.pose.pose;
    path_msg.poses.push_back(curr_pose);
  }

  path_publishers_.at(topic).publish(path_msg);
}

void Visualizer::visualizeExecutedReference(
    const quadrotor_common::TrajectoryPoint& reference) {
  std::string topic = "executed_reference";
  std::for_each(topic.begin(), topic.end(), [](char& c) { c = ::tolower(c); });
  if (publishers_.find(topic) == publishers_.end()) {
    ROS_INFO("Creating a new publisher for topic [%s].", topic.c_str());
    ros::Publisher curr_publisher, path_publisher;
    curr_publisher = nh_.advertise<visualization_msgs::MarkerArray>(
        identifier_ + "/trajectory/" + topic, 1);
    path_publisher =
        nh_.advertise<nav_msgs::Path>(identifier_ + "/path/" + topic, 1);
    publishers_.insert(

        std::pair<std::string, ros::Publisher>(topic, curr_publisher));
    path_publishers_.insert(
        std::pair<std::string, ros::Publisher>(topic, path_publisher));
  }

  reference_buffer_.push_back(reference);

  nav_msgs::Path path_msg;
  path_msg.header.stamp = ros::Time::now();
  path_msg.header.frame_id = "world";

  for (auto curr_ref : reference_buffer_) {
    geometry_msgs::PoseStamped curr_pose;
    curr_pose.header.stamp = path_msg.header.stamp;
    curr_pose.pose.position.x = curr_ref.position.x();
    curr_pose.pose.position.y = curr_ref.position.y();
    curr_pose.pose.position.z = curr_ref.position.z();
    curr_pose.pose.orientation.w = curr_ref.orientation.w();
    curr_pose.pose.orientation.x = curr_ref.orientation.x();
    curr_pose.pose.orientation.y = curr_ref.orientation.y();
    curr_pose.pose.orientation.z = curr_ref.orientation.z();
    path_msg.poses.push_back(curr_pose);
  }

  path_publishers_.at(topic).publish(path_msg);
}

void Visualizer::clearBuffers() {
  odometry_buffer_.clear();
  reference_buffer_.clear();
}

void Visualizer::displayQuadrotor() {
  vehicle_marker_pub_.publish(*vehicle_marker_);
}

void Visualizer::create_vehicle_markers(int num_rotors, float arm_len,
                                        float body_width, float body_height) {
  if (num_rotors <= 0) num_rotors = 2;

  if (vehicle_marker_) return;

  double marker_scale_ = 1.0;
  vehicle_marker_ = std::make_shared<visualization_msgs::MarkerArray>();
  vehicle_marker_->markers.reserve(2 * num_rotors + 1);
  // child_frame_id_ = "hawk_corrected";
  // rotor marker template
  visualization_msgs::Marker rotor;
  rotor.header.stamp = ros::Time();
  rotor.header.frame_id = "/hummingbird/base_link";
  rotor.ns = "vehicle_rotor";
  rotor.action = visualization_msgs::Marker::ADD;
  rotor.type = visualization_msgs::Marker::CYLINDER;
  rotor.scale.x = 0.2 * marker_scale_;
  rotor.scale.y = 0.2 * marker_scale_;
  rotor.scale.z = 0.01 * marker_scale_;
  rotor.color.r = 0.4;
  rotor.color.g = 0.4;
  rotor.color.b = 0.4;
  rotor.color.a = 0.8;
  rotor.pose.position.z = 0;

  // arm marker template
  visualization_msgs::Marker arm;
  arm.header.stamp = ros::Time();
  arm.header.frame_id = "/hummingbird/base_link";
  arm.ns = "vehicle_arm";
  arm.action = visualization_msgs::Marker::ADD;
  arm.type = visualization_msgs::Marker::CUBE;
  arm.scale.x = arm_len * marker_scale_;
  arm.scale.y = 0.02 * marker_scale_;
  arm.scale.z = 0.01 * marker_scale_;
  arm.color.r = 0.0;
  arm.color.g = 0.0;
  arm.color.b = 1.0;
  arm.color.a = 1.0;
  arm.pose.position.z = -0.015 * marker_scale_;

  float angle_increment = 2 * M_PI / num_rotors;

  for (float angle = angle_increment / 2; angle <= (2 * M_PI);
       angle += angle_increment) {
    rotor.pose.position.x = arm_len * cos(angle) * marker_scale_;
    rotor.pose.position.y = arm_len * sin(angle) * marker_scale_;
    rotor.id++;

    arm.pose.position.x = rotor.pose.position.x / 2;
    arm.pose.position.y = rotor.pose.position.y / 2;
    arm.pose.orientation = tf::createQuaternionMsgFromYaw(angle);
    arm.id++;

    vehicle_marker_->markers.push_back(rotor);
    vehicle_marker_->markers.push_back(arm);
  }

  // body marker template
  visualization_msgs::Marker body;
  body.header.stamp = ros::Time();
  body.header.frame_id = "/hummingbird/base_link";
  body.ns = "vehicle_body";
  body.action = visualization_msgs::Marker::ADD;
  body.type = visualization_msgs::Marker::CUBE;
  body.scale.x = body_width * marker_scale_;
  body.scale.y = body_width * marker_scale_;
  body.scale.z = body_height * marker_scale_;
  body.color.r = 0.0;
  body.color.g = 1.0;
  body.color.b = 0.0;
  body.color.a = 0.8;

  vehicle_marker_->markers.push_back(body);
}

}  // namespace visualizer
