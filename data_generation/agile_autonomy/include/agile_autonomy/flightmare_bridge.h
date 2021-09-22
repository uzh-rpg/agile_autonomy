#pragma once

#include "geometry_msgs/PoseStamped.h"
#include "quadrotor_common/quad_state_estimate.h"
#include "quadrotor_common/trajectory.h"
#include "ros/ros.h"
#include "sgm_gpu/sgm_gpu.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Empty.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Float32MultiArray.h"

#include <random>

// Flightmare dependencies
// rpgq simulator
#include <rpgq_simulator/implementation/objects/quadrotor_vehicle/quad_and_rgb_camera.h>
#include <rpgq_simulator/visualization/flightmare_bridge.hpp>
#include <rpgq_simulator/visualization/flightmare_message_types.hpp>

namespace flightmare_bridge {

class FlightmareBridge {
 public:
  FlightmareBridge(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  void spawnObjects(const quadrotor_common::TrajectoryPoint& start_state);
  void generatePointcloud(const Eigen::Ref<Eigen::Vector3d>& min_corner,
                          const Eigen::Ref<Eigen::Vector3d>& max_corner,
                          const std::string& curr_data_dir);
  void disconnect();
  void getImages(const quadrotor_common::QuadStateEstimate& state_estimate,
                 const std::string curr_data_dir, const int frame_counter);

 private:
  void treeSpacingCallback(const std_msgs::Float32ConstPtr& msg);
  void objectSpacingCallback(const std_msgs::Float32ConstPtr& msg);
  void removeObjectsCallback(const std_msgs::EmptyConstPtr& msg);
  void getImageFromUnity(
      const quadrotor_common::QuadStateEstimate& state_estimate,
      cv::Mat* left_frame, cv::Mat* right_frame, cv::Mat* gt_depth_frame);
  void computeDepthImage(const cv::Mat& left_frame, const cv::Mat& right_frame,
                         cv::Mat* const depth);
  void startPositionCallback(const nav_msgs::OdometryConstPtr& msg);

  bool loadParameters();

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber tree_spacing_sub_, object_spacing_sub_, gap_width_sub_,
      spawn_single_pole_sub_, remove_objects_sub_;

  // Quadrotor and Flightmare
  std::shared_ptr<RPGQ::Simulator::QuadrotorVehicle> quad_;
  std::shared_ptr<RPGQ::Simulator::RGBCamera> left_rgb_cam_;
  std::shared_ptr<RPGQ::Simulator::RGBCamera> right_rgb_cam_;

  std::shared_ptr<RPGQ::Simulator::FlightmareBridge> flightmareBridge_ptr_;
  RPGQ::FlightmareTypes::SceneID scene_id_;
  bool flightmare_ready_{false};
  bool unity_is_ready_{false};
  std::shared_ptr<RPGQ::Simulator::QuadStereoRGBCamera> quad_stereo_;
  double stereo_baseline_;
  double pitch_angle_deg_;
  double rgb_fov_deg_;
  int img_cols_, img_rows_;

  // Spawning objects and trees
  bool spawn_trees_;
  double avg_tree_spacing_;
  bool spawn_objects_;
  double avg_object_spacing_;
  double rand_width_;
  int seed_;
  int rollout_idx_ = 0;
  int env_idx_;
  std::vector<double> bounding_box_;
  std::vector<double> bounding_box_origin_;
  std::vector<double> min_object_scale_;
  std::vector<double> max_object_scale_;
  std::vector<double> min_object_angles_;
  std::vector<double> max_object_angles_;
  std::vector<std::string> object_names_;
  double pc_resolution_;
  bool perform_sgm_;

  // image transport and SGM stuff
  image_transport::Publisher unity_rgb_pub_;
  image_transport::Publisher unity_depth_pub_;
  image_transport::Publisher sgm_depth_pub_;

  std::shared_ptr<sgm_gpu::SgmGpu> sgm_;

  // Used for trees
  std::random_device random_device_;
  std::default_random_engine generator_;
};

}  // namespace flightmare_bridge
