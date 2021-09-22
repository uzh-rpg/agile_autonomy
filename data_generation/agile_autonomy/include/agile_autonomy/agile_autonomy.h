#pragma once

#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <fstream>
#include <mutex>
#include <random>
#include <shared_mutex>

#include "agile_autonomy_msgs/Bspline.h"
#include "agile_autonomy_msgs/MultiPolyCoeff.h"
#include "agile_autonomy_msgs/MultiTrajectory.h"
#include "agile_autonomy_msgs/PositionCommand.h"
#include "agile_autonomy_utils/logging.h"
#include "agile_autonomy_utils/trajectory_ext.h"
#include "agile_autonomy_utils/visualize.h"
#include "autopilot/autopilot_helper.h"
#include "nav_msgs/Odometry.h"
#include "quadrotor_common/trajectory.h"
#include "quadrotor_msgs/Trajectory.h"
#include "ros/ros.h"
#include "rpg_common/pose.h"
#include "rpg_mpc/mpc_controller.h"
#include "sensor_msgs/Image.h"
#include "sgm_gpu/sgm_gpu.h"
#include "state_predictor/state_predictor.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Empty.h"
#include "std_msgs/Float32.h"
#include "visualization_msgs/MarkerArray.h"

#include "agile_autonomy/flightmare_bridge.h"

namespace agile_autonomy {

class AgileAutonomy {
 public:
  AgileAutonomy(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

  AgileAutonomy() : AgileAutonomy(ros::NodeHandle(), ros::NodeHandle("~")) {}

  virtual ~AgileAutonomy();

 private:
  enum class StateMachine {
    kOff,
    kAutopilot,
    kExecuteExpert,
    kNetwork,
    kComputeLabels
  };
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  logging::Logging logging_helper_;
  std::shared_ptr<flightmare_bridge::FlightmareBridge> flightmare_bridge_;
  std::shared_ptr<visualizer::Visualizer> visualizer_;

  ros::Subscriber toggle_experiment_sub_;
  ros::Subscriber odometry_sub_;
  ros::Subscriber setup_logging_sub_;
  ros::Subscriber start_flying_sub_;
  ros::Subscriber traj_sub_;
  ros::Subscriber land_sub_;
  ros::Subscriber force_hover_sub_;
  ros::Subscriber off_sub_;
  ros::Subscriber completed_global_plan_sub_;

  ros::Publisher control_command_pub_;
  ros::Publisher start_flying_pub_;
  ros::Publisher ref_progress_pub_;
  ros::Publisher setpoint_pub_;
  ros::Publisher compute_global_path_pub_;

  ros::Timer save_timer_;

  void computeManeuver(const bool only_expert);

  void startExecutionCallback(const std_msgs::BoolConstPtr& msg);

  void setupLoggingCallback(const std_msgs::BoolConstPtr& msg);

  void stopFlyingCallback(const std_msgs::BoolConstPtr& msg);

  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);

  void saveLoop(const ros::TimerEvent& time);

  void trajectoryCallback(
      const agile_autonomy_msgs::MultiTrajectoryConstPtr& msg);

  void landCallback(const std_msgs::EmptyConstPtr& msg);

  void offCallback(const std_msgs::EmptyConstPtr& msg);

  void forceHoverCallback(const std_msgs::EmptyConstPtr& msg);

  quadrotor_common::QuadStateEstimate getPredictedStateEstimate(
      const ros::Time& time, const state_predictor::StatePredictor* predictor);

  void publishControlCommand(
      const quadrotor_common::ControlCommand& control_command);

  bool loadParameters();

  bool selectBestNetworkPrediction(
      const std::vector<quadrotor_common::Trajectory>& nw_trajectories,
      const Eigen::Vector3d& start_pos, const Eigen::Quaterniond& start_att,
      quadrotor_common::Trajectory* const selected_trajectory);

  double yawFromQuaternion(const Eigen::Quaterniond& q);

  void completedGlobalPlanCallback(const std_msgs::BoolConstPtr& msg);

  bool convertTrajectoriesToWorldFrame(
      const std::vector<quadrotor_common::Trajectory>& nw_trajectories,
      const rpg::Pose& T_W_S, const TrajectoryExt& prev_ref,
      std::vector<quadrotor_common::Trajectory>* world_trajectories,
      std::vector<double>* trajectory_costs);

  quadrotor_common::Trajectory acrobatic_trajectory_;
  int reference_progress_abs_;

  bool save_network_trajectories_ = true;
  bool unity_is_ready_ = false;
  bool enable_yawing_ = false;
  bool reference_ready_ = false;

  std::mutex odom_mtx_;
  std::mutex nw_pred_mtx_;
  std::mutex curr_ref_mtx_;

  std::string data_dir_;
  std::string curr_data_dir_;

  quadrotor_common::TrajectoryPoint curr_reference_;
  quadrotor_common::Trajectory prev_ref_traj_;
  ros::Time t_prev_ref_traj_;

  TrajectoryExt prev_solution_;

  // MPC controller variant
  rpg_mpc::MpcController<double> base_controller_ =
      rpg_mpc::MpcController<double>(ros::NodeHandle(), ros::NodeHandle("~"),
                                     "vio_mpc_path");
  rpg_mpc::MpcParams<double> base_controller_params_;
  state_predictor::StatePredictor state_predictor_;

  /////////////////////////////////////
  // Parameters
  /////////////////////////////////////

  // General
  bool velocity_estimate_in_world_frame_;
  double ctrl_cmd_delay_;
  FrameID nw_predictions_frame_id_;
  double test_time_velocity_;
  double test_time_max_z_;
  double test_time_min_z_;

  // Trajectory
  TrajectoryExt network_prediction_;
  ros::Time time_received_prediction_;
  bool received_network_prediction_ = false;
  int process_every_nth_odometry_;
  int odom_idx_ = 0;
  static constexpr int viz_id_start_ = 10;
  int viz_id_ = viz_id_start_;
  int num_traj_viz_;
  std::vector<double> sample_times_;
  std::vector<double> fine_sample_times_;

  // Maneuver
  int rollout_counter_ = 0;
  double length_straight_;
  double maneuver_velocity_;
  double save_freq_;

  bool perform_global_planning_;

  // Polynomial trajectory representation
  unsigned int traj_len_;
  double traj_dt_;
  int continuity_order_;

  StateMachine state_machine_ = StateMachine::kAutopilot;
  ros::Time time_start_logging_;
  quadrotor_common::QuadStateEstimate received_state_est_;
  double traj_sampling_freq_ = 50.0;
  double cam_pitch_angle_ = 0.0;

  // Data generation
  int frame_counter_ = 0;
  int pred_traj_idx_ = 0;

  bool only_expert_ = false;
  bool setup_done_ = false;
};

}  // namespace agile_autonomy
