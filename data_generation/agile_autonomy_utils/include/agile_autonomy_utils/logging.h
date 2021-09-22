#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include "Eigen/Eigen"
#include "agile_autonomy_utils/trajectory_ext.h"
#include "quadrotor_common/quad_state_estimate.h"
#include "quadrotor_common/trajectory.h"

namespace logging {
class Logging {
 public:
  Logging();
  virtual ~Logging();

  void save_rollout_to_csv(const std::vector<TrajectoryExt>& rollouts,
                           const int traj_len, const int save_n_best,
                           const double max_threshold,
                           const std::string states_filename_wf,
                           const std::string states_filename_bf,
                           const std::string coeff_filename_wf,
                           const std::string coeff_filename_bf,
                           const bool verbose = false);

  void save_nw_pred_to_csv(const TrajectoryExt network_prediction,
                           const std::string filename);

  void createDirectories(const std::string data_dir,
                         std::string* curr_data_dir);
  void saveTrajectorytoCSV(const std::string& csv_filename,
                           const quadrotor_common::Trajectory& trajectory);
  void newOdometryLog(const std::string& filename);
  void closeOdometryLog();
  bool logOdometry(const quadrotor_common::QuadStateEstimate& state_estimate,
                   const quadrotor_common::TrajectoryPoint& curr_reference,
                   const ros::Time& time_start_logging,
                   const int reference_progress, const double& cam_pitch_angle);

 private:
  struct StreamWithFilename {
    std::ofstream filestream;
    std::string filename;
  };
  StreamWithFilename log_file_odometry_;
  StreamWithFilename log_file_states_wf_;
  StreamWithFilename log_file_states_bf_;
  StreamWithFilename log_file_polycoeffs_wf_;
  StreamWithFilename log_file_polycoeffs_bf_;
  void save_trajectory_to_csv(StreamWithFilename& log_file,
                              const TrajectoryExt& rollout,
                              const FrameID frame_id);
  void save_polycoeffs_to_csv(StreamWithFilename& log_file,
                              const TrajectoryExt& rollout,
                              const FrameID frame_id);
  void write_states_csv_header(StreamWithFilename& log_file,
                               const std::string& csv_filename,
                               const int traj_len);
  void write_coeff_csv_header(StreamWithFilename& log_file,
                              const std::string& csv_filename,
                              const int poly_order);
  void writeOdometryHeader();
};

}  // namespace logging
