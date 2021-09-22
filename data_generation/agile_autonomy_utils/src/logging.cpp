#include "agile_autonomy_utils/logging.h"
#include <algorithm>
#include <experimental/filesystem>

namespace logging {

Logging::Logging() = default;

Logging::~Logging() = default;

void Logging::save_rollout_to_csv(const std::vector<TrajectoryExt>& rollouts,
                                  const int traj_len, const int save_n_best,
                                  const double max_threshold,
                                  const std::string states_filename_wf,
                                  const std::string states_filename_bf,
                                  const std::string coeff_filename_wf,
                                  const std::string coeff_filename_bf,
                                  const bool verbose) {
  int counter = 0;
  int poly_order = static_cast<int>(rollouts.front().getPolyOrder());

  // write all csv headers
  write_states_csv_header(log_file_states_wf_, states_filename_wf, traj_len);
  write_states_csv_header(log_file_states_bf_, states_filename_bf, traj_len);
  write_coeff_csv_header(log_file_polycoeffs_wf_, coeff_filename_wf,
                         poly_order);
  write_coeff_csv_header(log_file_polycoeffs_bf_, coeff_filename_bf,
                         poly_order);

  int save_n_traj = std::min(static_cast<int>(rollouts.size()), save_n_best);
  for (int i = 0; i < save_n_traj; i++) {
    if (rollouts.at(i).getCost() > max_threshold) {
      if (verbose) {
        printf(
            "Trajectory not collision free (cost = %.2f, threshold = %.2f), "
            "did "
            "not save full number of "
            "trajectories!\n",
            rollouts.at(i).getCost(), max_threshold);
      }
      break;
    }
    if (verbose) {
      printf("Saving trajectory %d/%d.\r", i, save_n_traj);
      fflush(stdout);
    }
    save_trajectory_to_csv(log_file_states_wf_, rollouts.at(i), FrameID::World);
    save_polycoeffs_to_csv(log_file_polycoeffs_wf_, rollouts.at(i),
                           FrameID::World);
    save_trajectory_to_csv(log_file_states_bf_, rollouts.at(i), FrameID::Body);
    save_polycoeffs_to_csv(log_file_polycoeffs_bf_, rollouts.at(i),
                           FrameID::Body);
    counter++;
  }

  if (max_threshold > 1.0 && verbose) {
    printf("Saved %d best trajectories (cost < %.2f) out of %d.\n", counter,
           max_threshold, static_cast<int>(rollouts.size()));
  }
  log_file_states_wf_.filestream.close();
  log_file_states_bf_.filestream.close();
  log_file_polycoeffs_wf_.filestream.close();
  log_file_polycoeffs_bf_.filestream.close();
}

void Logging::save_trajectory_to_csv(StreamWithFilename& log_file,
                                     const TrajectoryExt& rollout,
                                     const FrameID frame_id) {
  if (!log_file.filestream.is_open()) {
    return;
  }
  TrajectoryExt rollout_temp = rollout;
  if (frame_id != rollout_temp.getFrameID()) {
    rollout_temp.convertToFrame(frame_id);
    rollout_temp.fitPolynomialCoeffs(rollout_temp.getPolyOrder(),
                                     rollout_temp.getContinuityOrder());
  }
  for (auto point : rollout_temp.getPoints()) {
    if (point.collective_thrust == 0.0) {
      point.collective_thrust =
          (point.acceleration + 9.81 * Eigen::Vector3d::UnitZ()).norm();
    }

    // clang-format off
    log_file.filestream << std::fixed
                        << std::setprecision(8) << point.position.x() << ","
                        << std::setprecision(8) << point.position.y() << ","
                        << std::setprecision(8) << point.position.z() << ","
                        << std::setprecision(8) << point.velocity.x() << ","
                        << std::setprecision(8) << point.velocity.y() << ","
                        << std::setprecision(8) << point.velocity.z() << ","
                        << std::setprecision(8) << point.acceleration.x() << ","
                        << std::setprecision(8) << point.acceleration.y() << ","
                        << std::setprecision(8) << point.acceleration.z() << ","
                        << std::setprecision(8) << point.attitude.w() << ","
                        << std::setprecision(8) << point.attitude.x() << ","
                        << std::setprecision(8) << point.attitude.y() << ","
                        << std::setprecision(8) << point.attitude.z() << ","
                        << std::setprecision(8) << point.bodyrates.x() << ","
                        << std::setprecision(8) << point.bodyrates.y() << ","
                        << std::setprecision(8) << point.bodyrates.z() << ","
                        << std::setprecision(8) << point.collective_thrust << ",";
    // clang-format on
  }
  log_file.filestream << std::fixed << std::setprecision(8)
                      << rollout_temp.getCost() << "\n";
}

void Logging::save_polycoeffs_to_csv(StreamWithFilename& log_file,
                                     const TrajectoryExt& rollout,
                                     const FrameID frame_id) {
  if (!log_file.filestream.is_open()) {
    return;
  }
  TrajectoryExt rollout_temp = rollout;

  if (frame_id != rollout_temp.getFrameID()) {
    rollout_temp.convertToFrame(frame_id);
    rollout_temp.fitPolynomialCoeffs(rollout_temp.getPolyOrder(),
                                     rollout_temp.getContinuityOrder());
  }

  for (auto polycoeff : rollout_temp.getPolyCoeffs()) {
    // clang-format off
    log_file.filestream << std::fixed
                        << std::setprecision(8) << polycoeff.x() << ","
                        << std::setprecision(8) << polycoeff.y() << ","
                        << std::setprecision(8) << polycoeff.z() << ",";
    // clang-format on
  }
  log_file.filestream << std::setprecision(8) << rollout_temp.getCost() << "\n";
}
void Logging::write_states_csv_header(StreamWithFilename& log_file,
                                      const std::string& csv_filename,
                                      const int traj_len) {
  if (csv_filename == "") {
    return;
  }
  log_file.filename = csv_filename;
  log_file.filestream.open(log_file.filename,
                           std::ofstream::out | std::ofstream::trunc);

  for (int j = 0; j < traj_len + 1; j++) {
    log_file.filestream << "pos_x_" << j << ","
                        << "pos_y_" << j << ","
                        << "pos_z_" << j << ","
                        << "vel_x_" << j << ","
                        << "vel_y_" << j << ","
                        << "vel_z_" << j << ","
                        << "acc_x_" << j << ","
                        << "acc_y_" << j << ","
                        << "acc_z_" << j << ","
                        << "q_w_" << j << ","
                        << "q_x_" << j << ","
                        << "q_y_" << j << ","
                        << "q_z_" << j << ","
                        << "br_x_" << j << ","
                        << "br_y_" << j << ","
                        << "br_z_" << j << ","
                        << "thrust_" << j << ",";
  }
  log_file.filestream << "rel_cost"
                      << "\n";
}

void Logging::write_coeff_csv_header(StreamWithFilename& log_file,
                                     const std::string& csv_filename,
                                     const int poly_order) {
  if (csv_filename == "") {
    return;
  }
  log_file.filename = csv_filename;
  log_file.filestream.open(log_file.filename,
                           std::ofstream::out | std::ofstream::trunc);

  for (int j = 0; j <= poly_order; j++) {
    log_file.filestream << "coeff_x_" << j << ","
                        << "coeff_y_" << j << ","
                        << "coeff_z_" << j << ",";
  }
  log_file.filestream << "rel_cost"
                      << "\n";
}

void Logging::save_nw_pred_to_csv(const TrajectoryExt network_prediction,
                                  const std::string filename) {
  if (network_prediction.getFrameID() != FrameID::World) {
    std::printf(
        "Network predictions need to be converted to world frame before "
        "saving!\n");
  }
  StreamWithFilename stream;
  stream.filestream.open(filename, std::ios_base::app | std::ios_base::in);
  int traj_len = static_cast<int>(network_prediction.getPoints().size());
  write_states_csv_header(stream, filename, traj_len);
  save_trajectory_to_csv(stream, network_prediction, FrameID::World);
  stream.filestream.close();
}

void Logging::createDirectories(const std::string data_dir,
                                std::string* curr_data_dir) {
  std::printf("Creating directories in [%s]\n", data_dir.c_str());
  // create directory
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%y-%m-%d_%H-%M-%S");
  auto time_str = oss.str();
  std::string rollout_dir = data_dir + "/rollout_" + time_str;

  std::experimental::filesystem::create_directory(rollout_dir);
  std::experimental::filesystem::create_directory(rollout_dir + "/img");
  std::experimental::filesystem::create_directory(rollout_dir +
                                                  "/trajectories");
  std::experimental::filesystem::create_directory(rollout_dir + "/plots");

  *curr_data_dir = rollout_dir;
}

void Logging::saveTrajectorytoCSV(
    const std::string& csv_filename,
    const quadrotor_common::Trajectory& trajectory) {
  printf("Saving trajectory to CSV. \n");
  StreamWithFilename reference_trajectory_file;
  reference_trajectory_file.filename = csv_filename;
  printf("Trajectory filename: %s\n",
         reference_trajectory_file.filename.c_str());
  reference_trajectory_file.filestream.open(
      reference_trajectory_file.filename,
      std::ios_base::out | std::ios_base::trunc);
  // write header
  // clang-format off
  reference_trajectory_file.filestream << "time_from_start" << ","
                                       << "pos_x" << ","
                                       << "pos_y" << ","
                                       << "pos_z" << ","
                                       << "vel_x" << ","
                                       << "vel_y" << ","
                                       << "vel_z" << ","
                                       << "acc_x" << ","
                                       << "acc_y" << ","
                                       << "acc_z" << ","
                                       << "jerk_x" << ","
                                       << "jerk_y" << ","
                                       << "jerk_z" << ","
                                       << "snap_x" << ","
                                       << "snap_y" << ","
                                       << "snap_z" << ","
                                       << "q_w" << ","
                                       << "q_x" << ","
                                       << "q_y" << ","
                                       << "q_z" << ","
                                       << "omega_x" << ","
                                       << "omega_y" << ","
                                       << "omega_z" << ","
                                       << "angular_acc_x" << ","
                                       << "angular_acc_y" << ","
                                       << "angular_acc_z" << ","
                                       << "angular_jerk_x" << ","
                                       << "angular_jerk_y" << ","
                                       << "angular_jerk_z" << ","
                                       << "angular_snap_x" << ","
                                       << "angular_snap_y" << ","
                                       << "angular_snap_z" << ","
                                       << "heading" << ","
                                       << "heading_rate" << ","
                                       << "heading_acc" << "\n";
  // clang-format on

  for (auto point : trajectory.points) {
    // save trajectory point
    Eigen::Quaternion<double> q_pose =
        Eigen::Quaterniond(point.orientation.w(), point.orientation.x(),
                           point.orientation.y(), point.orientation.z());
    Eigen::Quaternion<double> q_heading =
        Eigen::Quaternion<double>(Eigen::AngleAxis<double>(
            point.heading, Eigen::Matrix<double, 3, 1>::UnitZ()));
    Eigen::Quaternion<double> q_orientation = q_pose * q_heading;

    double thrust =
        (point.acceleration + Eigen::Vector3d(0.0, 0.0, 9.81)).norm();
    double acceleration = point.acceleration.norm();

    // save current odometry & time to disk
    // clang-format off
    reference_trajectory_file.filestream
        << std::fixed
        << std::setprecision(8) << (point.time_from_start).toSec() << ","
        << std::setprecision(8) << point.position.x() << ","
        << std::setprecision(8) << point.position.y() << ","
        << std::setprecision(8) << point.position.z() << ","
        << std::setprecision(8) << point.velocity.x() << ","
        << std::setprecision(8) << point.velocity.y() << ","
        << std::setprecision(8) << point.velocity.z() << ","
        << std::setprecision(8) << point.acceleration.x() << ","
        << std::setprecision(8) << point.acceleration.y() << ","
        << std::setprecision(8) << point.acceleration.z() << ","
        << std::setprecision(8) << point.jerk.x() << ","
        << std::setprecision(8) << point.jerk.y() << ","
        << std::setprecision(8) << point.jerk.z() << ","
        << std::setprecision(8) << point.snap.x() << ","
        << std::setprecision(8) << point.snap.y() << ","
        << std::setprecision(8) << point.snap.z() << ","
        << std::setprecision(8) << q_orientation.w() << ","
        << std::setprecision(8) << q_orientation.x() << ","
        << std::setprecision(8) << q_orientation.y() << ","
        << std::setprecision(8) << q_orientation.z() << ","
        << std::setprecision(8) << point.bodyrates.x() << ","
        << std::setprecision(8) << point.bodyrates.y() << ","
        << std::setprecision(8) << point.bodyrates.z() << ","
        << std::setprecision(8) << point.angular_acceleration.x() << ","
        << std::setprecision(8) << point.angular_acceleration.y() << ","
        << std::setprecision(8) << point.angular_acceleration.z() << ","
        << std::setprecision(8) << point.angular_jerk.x() << ","
        << std::setprecision(8) << point.angular_jerk.y() << ","
        << std::setprecision(8) << point.angular_jerk.z() << ","
        << std::setprecision(8) << point.angular_snap.x() << ","
        << std::setprecision(8) << point.angular_snap.y() << ","
        << std::setprecision(8) << point.angular_snap.z() << ","
        << std::setprecision(8) << point.heading << ","
        << std::setprecision(8) << point.heading_rate << ","
        << std::setprecision(8) << point.heading_acceleration << "\n";
    // clang-format on
  }
  reference_trajectory_file.filestream.close();
  printf("Saved trajectory to file.\n");
}

void Logging::newOdometryLog(const std::string& filename) {
  log_file_odometry_.filename = filename;
  log_file_odometry_.filestream.open(log_file_odometry_.filename,
                                     std::ios_base::app | std::ios_base::in);
  writeOdometryHeader();
  printf("Opened new odometry file: %s\n", log_file_odometry_.filename.c_str());
}

void Logging::closeOdometryLog() {
  printf("Close Odometry File.\n");
  if (log_file_odometry_.filestream.is_open()) {
    log_file_odometry_.filestream.close();
  }
}

void Logging::writeOdometryHeader() {
  // clang-format off
  log_file_odometry_.filestream << "time_from_start" << ","
                                << "pos_x" << ","
                                << "pos_y" << ","
                                << "pos_z" << ","
                                << "vel_x" << ","
                                << "vel_y" << ","
                                << "vel_z" << ","
                                << "acc_x" << ","
                                << "acc_y" << ","
                                << "acc_z" << ","
                                << "q_w" << ","
                                << "q_x" << ","
                                << "q_y" << ","
                                << "q_z" << ","
                                << "omega_x" << ","
                                << "omega_y" << ","
                                << "omega_z" << ","
                                << "reference_progress" << ","
                                << "pitch_angle" << "\n";
  // clang-format on
}

bool Logging::logOdometry(
    const quadrotor_common::QuadStateEstimate& state_estimate,
    const quadrotor_common::TrajectoryPoint& curr_reference,
    const ros::Time& time_start_logging, const int reference_progress,
    const double& cam_pitch_angle) {
  if (log_file_odometry_.filestream.is_open()) {
    // clang-format off
    log_file_odometry_.filestream << std::fixed
                                  << std::setprecision(8) << (state_estimate.timestamp - time_start_logging).toSec() << ","
                                  << std::setprecision(8) << state_estimate.position.x() << ","
                                  << std::setprecision(8) << state_estimate.position.y() << ","
                                  << std::setprecision(8) << state_estimate.position.z() << ","
                                  << std::setprecision(8) << state_estimate.velocity.x() << ","
                                  << std::setprecision(8) << state_estimate.velocity.y() << ","
                                  << std::setprecision(8) << state_estimate.velocity.z() << ","
                                  << std::setprecision(8) << curr_reference.acceleration.x() << ","
                                  << std::setprecision(8) << curr_reference.acceleration.y() << ","
                                  << std::setprecision(8) << curr_reference.acceleration.z() << ","
                                  << std::setprecision(8) << state_estimate.orientation.w() << ","
                                  << std::setprecision(8) << state_estimate.orientation.x() << ","
                                  << std::setprecision(8) << state_estimate.orientation.y() << ","
                                  << std::setprecision(8) << state_estimate.orientation.z() << ","
                                  << std::setprecision(8) << state_estimate.bodyrates.x() << ","
                                  << std::setprecision(8) << state_estimate.bodyrates.y() << ","
                                  << std::setprecision(8) << state_estimate.bodyrates.z() << ","
                                  << reference_progress << ","
                                  << std::setprecision(8) << cam_pitch_angle << "\n";
    log_file_odometry_.filestream.flush();
    // clang-format on
    return true;
  }
  return false;
}

}  // namespace logging
