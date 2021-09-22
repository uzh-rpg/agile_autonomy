#include "traj_sampler/generate_label.h"
#include <chrono>
#include <thread>

#include "agile_autonomy_utils/logging.h"
#include "traj_sampler/kdtree.h"

namespace generate_label {

GenerateLabel::GenerateLabel(const ros::NodeHandle& nh,
                             const ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {
  if (!loadParameters()) {
    ROS_ERROR("[%s] Failed to load all parameters",
              ros::this_node::getName().c_str());
    ros::shutdown();
  }

  perform_mpc_optim_sub_ = nh_.subscribe(
      "start_label", 1, &GenerateLabel::generateInitialGuesses, this);
  generate_label_sub_ = nh_.subscribe(
      "mpc_optim_completed", 1, &GenerateLabel::generateLabelCallback, this);
  thread_completed_sub_ = nh_.subscribe(
      "thread_completed", 1, &GenerateLabel::completedThreadCallback, this);
  mpc_completed_pub_ = nh_.advertise<std_msgs::Bool>("mpc_optim_completed", 1);
  completed_labelling_pub_ =
      nh_.advertise<std_msgs::Bool>("labelling_completed", 1);
  thread_completed_pub_ = nh_.advertise<std_msgs::Int8>("thread_completed", 1);
}

GenerateLabel::~GenerateLabel() {}

std::vector<std::string> GenerateLabel::getDirectories(const std::string& s) {
  std::vector<std::string> r;
  for (auto p = fs::recursive_directory_iterator(s);
       p != fs::recursive_directory_iterator(); ++p) {
    if (p->status().type() == fs::file_type::directory && p.depth() == 0) {
      fs::path p1 = p->path();
      p1 += "/trajectories/trajectories_bf_00000000_muted.csv";
      if (!fs::exists(p1)) {
        r.push_back(p->path().string());
        if (verbose_) {
          printf("Added Folder: %s\n", p->path().string().c_str());
        }
      }
    }
  }
  return r;
}

void GenerateLabel::generateInitialGuesses(const std_msgs::BoolConstPtr& msg) {
  if (thread_idx_ != 0) {
    return;
  }

  printf(
      "==============================\nGenerating initial guesses for "
      "sampling...\n==============================\n");
  quadrotor_common::Trajectory reference_trajectory;
  Eigen::MatrixXf reference_states;
  Eigen::MatrixXf reference_inputs;
  // horizon is hardcoded for rpg_mpc
  double* optimized_states =
      (double*)malloc(10 * (2 * traj_len_ + 1) * sizeof(double));
  double* optimized_inputs =
      (double*)malloc(4 * (2 * traj_len_) * sizeof(double));

  int num_labels = 0;
  bool first_state_in_traj;
  std::vector<std::string> directories = getDirectories(data_dir_);
  for (auto const& directory : directories) {
    int idx = 0;
    first_state_in_traj = true;
    // iterate over generated csv file, generate label for each row
    std::string odometry_filename = directory + "/odometry.csv";

    std::ifstream odometry_file;
    odometry_file.open(odometry_filename.c_str());

    if (!odometry_file.is_open()) {
      std::cout << "Did not find the following file!!" << std::endl;
      std::cout << odometry_filename << std::endl;
      std::cout << "Will skip folder" << std::endl;
      continue;
    }

    std::string reference_trajectory_filename;
    if (perform_global_planning_) {
      reference_trajectory_filename = directory + "/ellipsoid_trajectory.csv";
    } else {
      reference_trajectory_filename = directory + "/reference_trajectory.csv";
    }

    loadReferenceTrajectory(&reference_trajectory,
                            reference_trajectory_filename, verbose_);

    std::string odometry_line;
    // skip header line
    getline(odometry_file, odometry_line);
    int progress_idx = 0;

    while (getline(odometry_file, odometry_line) && idx < end_idx_) {
      if (odometry_line.empty() || idx < start_idx_) {
        idx += 1;
        continue;  // skip empty lines
      }

      std::istringstream iss(odometry_line);
      std::string lineStream;
      std::string::size_type sz;

      std::vector<double> row;
      while (getline(iss, lineStream, ',')) {
        row.push_back(stold(lineStream, &sz));  // convert to double
      }
      // start_time measures time since trajectory execution start
      ros::Duration start_time = ros::Duration(std::max(0.0, row[0]));

      quadrotor_common::TrajectoryPoint state_estimate;
      state_estimate.position = Eigen::Vector3d(row[1], row[2], row[3]);
      state_estimate.velocity = Eigen::Vector3d(row[4], row[5], row[6]);
      state_estimate.acceleration = Eigen::Vector3d(row[7], row[8], row[9]);
      state_estimate.orientation =
          Eigen::Quaterniond(row[10], row[11], row[12], row[13]);
      int progress_idx_saved = int(row[17]);
      double pitch_angle = double(row[18]);

      quadrotor_common::QuadStateEstimate quad_state_estimate;
      quad_state_estimate.position = state_estimate.position;
      quad_state_estimate.velocity = state_estimate.velocity;
      quad_state_estimate.orientation = state_estimate.orientation;

      if (verbose_) {
        printf("Idx: %d, computing label for t = %5.2f / %5.2f.\n", idx,
               start_time.toSec(),
               reference_trajectory.points.back().time_from_start.toSec());
        printf("Current position: %5.2f, %5.2f, %5.2f,.\n",
               state_estimate.position.x(), state_estimate.position.y(),
               state_estimate.position.z());
      }

      // based on current odometry, generate reference trajectory (just select
      // segment of reference trajectory)
      quadrotor_common::Trajectory horizon_reference;
      quadrotor_common::Trajectory optimized_reference;
      horizon_reference.trajectory_type =
          quadrotor_common::Trajectory::TrajectoryType::GENERAL;
      optimized_reference.trajectory_type =
          quadrotor_common::Trajectory::TrajectoryType::GENERAL;
      // sample with 50Hz for more than two seconds (excess reference will be
      // truncated)
      computeReferenceTrajectoryPosBased(
          state_estimate.position, reference_trajectory, traj_dt_ * traj_len_,
          &horizon_reference, &progress_idx);

      // first optimize reference using optimization-based MPC (that one
      // does not know obstacles) then, feed the adapted reference inputs to
      // sampler
      int n_mpc_iter = 50;
      ros::WallTime t_start_mpc_opt = ros::WallTime::now();
      if (perform_mpc_optimization_) {
        for (int mpc_iter = 0; mpc_iter < n_mpc_iter; mpc_iter++) {
          init_controller_.run(quad_state_estimate, horizon_reference,
                               init_controller_params_, first_state_in_traj,
                               optimized_states, optimized_inputs);
          first_state_in_traj = false;
        }

        for (int j = 0; j <= traj_len_; j++) {
          quadrotor_common::TrajectoryPoint point;
          // rpg_mpc orders states [pos, att, vel]
          point.time_from_start = ros::Duration(j * traj_dt_);
          point.position = Eigen::Vector3d(optimized_states[10 * j + 0],
                                           optimized_states[10 * j + 1],
                                           optimized_states[10 * j + 2]);
          point.velocity = Eigen::Vector3d(optimized_states[10 * j + 7],
                                           optimized_states[10 * j + 8],
                                           optimized_states[10 * j + 9]);
          point.orientation = Eigen::Quaterniond(
              optimized_states[10 * j + 3], optimized_states[10 * j + 4],
              optimized_states[10 * j + 5], optimized_states[10 * j + 6]);

          if (j < traj_len_) {
            point.bodyrates = Eigen::Vector3d(optimized_inputs[4 * j + 1],
                                              optimized_inputs[4 * j + 2],
                                              optimized_inputs[4 * j + 3]);
            point.acceleration = point.orientation * Eigen::Vector3d::UnitZ() *
                                     optimized_inputs[4 * j + 0] -
                                 9.81 * Eigen::Vector3d::UnitZ();
          }

          if (std::abs(point.orientation.coeffs().norm() - 1.0) > 1.0e-5) {
            ROS_WARN(
                "Fixing reference, seems MPC did not fully converge! (norm "
                "attitude: %.5f)",
                point.orientation.coeffs().norm());
            point.orientation.normalize();
          }
          optimized_reference.points.push_back(point);
        }
        if (verbose_) {
          std::printf("Completed %d MPC optimizations in %.3f seconds.\n",
                      n_mpc_iter,
                      (ros::WallTime::now() - t_start_mpc_opt).toSec());
        }
      }

      logging::Logging logging_helper;
      std::ostringstream ss;
      ss << std::setw(8) << std::setfill('0') << idx;
      std::string idx_str(ss.str());
      std::vector<TrajectoryExt> single_reference;
      quadrotor_common::TrajectoryPoint state_estimate_point;
      state_estimate_point.position = quad_state_estimate.position;
      state_estimate_point.orientation = quad_state_estimate.orientation;
      std::string states_filename_wf, states_filename_bf, coeff_filename_wf,
          coeff_filename_bf;
      // resample horizon_reference to match sampling frequency
      std::vector<double> sample_times;
      sample_times.clear();
      for (unsigned int i = 0; i <= traj_len_; i++) {
        sample_times.push_back(traj_dt_ * i);
      }
      // save reference trajectory to disk
      TrajectoryExt horizon_reference_ext(horizon_reference, FrameID::World,
                                           state_estimate_point);
      horizon_reference_ext.setCost(0.0);
      single_reference.clear();
      single_reference.push_back(horizon_reference_ext);
      states_filename_wf =
          directory + "/trajectories/trajectory_ref_wf_" + idx_str + ".csv";
      states_filename_bf = "";
      coeff_filename_wf = "";
      coeff_filename_bf = "";
      logging_helper.save_rollout_to_csv(single_reference, traj_len_, 1, 1.0,
                                         states_filename_wf, states_filename_bf,
                                         coeff_filename_wf, coeff_filename_bf);

      // save mpc optimization trajectory to disk
      TrajectoryExt optim_reference_ext;
      if (!perform_mpc_optimization_) {
        optim_reference_ext = horizon_reference_ext;
      } else {
        optim_reference_ext = TrajectoryExt(
            optimized_reference, FrameID::World, state_estimate_point);
      }
      optim_reference_ext.setCost(0.0);
      single_reference.clear();
      single_reference.push_back(optim_reference_ext);
      // we are only interested in the world frame one
      states_filename_wf =
          directory + "/trajectories/trajectory_mpc_opt_wf_" + idx_str + ".csv";
      states_filename_bf = "";
      coeff_filename_wf = "";
      coeff_filename_bf = "";
      logging_helper.save_rollout_to_csv(single_reference, traj_len_, 1, 1.0,
                                         states_filename_wf, states_filename_bf,
                                         coeff_filename_wf, coeff_filename_bf);
      idx += 1;
      num_labels++;
    }
  }

  std_msgs::Bool true_msg;
  true_msg.data = true;
  mpc_completed_pub_.publish(true_msg);
}

void GenerateLabel::generateLabelCallback(const std_msgs::BoolConstPtr& msg) {
  printf(
      "==============================\nLabelling data in directory "
      "%s\n==============================\n",
      data_dir_.c_str());
  std::vector<std::string> directories = getDirectories(data_dir_);
  num_completed_threads_ = 0;
  traj_sampler::TrajSampler traj_sampler(traj_len_, traj_dt_, rand_theta_,
                                         rand_phi_, verbose_);

  quadrotor_common::Trajectory reference_trajectory;
  Eigen::MatrixXf reference_states;
  Eigen::MatrixXf reference_inputs;

  // horizon is hardcoded for rpg_mpc
  double* optimized_states =
      (double*)malloc(10 * (traj_len_ + 1) * sizeof(double));
  double* optimized_inputs = (double*)malloc(4 * (traj_len_) * sizeof(double));

  ros::Time start_time = ros::Time::now();
  int num_labels = 0;
  int directory_idx = 0;
  for (auto const& directory : directories) {
    int idx = 0;
    ros::Duration start_time = ros::Duration(0.0);
    std::string odometry_filename = directory + "/odometry.csv";
    std::string pointcloud_filename = directory + "/pointcloud-unity.ply";

    std::ifstream odometry_file;
    odometry_file.open(odometry_filename.c_str());

    if (!odometry_file.is_open()) {
      std::cout << "Did not find the following file!!" << std::endl;
      std::cout << odometry_filename << std::endl;
      std::cout << "Will skip folder" << std::endl;
      continue;
    }

    std::string reference_trajectory_filename =
        directory + "/reference_trajectory.csv";

    loadReferenceTrajectory(&reference_trajectory,
                            reference_trajectory_filename, verbose_);

    // intialize kd-tree from saved point cloud
    std::shared_ptr<KdTreeSampling> kd_tree = std::make_shared<KdTreeSampling>(
        pointcloud_filename, crash_dist_, crash_penalty_, drone_dimensions_);

    std::string odometry_line;
    // skip header line
    getline(odometry_file, odometry_line);

    while (getline(odometry_file, odometry_line) && idx < end_idx_ &&
           ros::ok()) {
      if (odometry_line.empty() || idx < start_idx_) {
        idx += 1;
        continue;  // skip empty lines
      }
      if (idx % n_threads_ != thread_idx_) {
        idx += 1;
        continue;
      }

      std::istringstream iss(odometry_line);
      std::string lineStream;
      std::string::size_type sz;

      std::vector<double> row;
      while (getline(iss, lineStream, ',')) {
        row.push_back(stold(lineStream, &sz));  // convert to double
      }
      ros::Duration start_time = ros::Duration(std::max(0.0, row[0]));
      quadrotor_common::TrajectoryPoint state_estimate;
      state_estimate.position = Eigen::Vector3d(row[1], row[2], row[3]);
      state_estimate.velocity = Eigen::Vector3d(row[4], row[5], row[6]);
      state_estimate.acceleration = Eigen::Vector3d(row[7], row[8], row[9]);
      state_estimate.orientation =
          Eigen::Quaterniond(row[10], row[11], row[12], row[13]);
      int progress_idx_saved = int(row[17]);
      double cam_pitch_angle = row[18];

      quadrotor_common::QuadStateEstimate quad_state_estimate;
      quad_state_estimate.position = state_estimate.position;
      quad_state_estimate.velocity = state_estimate.velocity;
      quad_state_estimate.orientation = state_estimate.orientation;

      if (thread_idx_ == 0) {
        printf(
            "Directory [%d/%d] -- Idx: %d, computing label for t = %5.2f / "
            "%5.2f.\n",
            directory_idx, static_cast<int>(directories.size()), idx,
            start_time.toSec(),
            reference_trajectory.points.back().time_from_start.toSec());
      }
      // Set the reference for the sampler
      quadrotor_common::Trajectory selected_reference;
      // load initial guess from file
      std::ostringstream ss;
      ss << std::setw(8) << std::setfill('0') << idx;
      std::string idx_str(ss.str());
      std::string initial_guess_filename;
      if (perform_mpc_optimization_) {
        initial_guess_filename = directory +
                                 "/trajectories/trajectory_mpc_opt_wf_" +
                                 idx_str + ".csv";
      } else {
        initial_guess_filename =
            directory + "/trajectories/trajectory_ref_wf_" + idx_str + ".csv";
      }

      loadReferenceFromFile(&selected_reference, traj_dt_,
                            initial_guess_filename, verbose_);
      if (verbose_) {
        printf("Loaded reference of size [%lu] from file.\n",
               selected_reference.points.size());
      }
      traj_sampler.setReferenceFromTrajectory(selected_reference);

      // Set state estimate for sampler
      traj_sampler.setStateEstimate(
          state_estimate.position, state_estimate.velocity,
          state_estimate.acceleration, state_estimate.orientation);

      // perform fancy extensive sampling
      traj_sampler.computeLabelBSplineSampling(
          idx, directory, kd_tree, bspline_anchors_, continuity_order_,
          max_steps_metropolis_, save_max_cost_, save_n_best_, save_wf_,
          save_bf_, nullptr, selected_reference);

      idx += 1;
      num_labels++;
    }
    directory_idx++;
  }

  free(optimized_states);
  free(optimized_inputs);

  if (thread_idx_ == 0) {
    int waited_n_sec = 0;
    num_completed_threads_ += 1;
    while (waited_n_sec < 60 && num_completed_threads_ < n_threads_) {
      ROS_INFO("Waiting for other nodes to finish labelling...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
      waited_n_sec += 1;
    }
    ros::Time end_time = ros::Time::now();
    double duration = (end_time - start_time).toSec();
    printf("Completed label generation in %.2fs (%.3fs per label).\n", duration,
           duration / (num_labels * n_threads_));
    ROS_INFO("Sending labelling completed message!");

    std_msgs::Bool completed_labelling_msg;
    completed_labelling_msg.data = true;
    completed_labelling_pub_.publish(completed_labelling_msg);
  } else {
    std_msgs::Int8 thread_completed_msg;
    thread_completed_msg.data = thread_idx_;
    thread_completed_pub_.publish(thread_completed_msg);
  }
}

void GenerateLabel::completedThreadCallback(const std_msgs::Int8ConstPtr& msg) {
  if (thread_idx_ != 0) return;
  ROS_INFO("Received completion of thread #%d!", msg->data);
  num_completed_threads_ += 1;
}

bool GenerateLabel::loadParameters() {
  if (!pnh_.getParam("data_dir", data_dir_)) return false;
  int continuity_order;
  double drone_dim_x, drone_dim_y, drone_dim_z;
  if (!quadrotor_common::getParam("traj_len", traj_len_, 15)) return false;
  if (!quadrotor_common::getParam("continuity_order", continuity_order, 0))
    return false;
  if (!quadrotor_common::getParam("save_n_best", save_n_best_, 100))
    return false;
  if (!quadrotor_common::getParam("save_max_cost", save_max_cost_, 9000.0))
    return false;
  if (!quadrotor_common::getParam("traj_dt", traj_dt_, 0.1)) return false;
  if (!quadrotor_common::getParam("crash_dist", crash_dist_, 0.1)) return false;
  if (!quadrotor_common::getParam("crash_penalty", crash_penalty_, 9999.0))
    return false;
  if (!quadrotor_common::getParam("max_steps_metropolis", max_steps_metropolis_,
                                  10000))
    return false;
  if (!quadrotor_common::getParam("rand_theta", rand_theta_, 0.15))
    return false;
  if (!quadrotor_common::getParam("rand_phi", rand_phi_, 0.2)) return false;
  if (!quadrotor_common::getParam("drone_dim_x", drone_dim_x, 0.3))
    return false;
  if (!quadrotor_common::getParam("drone_dim_y", drone_dim_y, 0.3))
    return false;
  if (!quadrotor_common::getParam("drone_dim_z", drone_dim_z, 0.1))
    return false;
  if (!quadrotor_common::getParam("perform_mpc_optimization",
                                  perform_mpc_optimization_, false))
    return false;
  if (!quadrotor_common::getParam("perform_global_planning",
                                  perform_global_planning_, false))
    return false;

  if (!quadrotor_common::getParam("save_wf", save_wf_, false)) return false;
  if (!quadrotor_common::getParam("save_bf", save_bf_, false)) return false;

  if (!pnh_.getParam("n_threads", n_threads_)) return false;
  if (!pnh_.getParam("thread_idx", thread_idx_)) return false;

  // BSpline sampling parameters
  if (!quadrotor_common::getParam("bspline/n_anchors", bspline_anchors_, 3))
    return false;

  if (!quadrotor_common::getParam("verbose", verbose_, false)) return false;

  if (!quadrotor_common::getParam("start_idx", start_idx_, 0)) return false;

  if (!quadrotor_common::getParam("end_idx", end_idx_, -1)) return false;

  if (end_idx_ < 0) {
    end_idx_ = std::numeric_limits<int>::max();
  }

  continuity_order_ = static_cast<unsigned int>(continuity_order);
  drone_dimensions_ = Eigen::Vector3d(drone_dim_x, drone_dim_y, drone_dim_z);

  return true;
}
}  // namespace generate_label

int main(int argc, char** argv) {
  ros::init(argc, argv, "generate_label");
  generate_label::GenerateLabel generate_label;

  ros::MultiThreadedSpinner spinner(2);
  spinner.spin();

  return 0;
}
