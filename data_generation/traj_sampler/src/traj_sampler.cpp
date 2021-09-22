#include "traj_sampler/traj_sampler.h"

#include <Eigen/Dense>
#include <functional>

#include "agile_autonomy_utils/spline.h"
#include "rpg_common/pose.h"
#include "traj_sampler/kdtree.h"
#include "traj_sampler/timing.hpp"

namespace traj_sampler {

TrajSampler::TrajSampler(const int traj_len, const double traj_dt,
                         const double rand_theta, const double rand_phi,
                         const bool verbose)
    : traj_len_(traj_len),
      traj_dt_(traj_dt),
      rand_theta_(rand_theta),
      rand_phi_(rand_phi),
      verbose_(verbose) {
  // allocate host memory
  state_array_h_ = (double *)malloc(13 * (traj_len_ + 1) * sizeof(double));
  input_array_h_ = (double *)malloc(4 * traj_len_ * sizeof(double));
  reference_states_h_ = (double *)malloc(13 * (traj_len_ + 1) * sizeof(double));
  reference_inputs_h_ = (double *)malloc(4 * traj_len_ * sizeof(double));
  cost_array_h_ = (double *)malloc((traj_len_ + 1) * sizeof(double));
  accumulated_cost_array_h_ = (double *)malloc(sizeof(double));

  generator_.seed(random_device_());
}

TrajSampler::~TrajSampler() {
  free(state_array_h_);
  free(input_array_h_);
  free(reference_states_h_);
  free(reference_inputs_h_);
  free(cost_array_h_);
  free(accumulated_cost_array_h_);
}

bool TrajSampler::setReferenceFromTrajectory(
    const quadrotor_common::Trajectory &trajectory) {
  auto iterator(trajectory.points.begin());
  ros::Duration t_start = trajectory.points.begin()->time_from_start;
  auto last_element = trajectory.points.end();
  last_element = std::prev(last_element);
  double eps = 1.0e-5;
  for (int j = 0; j <= traj_len_; j++) {
    while ((iterator->time_from_start - t_start).toSec() <=
               (j * traj_dt_ - eps) &&
           iterator != last_element) {
      iterator++;
    }
    // Position
    reference_states_h_[13 * j + 0] = iterator->position.x();
    reference_states_h_[13 * j + 1] = iterator->position.y();
    reference_states_h_[13 * j + 2] = iterator->position.z();
    // Velocity
    reference_states_h_[13 * j + 3] = iterator->velocity.x();
    reference_states_h_[13 * j + 4] = iterator->velocity.y();
    reference_states_h_[13 * j + 5] = iterator->velocity.z();
    // Acceleration
    reference_states_h_[13 * j + 6] = iterator->acceleration.x();
    reference_states_h_[13 * j + 7] = iterator->acceleration.y();
    reference_states_h_[13 * j + 8] = iterator->acceleration.z();
    // Attitude
    reference_states_h_[13 * j + 9] = iterator->orientation.w();
    reference_states_h_[13 * j + 10] = iterator->orientation.x();
    reference_states_h_[13 * j + 11] = iterator->orientation.y();
    reference_states_h_[13 * j + 12] = iterator->orientation.z();
    // Input
    if (j < traj_len_) {
      reference_inputs_h_[4 * j + 0] =
          (iterator->acceleration - Eigen::Vector3d(0.0, 0.0, -9.81)).norm();
      reference_inputs_h_[4 * j + 1] = iterator->bodyrates.x();
      reference_inputs_h_[4 * j + 2] = iterator->bodyrates.y();
      reference_inputs_h_[4 * j + 3] = iterator->bodyrates.z();
    }
  }
}

void TrajSampler::setStateEstimate(const Eigen::Vector3d &pos,
                                   const Eigen::Vector3d &vel,
                                   const Eigen::Vector3d &acc,
                                   const Eigen::Quaterniond &att) {
  // This function assumes the linear velocity estimate to be in world frame
  double state_estimate[13] = {
      static_cast<double>(pos.x()), static_cast<double>(pos.y()),
      static_cast<double>(pos.z()), static_cast<double>(vel.x()),
      static_cast<double>(vel.y()), static_cast<double>(vel.z()),
      static_cast<double>(acc.x()), static_cast<double>(acc.y()),
      static_cast<double>(acc.z()), static_cast<double>(att.w()),
      static_cast<double>(att.x()), static_cast<double>(att.y()),
      static_cast<double>(att.z())};
  state_estimate_ = (Eigen::Matrix<double, 13, 1>() << pos.x(), pos.y(),
                     pos.z(), vel.x(), vel.y(), vel.z(), acc.x(), acc.y(),
                     acc.z(), att.w(), att.x(), att.y(), att.z())
                        .finished();
  initialize(state_estimate);
}

void TrajSampler::initialize(const double *start_state) {
  // Position
  state_array_h_[0] = start_state[0];
  state_array_h_[1] = start_state[1];
  state_array_h_[2] = start_state[2];
  // Velocity
  state_array_h_[3] = start_state[3];
  state_array_h_[4] = start_state[4];
  state_array_h_[5] = start_state[5];
  // Acceleration
  state_array_h_[6] = start_state[6];
  state_array_h_[7] = start_state[7];
  state_array_h_[8] = start_state[8];
  // Attitude
  state_array_h_[9] = start_state[9];
  state_array_h_[10] = start_state[10];
  state_array_h_[11] = start_state[11];
  state_array_h_[12] = start_state[12];
}

void TrajSampler::computeCost(const double *state_array,
                              const double *reference_states,
                              const double *input_array,
                              const double *reference_inputs,
                              double *cost_array,
                              double *accumulated_cost_array) {
  double exponent = 2.0;
  for (int j = 0; j <= traj_len_; j++) {
    cost_array[j] =
        Q_xy_ *
            abs(std::pow(state_array[13 * j + 0] - reference_states[13 * j + 0],
                         exponent)) +
        Q_xy_ *
            abs(std::pow(state_array[13 * j + 1] - reference_states[13 * j + 1],
                         exponent)) +
        Q_z_ *
            abs(std::pow(state_array[13 * j + 2] - reference_states[13 * j + 2],
                         exponent)) +
        Q_vel_ *
            abs(std::pow(state_array[13 * j + 3] - reference_states[13 * j + 3],
                         exponent)) +
        Q_vel_ *
            abs(std::pow(state_array[13 * j + 4] - reference_states[13 * j + 4],
                         exponent)) +
        Q_vel_ *
            abs(std::pow(state_array[13 * j + 5] - reference_states[13 * j + 5],
                         exponent)) +
        Q_att_ *
            abs(std::pow(state_array[13 * j + 9] - reference_states[13 * j + 9],
                         exponent)) +
        Q_att_ * abs(std::pow(
                     state_array[13 * j + 10] - reference_states[13 * j + 10],
                     exponent)) +
        Q_att_ * abs(std::pow(
                     state_array[13 * j + 11] - reference_states[13 * j + 11],
                     exponent)) +
        Q_att_ * abs(std::pow(
                     state_array[13 * j + 12] - reference_states[13 * j + 12],
                     exponent));
  }
  double temp_sum = 0.0;
  for (int j = 0; j <= traj_len_; j++) {
    temp_sum += cost_array[j];
  }
  accumulated_cost_array[0] = temp_sum / static_cast<double>(traj_len_);
}

void TrajSampler::sampleAnchorPoint(const Eigen::Vector3d &ref_pos,
                                    const double &rand_theta,
                                    const double &rand_phi,
                                    double *const anchor_pos_x,
                                    double *const anchor_pos_y,
                                    double *const anchor_pos_z) {
  double radius = ref_pos.norm();
  double ref_theta = std::acos(ref_pos.z() / radius);
  double ref_phi = std::atan2(ref_pos.y(), ref_pos.x());

  // we sample anchor points in spherical coordinates in the body frame
  std::uniform_real_distribution<double> theta_dist =
      std::uniform_real_distribution<double>(ref_theta - rand_theta,
                                             ref_theta + rand_theta);

  std::uniform_real_distribution<double> phi_dist =
      std::uniform_real_distribution<double>(ref_phi - rand_phi,
                                             ref_phi + rand_phi);

  // convert to cartesian coordinates
  double theta = theta_dist(generator_);
  double phi = phi_dist(generator_);

  *anchor_pos_x = radius * std::sin(theta) * std::cos(phi);
  *anchor_pos_y = radius * std::sin(theta) * std::sin(phi);
  *anchor_pos_z = radius * std::cos(theta);
}

void TrajSampler::getRolloutData(const TrajectoryExt &rollout) {
  int j = 0;
  for (auto traj_point : rollout.getPoints()) {
    if (j > traj_len_) {
      ROS_ERROR("Overstepping state array!");
    }
    double dt = j * traj_dt_;
    state_array_h_[13 * j + 0] = static_cast<double>(traj_point.position.x());
    state_array_h_[13 * j + 1] = static_cast<double>(traj_point.position.y());
    state_array_h_[13 * j + 2] = static_cast<double>(traj_point.position.z());
    state_array_h_[13 * j + 3] = static_cast<double>(traj_point.velocity.x());
    state_array_h_[13 * j + 4] = static_cast<double>(traj_point.velocity.y());
    state_array_h_[13 * j + 5] = static_cast<double>(traj_point.velocity.z());
    state_array_h_[13 * j + 6] =
        static_cast<double>(traj_point.acceleration.x());
    state_array_h_[13 * j + 7] =
        static_cast<double>(traj_point.acceleration.y());
    state_array_h_[13 * j + 8] =
        static_cast<double>(traj_point.acceleration.z());
    state_array_h_[13 * j + 9] = static_cast<double>(traj_point.attitude.x());
    state_array_h_[13 * j + 10] = static_cast<double>(traj_point.attitude.y());
    state_array_h_[13 * j + 11] = static_cast<double>(traj_point.attitude.w());
    state_array_h_[13 * j + 12] = static_cast<double>(traj_point.attitude.z());
    if (j < traj_len_) {
      input_array_h_[4 * j + 0] =
          static_cast<double>(traj_point.collective_thrust);
      input_array_h_[4 * j + 2] = static_cast<double>(traj_point.bodyrates.y());
      input_array_h_[4 * j + 3] = static_cast<double>(traj_point.bodyrates.z());
      input_array_h_[4 * j + 1] = static_cast<double>(traj_point.bodyrates.x());
    }
    j += 1;
  }
}

void TrajSampler::createReferenceTrajectory(
    quadrotor_common::Trajectory *reference_trajectory) {
  reference_trajectory->points.clear();
  for (int idx_point = 0; idx_point <= traj_len_; idx_point++) {
    quadrotor_common::TrajectoryPoint traj_point;
    traj_point.time_from_start = ros::Duration(idx_point * traj_dt_);
    traj_point.position =
        Eigen::Vector3d(reference_states_h_[13 * idx_point + 0],
                        reference_states_h_[13 * idx_point + 1],
                        reference_states_h_[13 * idx_point + 2]);
    traj_point.velocity =
        Eigen::Vector3d(reference_states_h_[13 * idx_point + 3],
                        reference_states_h_[13 * idx_point + 4],
                        reference_states_h_[13 * idx_point + 5]);
    traj_point.acceleration =
        Eigen::Vector3d(reference_states_h_[13 * idx_point + 6],
                        reference_states_h_[13 * idx_point + 7],
                        reference_states_h_[13 * idx_point + 8]);

    reference_trajectory->points.push_back(traj_point);
  }
}

bool TrajSampler::createBSpline(const std::vector<double> &t_vec,
                                const std::vector<double> &x_vec,
                                const std::vector<double> &y_vec,
                                const std::vector<double> &z_vec,
                                TrajectoryExt *const bspline_traj) {
  bspline_traj->clear();

  if (t_vec.size() != x_vec.size() || t_vec.size() != y_vec.size() ||
      t_vec.size() != z_vec.size()) {
    return false;
  }

  Timing timing_spline;
  timing_spline.tic();

  tk::spline x_spline, y_spline, z_spline;
  x_spline.set_points(t_vec, x_vec);
  y_spline.set_points(t_vec, y_vec);
  z_spline.set_points(t_vec, z_vec);

  // sample the spline, compute trajectory from it
  for (int i = 0; i <= traj_len_; i++) {
    TrajectoryExtPoint point;
    point.time_from_start = traj_dt_ * i;
    point.position = Eigen::Vector3d(x_spline(point.time_from_start),
                                     y_spline(point.time_from_start),
                                     z_spline(point.time_from_start));
    point.velocity = Eigen::Vector3d(x_spline.deriv(1, point.time_from_start),
                                     y_spline.deriv(1, point.time_from_start),
                                     z_spline.deriv(1, point.time_from_start));
    point.acceleration =
        Eigen::Vector3d(x_spline.deriv(2, point.time_from_start),
                        y_spline.deriv(2, point.time_from_start),
                        z_spline.deriv(2, point.time_from_start));

    point.attitude = Eigen::Quaterniond::Identity();

    // Attitude
    Eigen::Vector3d thrust =
        point.acceleration + 9.81 * Eigen::Vector3d::UnitZ();
    double dt = 0.05;
    Eigen::Vector3d thrust_before =
        Eigen::Vector3d(x_spline.deriv(2, point.time_from_start - dt),
                        y_spline.deriv(2, point.time_from_start - dt),
                        z_spline.deriv(2, point.time_from_start - dt));
    Eigen::Vector3d thrust_after =
        Eigen::Vector3d(x_spline.deriv(2, point.time_from_start + dt),
                        y_spline.deriv(2, point.time_from_start + dt),
                        z_spline.deriv(2, point.time_from_start + dt));

    Eigen::Vector3d I_eZ_I(0.0, 0.0, 1.0);
    Eigen::Quaterniond q_pitch_roll =
        Eigen::Quaterniond::FromTwoVectors(I_eZ_I, thrust);

    Eigen::Vector3d linvel_body = q_pitch_roll.inverse() * point.velocity;
    double heading = std::atan2(point.velocity.y(), point.velocity.x());

    Eigen::Quaterniond q_heading = Eigen::Quaterniond(
        Eigen::AngleAxisd(heading, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q_att = q_pitch_roll * q_heading;
    q_att.normalize();
    point.attitude = q_att;

    // Inputs
    point.collective_thrust = thrust.norm();
    thrust_before.normalize();
    thrust_after.normalize();
    Eigen::Vector3d crossProd = thrust_before.cross(thrust_after);
    Eigen::Vector3d angular_rates_wf = Eigen::Vector3d::Zero();
    if (crossProd.norm() > 0.0) {
      angular_rates_wf =
          std::acos(
              std::min(1.0, std::max(-1.0, thrust_before.dot(thrust_after)))) /
          dt * crossProd / (crossProd.norm() + 1.e-5);
    }
    point.bodyrates = q_att.inverse() * angular_rates_wf;

    bspline_traj->addPoint(point);
  }

  timing_spline.toc();
  return true;
}

void TrajSampler::computeLabelBSplineSampling(
    const int idx, const std::string directory,
    const std::shared_ptr<KdTreeSampling> kd_tree, const int bspline_anchors,
    const unsigned int continuity_order, const int max_steps_metropolis,
    const double max_threshold, const int save_n_best, const bool save_wf,
    const bool save_bf, quadrotor_common::Trajectory *const trajectory,
    const quadrotor_common::Trajectory &initial_guess) {
  quadrotor_common::TrajectoryPoint state_estimate_point;
  state_estimate_point.position =
      Eigen::Vector3d(static_cast<double>(state_estimate_[0]),
                      static_cast<double>(state_estimate_[1]),
                      static_cast<double>(state_estimate_[2]));
  state_estimate_point.velocity =
      Eigen::Vector3d(static_cast<double>(state_estimate_[3]),
                      static_cast<double>(state_estimate_[4]),
                      static_cast<double>(state_estimate_[5]));
  Eigen::Vector3d state_estimate_point_acc =
      Eigen::Vector3d(static_cast<double>(state_estimate_[6]),
                      static_cast<double>(state_estimate_[7]),
                      static_cast<double>(state_estimate_[8]));
  state_estimate_point.orientation =
      Eigen::Quaterniond(static_cast<double>(state_estimate_[9]),
                         static_cast<double>(state_estimate_[10]),
                         static_cast<double>(state_estimate_[11]),
                         static_cast<double>(state_estimate_[12]));

  std::vector<TrajectoryExt> rollouts;
  TrajectoryExt temp_rollout;
  temp_rollout.setCost(std::numeric_limits<double>::max());

  quadrotor_common::Trajectory reference_trajectory;
  // this reference contains as first point the current state estimate
  createReferenceTrajectory(&reference_trajectory);
  TrajectoryExt reference_ext(reference_trajectory, FrameID::World,
                              state_estimate_point);

  ///////////////////////////////////////////////////////////////////////////
  // main loop for Metropolis-Hastings
  ///////////////////////////////////////////////////////////////////////////
  Timing<double> timing_metropolis;
  timing_metropolis.tic();
  double best_cost = temp_rollout.getCost();
  std::uniform_real_distribution<double> accept_dist =
      std::uniform_real_distribution<double>(0.0, 1.0);

  std::vector<double> t_vec, x_vec, y_vec, z_vec;
  std::vector<double> t_vec_prev, x_vec_prev, y_vec_prev, z_vec_prev;
  double prev_cost = std::numeric_limits<double>::max();
  double rand_theta = 0.0;
  double rand_phi = 0.0;
  for (int step = 0; step < max_steps_metropolis; step++) {
    TrajectoryExt cand_rollout(reference_trajectory, FrameID::Body,
                               state_estimate_point);
    t_vec.clear();
    x_vec.clear();
    y_vec.clear();
    z_vec.clear();
    t_vec_prev.clear();
    x_vec_prev.clear();
    y_vec_prev.clear();
    z_vec_prev.clear();
    t_vec.push_back(0.0);
    x_vec.push_back(0.0);
    y_vec.push_back(0.0);
    z_vec.push_back(0.0);

    double anchor_dt = (traj_dt_ * traj_len_) / bspline_anchors;
    double anchor_px, anchor_py, anchor_pz;

    for (int anchor_idx = 1; anchor_idx <= bspline_anchors; anchor_idx++) {
      if (x_vec_prev.empty()) {
        rpg::Pose T_W_S = rpg::Pose(state_estimate_point.position,
                                    state_estimate_point.orientation);
        rpg::Pose T_W_C =
            rpg::Pose(reference_trajectory
                          .getStateAtTime(ros::Duration(anchor_idx * anchor_dt))
                          .position,
                      Eigen::Quaterniond::Identity());

        rpg::Pose T_W_C_ig = rpg::Pose(
            initial_guess.getStateAtTime(ros::Duration(anchor_idx * anchor_dt))
                .position,
            Eigen::Quaterniond::Identity());
        rpg::Pose T_S_C = T_W_S.inverse() * T_W_C;
        rpg::Pose T_S_C_ig = T_W_S.inverse() * T_W_C_ig;

        sampleAnchorPoint(
            T_S_C_ig.getPosition().normalized() * T_S_C.getPosition().norm(),
            rand_theta, rand_phi, &anchor_px, &anchor_py, &anchor_pz);
        // add sampled anchor point
        t_vec.push_back(anchor_idx * anchor_dt);
        x_vec.push_back(anchor_px);
        y_vec.push_back(anchor_py);
        z_vec.push_back(anchor_pz);
      } else {
        sampleAnchorPoint(
            Eigen::Vector3d(x_vec_prev[anchor_idx], y_vec_prev[anchor_idx],
                            z_vec_prev[anchor_idx]),
            rand_theta, rand_phi, &anchor_px, &anchor_py, &anchor_pz);
        t_vec.push_back(anchor_idx * anchor_dt);
        x_vec.push_back(anchor_px);
        y_vec.push_back(anchor_py);
        z_vec.push_back(anchor_pz);
      }
    }

    createBSpline(t_vec, x_vec, y_vec, z_vec, &cand_rollout);

    if (step > 0 && (step - 1) % (max_steps_metropolis / 3) == 0) {
      // we can stop early if we already found good trajectories
      std::sort(rollouts.begin(), rollouts.end(), compareTrajectories);
      if (rollouts.size() >= save_n_best &&
          rollouts.at(save_n_best - 1).getCost() < 100.0) {
        if (directory != "" && verbose_) {
          printf(
              "\nFound %d trajectories with cost lower than %.3f, stopping "
              "early.\n",
              save_n_best, 100.0);
        }
        break;
      }
      // increase the width of the sampling distribution
      rand_theta += rand_theta_;
      rand_phi += rand_phi_;
    }

    cand_rollout.enableYawing(true);
    cand_rollout.convertToFrame(FrameID::World, state_estimate_point.position,
                                state_estimate_point.orientation);

    cand_rollout.recomputeTrajectory();
    getRolloutData(cand_rollout);

    // compute cost for each trajectory
    computeCost(state_array_h_, reference_states_h_, input_array_h_,
                reference_inputs_h_, cost_array_h_, accumulated_cost_array_h_);
    int query_every_nth_point = 1;
    bool in_collision =
        kd_tree->query_kdtree(state_array_h_, accumulated_cost_array_h_,
                              traj_len_, query_every_nth_point, false);

    if (in_collision) {
      // bad sample, start with new one
      continue;
    }

    // since the continuity is enforced at the first point, we replace it with
    // the current state estimate
    TrajectoryExtPoint state_est_plus;
    state_est_plus.time_from_start = 0.0;
    state_est_plus.position = state_estimate_point.position;
    state_est_plus.attitude = state_estimate_point.orientation;
    state_est_plus.velocity = state_estimate_point.velocity;
    state_est_plus.acceleration = state_estimate_point_acc;
    cand_rollout.replaceFirstPoint(state_est_plus);
    cand_rollout.fitPolynomialCoeffs(8, 1);
    cand_rollout.resamplePointsFromPolyCoeffs();
    cand_rollout.recomputeTrajectory();
    cand_rollout.replaceFirstPoint(state_est_plus);
    getRolloutData(cand_rollout);

    computeCost(state_array_h_, reference_states_h_, input_array_h_,
                reference_inputs_h_, cost_array_h_, accumulated_cost_array_h_);
    kd_tree->query_kdtree(state_array_h_, accumulated_cost_array_h_, traj_len_,
                          query_every_nth_point, true);
    cand_rollout.setCost(static_cast<double>(accumulated_cost_array_h_[0]));

    // accept/reject sample
    double curr_cost = cand_rollout.getCost();
    double alpha = std::min(1.0, (std::exp(-0.01 * curr_cost) + 1.0e-7) /
                                     (std::exp(-0.01 * prev_cost) + 1.0e-7));

    double random_sample = accept_dist(generator_);

    bool accept = random_sample <= alpha;
    if (accept) {
      x_vec_prev = x_vec;
      y_vec_prev = y_vec;
      z_vec_prev = z_vec;
      prev_cost = curr_cost;
    }

    rollouts.push_back(cand_rollout);
  }
  timing_metropolis.toc();
  if (verbose_) {
    std::printf("rollouts.size() = %d\n", static_cast<int>(rollouts.size()));
  }

  if (rollouts.empty()) return;

  std::sort(rollouts.begin(), rollouts.end(), compareTrajectories);

  // save trajectories to disk
  std::ostringstream ss;
  ss << std::setw(8) << std::setfill('0') << idx;
  std::string idx_str(ss.str());

  std::string states_filename_wf = "";
  std::string states_filename_bf = "";
  std::string states_filename_ga = "";

  if (save_wf) {
    states_filename_wf =
        directory + "/trajectories/trajectories_wf_" + idx_str + ".csv";
  }
  if (save_bf) {
    states_filename_bf =
        directory + "/trajectories/trajectories_bf_" + idx_str + ".csv";
  }
  std::string coeff_filename_wf = "";
  std::string coeff_filename_bf = "";
  std::string coeff_filename_ga = "";
  logging_helper_.save_rollout_to_csv(rollouts, traj_len_, save_n_best,
                                      static_cast<double>(max_threshold),
                                      states_filename_wf, states_filename_bf,
                                      coeff_filename_wf, coeff_filename_bf);
}

}  // namespace traj_sampler
