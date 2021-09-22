#pragma once

#include <Eigen/Eigen>
#include <random>

#include "quadrotor_common/quad_state_estimate.h"
#include "quadrotor_common/trajectory.h"
#include "rpg_mpc/mpc_controller.h"

#include "agile_autonomy_utils/logging.h"
#include "agile_autonomy_utils/trajectory_ext.h"
#include "traj_sampler/kdtree.h"
#include "traj_sampler/timing.hpp"

namespace traj_sampler {

using T = double;

class TrajSampler {
 public:
  TrajSampler(const int traj_len, const double traj_dt, const double rand_theta,
              const double rand_phi, const bool verbose = false);

  TrajSampler() : TrajSampler(10, 0.1, 0.15, 0.2) {}

  ~TrajSampler();

  void setStateEstimate(const Eigen::Vector3d &pos, const Eigen::Vector3d &vel,
                        const Eigen::Vector3d &acc,
                        const Eigen::Quaterniond &att);

  bool setReferenceFromTrajectory(
      const quadrotor_common::Trajectory &trajectory);

  void getRolloutData(const TrajectoryExt &rollout);

  void computeLabelBSplineSampling(
      const int idx, const std::string directory,
      const std::shared_ptr<KdTreeSampling> kd_tree, const int bspline_anchors,
      const unsigned int continuity_order, const int max_steps_metropolis,
      const double max_threshold, const int save_n_best, const bool save_wf,
      const bool save_bf, quadrotor_common::Trajectory *const trajectory,
      const quadrotor_common::Trajectory &initial_guess =
          quadrotor_common::Trajectory());

  void initialize(const double *start_state);

  void createReferenceTrajectory(
      quadrotor_common::Trajectory *reference_trajectory);

  void computeCost(const double *state_array, const double *reference_states,
                   const double *input_array, const double *reference_inputs,
                   double *cost_array, double *accumulated_cost_array);

 private:
  bool createBSpline(const std::vector<double> &t_vec,
                     const std::vector<double> &x_vec,
                     const std::vector<double> &y_vec,
                     const std::vector<double> &z_vec,
                     TrajectoryExt *const bspline_traj);
  void sampleAnchorPoint(const Eigen::Vector3d &ref_pos,
                         const double &rand_theta, const double &rand_phi,
                         double *const anchor_pos_x, double *const anchor_pos_y,
                         double *const anchor_pos_z);
  unsigned int traj_len_;
  double traj_dt_;
  double rand_theta_;
  double rand_phi_;

  std::random_device random_device_;
  std::default_random_engine generator_;

  logging::Logging logging_helper_;

  double Q_xy_ = 100.0;
  double Q_z_ = 300.0;
  double Q_att_ = 0.0;
  double Q_vel_ = 0.0;

  Eigen::Matrix<double, 13, 1> state_estimate_;

  double *state_array_h_;
  double *input_array_h_;
  double *reference_states_h_;
  double *reference_inputs_h_;
  double *cost_array_h_;
  double *accumulated_cost_array_h_;

  bool verbose_ = false;
};

}  // namespace traj_sampler
