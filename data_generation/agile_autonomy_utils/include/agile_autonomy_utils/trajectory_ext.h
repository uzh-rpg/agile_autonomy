#pragma once

#include <Eigen/Dense>
#include "quadrotor_common/trajectory.h"

struct TrajectoryExtPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d position;
  Eigen::Vector3d velocity;
  Eigen::Vector3d acceleration;
  Eigen::Vector3d jerk;
  Eigen::Vector3d snap;
  Eigen::Quaterniond attitude;
  Eigen::Vector3d bodyrates;
  double collective_thrust;
  double time_from_start;
};

enum class FrameID { Body, World };

class TrajectoryExt {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  TrajectoryExt();
  TrajectoryExt(const quadrotor_common::Trajectory& trajectory,
                 const FrameID frame_id,
                 const quadrotor_common::TrajectoryPoint& ref_odometry);
  ~TrajectoryExt();
  FrameID getFrameID() const { return frame_id_; };

  void convertToFrame(const FrameID frame_id);
  void convertToFrame(const FrameID frame_id, const Eigen::Vector3d& position,
                      const Eigen::Quaterniond& attitude);
  void fitPolynomialCoeffs(const unsigned int poly_order,
                           const unsigned int continuity_order);
  void scaleTime(double time_scale);
  void resamplePointsFromPolyCoeffs();
  void setPolynomialCoeffs(const double* coeff_x, const double* coeff_y,
                           const double* coeff_z, const unsigned int order);
  void setPolynomialCoeffs(const std::vector<Eigen::Vector3d>& coeff);
  void setFrame(const FrameID frame_id);
  void setSampleTimes(const std::vector<double>& sample_times);
  void truncateBack(const double& desired_duration);
  void truncateFront(const double& time_now);
  void print(const std::string& traj_name) const;
  std::vector<TrajectoryExtPoint> getPoints() const;
  std::vector<Eigen::Vector3d> getPolyCoeffs() const;
  void setCost(double cost);
  double getCost() const;
  unsigned int getPolyOrder() const { return poly_order_; };
  unsigned int getContinuityOrder() { return continuity_order_; };
  void getTrajectory(quadrotor_common::Trajectory* trajectory);
  double computeControlCost();
  void clear();
  void enableYawing(const bool enable_yawing);
  void addPoint(const TrajectoryExtPoint& point);
  void recomputeTrajectory();
  void replaceFirstPoint(const TrajectoryExtPoint& first_point);
  bool setConstantArcLengthSpeed(const double& speed, const int& traj_len,
                                 const double& traj_dt);
  void translate(const Eigen::Vector3d& displacement);
  void pushPointFront(const TrajectoryExtPoint& first_point,
                      const double& traj_dt);

 private:
  void convertToBodyFrame();
  void convertToWorldFrame();
  void recomputeVelocity();
  void recomputeAcceleration();
  Eigen::Vector3d evaluatePoly(const double dt, const int derivative);
  FrameID frame_id_;
  bool yawing_enabled_ = false;
  std::vector<Eigen::Vector3d> poly_coeff_;
  double cost_;
  unsigned int poly_order_;
  unsigned int continuity_order_;
  std::vector<TrajectoryExtPoint> points_;
  double time_of_creation_ = 0.0;

  // The reference odometry is always expressed in world frame
  TrajectoryExtPoint reference_odometry_;
};

static bool compareTrajectories(TrajectoryExt t1, TrajectoryExt t2) {
  return (t1.getCost() < t2.getCost());
}
