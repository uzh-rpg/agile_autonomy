#include "agile_autonomy_utils/trajectory_ext.h"
#include <iostream>
#include "rpg_common/pose.h"

TrajectoryExt::TrajectoryExt() {
  poly_order_ = 4;
  frame_id_ = FrameID::World;
  reference_odometry_ = TrajectoryExtPoint();
}

TrajectoryExt::TrajectoryExt(
    const quadrotor_common::Trajectory &trajectory, const FrameID frame_id,
    const quadrotor_common::TrajectoryPoint &ref_odometry)
    : frame_id_(frame_id) {
  poly_order_ = 4;
  continuity_order_ = 1;  // 0: position, 1: velocity, 2: acceleration, 3: jerk

  reference_odometry_.position = ref_odometry.position;
  reference_odometry_.velocity = ref_odometry.velocity;
  reference_odometry_.acceleration = ref_odometry.acceleration;
  reference_odometry_.attitude = ref_odometry.orientation;

  double first_timestamp = trajectory.points.front().time_from_start.toSec();
  for (auto point : trajectory.points) {
    TrajectoryExtPoint new_point;
    new_point.time_from_start = point.time_from_start.toSec() - first_timestamp;
    new_point.position = point.position;
    new_point.velocity = point.velocity;
    new_point.acceleration = point.acceleration;
    new_point.jerk = point.jerk;
    new_point.snap = point.snap;

    new_point.attitude = point.orientation;
    new_point.bodyrates = point.bodyrates;
    new_point.collective_thrust = 0.0;

    points_.push_back(new_point);
  }
}

TrajectoryExt::~TrajectoryExt() {}

void TrajectoryExt::convertToFrame(const FrameID frame_id) {
  if (frame_id_ == frame_id) {
    return;
  }

  if (frame_id_ != FrameID::World) {
    printf(
        "ERROR, cannot convert from frame other than world frame without "
        "information about "
        "reference pose!\n");
    return;
  }

  switch (frame_id) {
    case FrameID::World: {
      printf(
          "ERROR, cannot convert to world frame without information about "
          "reference pose!\n");
      break;
    }
    case FrameID::Body: {
      convertToBodyFrame();
      break;
    }
  }
}

void TrajectoryExt::convertToFrame(const FrameID frame_id,
                                    const Eigen::Vector3d &position,
                                    const Eigen::Quaterniond &attitude) {
  if (frame_id_ == frame_id) {
    return;
  }
  reference_odometry_.position = position;
  reference_odometry_.attitude = attitude;

  switch (frame_id) {
    case FrameID::World: {
      convertToWorldFrame();
      break;
    }
    case FrameID::Body: {
      convertToBodyFrame();
      break;
    }
  }
}

void TrajectoryExt::convertToBodyFrame() {
  //  printf("converting to bodyframe...\n");
  if (frame_id_ == FrameID::Body) {
    return;
  }

  rpg::Pose T_W_S;
  switch (frame_id_) {
    case FrameID::World: {
      T_W_S =
          rpg::Pose(reference_odometry_.position, reference_odometry_.attitude);
      break;
    }
  }

  //  std::cout << "refpos: " << reference_odometry_.position << std::endl;
  //  std::cout << "refatt: " << reference_odometry_.attitude.coeffs() <<
  //  std::endl;
  for (auto &point : points_) {
    //    printf("time from start: %.2f, norm attitude: %.6f\n",
    //           point.time_from_start, point.attitude.coeffs().norm());
    //    std::cout << "position: " << point.position << std::endl;
    //    std::cout << "attitude: " << point.attitude.coeffs() << std::endl;
    rpg::Pose T_W_C = rpg::Pose(point.position, point.attitude);
    rpg::Pose T_S_C = T_W_S.inverse() * T_W_C;
    Eigen::Vector3d linvel_bf =
        T_W_S.getEigenQuaternion().inverse() * point.velocity;
    Eigen::Vector3d linacc_bf =
        T_W_S.getEigenQuaternion().inverse() * point.acceleration;
    Eigen::Vector3d linjerk_bf =
        T_W_S.getEigenQuaternion().inverse() * point.jerk;
    Eigen::Vector3d linsnap_bf =
        T_W_S.getEigenQuaternion().inverse() * point.snap;

    point.position = T_S_C.getPosition();
    point.attitude = T_S_C.getEigenQuaternion();
    point.velocity = linvel_bf;
    point.acceleration = linacc_bf;
    point.jerk = linjerk_bf;
    point.snap = linsnap_bf;
  }
  frame_id_ = FrameID::Body;
}

void TrajectoryExt::convertToWorldFrame() {
  if (frame_id_ == FrameID::World) {
    return;
  }

  rpg::Pose T_W_S;
  switch (frame_id_) {
    case FrameID::Body: {
      T_W_S =
          rpg::Pose(reference_odometry_.position, reference_odometry_.attitude);
      break;
    }
  }

  for (auto &point : points_) {
    rpg::Pose T_S_C = rpg::Pose(point.position, Eigen::Quaterniond::Identity());
    rpg::Pose T_W_C = T_W_S * T_S_C;  // only used for position
    Eigen::Vector3d linvel_wf = T_W_S.getEigenQuaternion() * point.velocity;
    Eigen::Vector3d linacc_wf = T_W_S.getEigenQuaternion() * point.acceleration;
    Eigen::Vector3d linjerk_wf = T_W_S.getEigenQuaternion() * point.jerk;
    Eigen::Vector3d linsnap_wf = T_W_S.getEigenQuaternion() * point.snap;

    // compute attitude
    Eigen::Vector3d thrust = linacc_wf + 9.81 * Eigen::Vector3d::UnitZ();
    Eigen::Vector3d I_eZ_I(0.0, 0.0, 1.0);
    Eigen::Quaterniond q_pitch_roll =
        Eigen::Quaterniond::FromTwoVectors(I_eZ_I, thrust);

    Eigen::Vector3d linvel_world = q_pitch_roll * point.velocity;
    double heading = 0.0;
    if (yawing_enabled_) {
      heading = std::atan2(linvel_wf.y(), linvel_wf.x());
    }

    Eigen::Quaterniond q_heading = Eigen::Quaterniond(
        Eigen::AngleAxisd(heading, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q_att = q_pitch_roll * q_heading;
    q_att.normalize();
    point.attitude = q_att;

    // Inputs
    point.collective_thrust = thrust.norm();
    double time_step = 0.1;
    Eigen::Vector3d thrust_1 = thrust - time_step / 2.0 * linjerk_wf;
    Eigen::Vector3d thrust_2 = thrust + time_step / 2.0 * linjerk_wf;
    thrust_1.normalize();
    thrust_2.normalize();
    Eigen::Vector3d crossProd = thrust_1.cross(thrust_2);
    Eigen::Vector3d angular_rates_wf = Eigen::Vector3d::Zero();
    if (crossProd.norm() > 0.0) {
      angular_rates_wf =
          std::acos(std::min(1.0, std::max(-1.0, thrust_1.dot(thrust_2)))) /
          time_step * crossProd / (crossProd.norm() + 1.0e-5);
    }
    point.bodyrates = q_att.inverse() * angular_rates_wf;

    point.position = T_W_C.getPosition();
    point.velocity = linvel_wf;
    point.acceleration = linacc_wf;
    point.jerk = linjerk_wf;
    point.snap = linsnap_wf;
  }
  frame_id_ = FrameID::World;
}

void TrajectoryExt::setFrame(const FrameID frame_id) { frame_id_ = frame_id; }

/// fits a polynomial to a sequence of points, continuity constraints are
/// enforced with respect to the first point
void TrajectoryExt::fitPolynomialCoeffs(const unsigned int poly_order,
                                         const unsigned int continuity_order) {
  poly_order_ = poly_order;
  continuity_order_ = continuity_order;
  poly_coeff_.clear();

  if (points_.front().time_from_start != 0.0) {
    std::printf(
        "Cannot fit polynomial when first point does not start from zero!\n");
    return;
  }

  double scaling[4] = {1.0, 1.0, 0.5, 1.0 / 6.0};
  for (int i = 0; i <= poly_order_; i++) {
    poly_coeff_.push_back(Eigen::Vector3d::Zero());
  }
  // constraint at beginning
  for (int axis = 0; axis < 3; axis++) {
    for (unsigned int cont_idx = 0; cont_idx <= continuity_order_; cont_idx++) {
      switch (cont_idx) {
        case 0: {
          poly_coeff_.at(cont_idx)[axis] =
              scaling[cont_idx] * points_.front().position[axis];
          break;
        }
        case 1: {
          poly_coeff_.at(cont_idx)[axis] =
              scaling[cont_idx] * points_.front().velocity[axis];
          if (points_.front().velocity.norm() < 0.1 && axis == 0) {
            poly_coeff_.at(cont_idx)[axis] += 0.2;
          }
          break;
        }
        case 2: {
          poly_coeff_.at(cont_idx)[axis] =
              scaling[cont_idx] * points_.front().acceleration[axis];
          break;
        }
        case 3: {
          poly_coeff_.at(cont_idx)[axis] =
              scaling[cont_idx] * points_.front().jerk[axis];
          break;
        }
      }
    }
  }
  if (points_.size() < 3) {
    printf("Trajectory of length [%lu] too short to compute meaningful fit!\n",
           points_.size());
    return;
  }

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(points_.size() - 1,
                                            poly_order_ - continuity_order_);
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(points_.size() - 1, 1);

  for (int axis = 0; axis < 3; axis++) {
    for (int i = 1; i < points_.size(); i++) {
      double t = points_.at(i).time_from_start;
      for (unsigned int j = continuity_order_ + 1; j <= poly_order_; j++) {
        A(i - 1, j - (continuity_order_ + 1)) =
            std::pow(t, static_cast<double>(j));
      }
      b(i - 1) = points_.at(i).position[axis];
      for (unsigned int cont_idx = 0; cont_idx <= continuity_order_;
           cont_idx++) {
        b(i - 1) -= poly_coeff_.at(cont_idx)[axis] *
                    std::pow(t, static_cast<double>(cont_idx));
      }
    }
    Eigen::MatrixXd x_coeff =
        A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    for (unsigned int j = (continuity_order_ + 1); j <= poly_order_; j++) {
      poly_coeff_.at(j)[axis] = x_coeff(j - (continuity_order_ + 1));
    }
  }
}

void TrajectoryExt::setPolynomialCoeffs(const double *coeff_x,
                                         const double *coeff_y,
                                         const double *coeff_z,
                                         const unsigned int order) {
  poly_order_ = order;
  poly_coeff_.clear();
  for (int i = 0; i <= poly_order_; i++) {
    poly_coeff_.push_back(Eigen::Vector3d(static_cast<double>(coeff_x[i]),
                                          static_cast<double>(coeff_y[i]),
                                          static_cast<double>(coeff_z[i])));
  }
}

void TrajectoryExt::setPolynomialCoeffs(
    const std::vector<Eigen::Vector3d> &coeff) {
  poly_order_ = coeff.size() - 1;
  poly_coeff_.clear();
  for (int i = 0; i <= poly_order_; i++) {
    poly_coeff_.push_back(coeff.at(i));
  }
}

void TrajectoryExt::truncateBack(const double &desired_duration) {
  while (points_.back().time_from_start > desired_duration) {
    points_.pop_back();
  }
}

void TrajectoryExt::truncateFront(const double &time_now) {
  while ((time_of_creation_ + points_.front().time_from_start) < time_now) {
    points_.erase(points_.begin());
  }

  // update the times for the remaining points
  time_of_creation_ = time_now;
  for (auto &point : points_) {
    point.time_from_start -= points_.front().time_from_start;
  }
}

void TrajectoryExt::replaceFirstPoint(const TrajectoryExtPoint &first_point) {
  if (frame_id_ != FrameID::World) {
    std::printf(
        "Can only replace first point when expressed in world frame!\n");
    return;
  }
  points_.at(0) = first_point;
}

void TrajectoryExt::pushPointFront(const TrajectoryExtPoint &first_point,
                                    const double &traj_dt) {
  for (auto &point : points_) {
    point.time_from_start += traj_dt;
  }
  points_.insert(points_.begin(), 1, first_point);
}

void TrajectoryExt::scaleTime(double time_scale) {
  if (time_scale > 1.0) {
    printf(
        "Can only slow down trajectories, not speeding them up! Clipping time "
        "remapping to [1.0].\n");
    time_scale = 1.0;
  }
  for (int i = 0; i <= poly_order_; i++) {
    poly_coeff_.at(i) *= std::pow(time_scale, i);
  }
}

void TrajectoryExt::resamplePointsFromPolyCoeffs() {
  for (auto &traj_point : points_) {
    traj_point.position = evaluatePoly(traj_point.time_from_start, 0);
    traj_point.velocity = evaluatePoly(traj_point.time_from_start, 1);
    traj_point.acceleration = evaluatePoly(traj_point.time_from_start, 2);
    traj_point.jerk = evaluatePoly(traj_point.time_from_start, 3);

    if (frame_id_ != FrameID::World) {
      continue;
    }
    // Attitude
    Eigen::Vector3d thrust =
        traj_point.acceleration + 9.81 * Eigen::Vector3d::UnitZ();
    Eigen::Vector3d I_eZ_I(0.0, 0.0, 1.0);
    Eigen::Quaterniond q_pitch_roll =
        Eigen::Quaterniond::FromTwoVectors(I_eZ_I, thrust);

    Eigen::Vector3d linvel_body = q_pitch_roll.inverse() * traj_point.velocity;
    double heading = 0.0;
    if (yawing_enabled_) {
      heading = std::atan2(traj_point.velocity.y(), traj_point.velocity.x());
    }

    Eigen::Quaterniond q_heading = Eigen::Quaterniond(
        Eigen::AngleAxisd(heading, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q_att = q_pitch_roll * q_heading;
    q_att.normalize();
    traj_point.attitude = q_att;

    // Inputs
    traj_point.collective_thrust = thrust.norm();
    double time_step = 0.1;
    Eigen::Vector3d thrust_1 = thrust - time_step / 2.0 * traj_point.jerk;
    Eigen::Vector3d thrust_2 = thrust + time_step / 2.0 * traj_point.jerk;
    thrust_1.normalize();
    thrust_2.normalize();
    Eigen::Vector3d crossProd = thrust_1.cross(thrust_2);
    Eigen::Vector3d angular_rates_wf = Eigen::Vector3d::Zero();
    if (crossProd.norm() > 0.0) {
      angular_rates_wf =
          std::acos(std::min(1.0, std::max(-1.0, thrust_1.dot(thrust_2)))) /
          time_step * crossProd / (crossProd.norm() + 1.0e-5);
    }
    traj_point.bodyrates = q_att.inverse() * angular_rates_wf;
  }
}

void TrajectoryExt::translate(const Eigen::Vector3d &displacement) {
  for (auto &point : points_) {
    point.position += displacement;
  }
  // also needs to adapt the polynomial coefficients
  poly_coeff_.at(0) += displacement;
}

void TrajectoryExt::setSampleTimes(const std::vector<double> &sample_times) {
  points_.clear();
  for (auto sample_time : sample_times) {
    TrajectoryExtPoint traj_point;
    traj_point.time_from_start = sample_time;
    points_.push_back(traj_point);
  }
}

void TrajectoryExt::setCost(double cost) { cost_ = cost; }

double TrajectoryExt::getCost() const { return cost_; }

std::vector<TrajectoryExtPoint> TrajectoryExt::getPoints() const {
  return points_;
}

std::vector<Eigen::Vector3d> TrajectoryExt::getPolyCoeffs() const {
  return poly_coeff_;
}

void TrajectoryExt::getTrajectory(quadrotor_common::Trajectory *trajectory) {
  trajectory->points.clear();
  for (auto point : points_) {
    quadrotor_common::TrajectoryPoint traj_point;
    traj_point.time_from_start = ros::Duration(point.time_from_start);
    traj_point.position = point.position;
    traj_point.velocity = point.velocity;
    traj_point.acceleration = point.acceleration;
    traj_point.jerk = point.jerk;
    traj_point.snap = point.snap;

    traj_point.orientation = point.attitude;
    traj_point.bodyrates = point.bodyrates;

    trajectory->points.push_back(traj_point);
  }
}

void TrajectoryExt::addPoint(const TrajectoryExtPoint &point) {
  points_.push_back(point);
}

void TrajectoryExt::print(const std::string &traj_name) const {
  printf("========\n");
  printf("%s\n", traj_name.c_str());
  switch (frame_id_) {
    case FrameID::World: {
      printf("Is in world frame\n");
      break;
    }
    case FrameID::Body: {
      printf("Is in body frame\n");
      break;
    }
    default: {
      printf("Unknown frame ID!\n");
      break;
    }
  }
  printf("Ref. Odom.: Pos: %.2f, %.2f, %.2f | Att: %.2f, %.2f, %.2f, %.2f\n",
         reference_odometry_.position.x(), reference_odometry_.position.y(),
         reference_odometry_.position.z(), reference_odometry_.attitude.w(),
         reference_odometry_.attitude.x(), reference_odometry_.attitude.y(),
         reference_odometry_.attitude.z());
  printf("Points: \n");
  for (auto point : points_) {
    printf("t = %.2f | ", point.time_from_start);
    printf("Pos: %.2f, %.2f, %.2f | ", point.position.x(), point.position.y(),
           point.position.z());
    printf("Att: %.2f, %.2f, %.2f, %.2f | ", point.attitude.w(),
           point.attitude.x(), point.attitude.y(), point.attitude.z());
    printf("Vel: %.2f, %.2f, %.2f | ", point.velocity.x(), point.velocity.y(),
           point.velocity.z());
    printf("Acc: %.2f, %.2f, %.2f \n", point.acceleration.x(),
           point.acceleration.y(), point.acceleration.z());
  }
  printf("========\n");
}

Eigen::Vector3d TrajectoryExt::evaluatePoly(const double dt,
                                             const int derivative) {
  Eigen::Vector3d result = Eigen::Vector3d::Zero();
  switch (derivative) {
    case 0: {
      for (int j = 0; j <= poly_order_; j++) {
        result += poly_coeff_.at(j) * std::pow(dt, j);
      }
      break;
    }
    case 1: {
      for (int j = derivative; j <= poly_order_; j++) {
        result += j * poly_coeff_.at(j) * std::pow(dt, j - derivative);
      }
      break;
    }
    case 2: {
      for (int j = derivative; j <= poly_order_; j++) {
        result +=
            j * (j - 1) * poly_coeff_.at(j) * std::pow(dt, j - derivative);
      }
      break;
    }
    case 3: {
      for (int j = 3; j <= poly_order_; j++) {
        result += (j) * (j - 1) * (j - 2) * poly_coeff_.at(j) *
                  std::pow(dt, j - derivative);
      }
      break;
    }
  }

  return result;
}

void TrajectoryExt::clear() {
  points_.clear();
  poly_coeff_.clear();
}

void TrajectoryExt::enableYawing(const bool enable_yawing) {
  yawing_enabled_ = enable_yawing;
}

double TrajectoryExt::computeControlCost() {
  double cumulative_thrust = 0.0;
  double cumulative_bodyrate_x = 0.0;
  double cumulative_bodyrate_y = 0.0;

  for (auto point : points_) {
    cumulative_thrust += point.collective_thrust;
    cumulative_bodyrate_x += point.bodyrates.x();
    cumulative_bodyrate_y += point.bodyrates.y();
  }

  return cumulative_thrust + cumulative_bodyrate_x + cumulative_bodyrate_y;
}

void TrajectoryExt::recomputeTrajectory() {
  if (frame_id_ != FrameID::World) {
    std::printf("Can only recompute trajectory in world frame!\n");
    return;
  }
  recomputeVelocity();
  recomputeAcceleration();
}

void TrajectoryExt::recomputeVelocity() {
  // iterate over points, compute numerical derivatives
  TrajectoryExtPoint prev_point = points_.front();
  for (int i = 1; i < points_.size(); i++) {
    TrajectoryExtPoint curr_point = points_.at(i);

    points_.at(i).velocity =
        (curr_point.position - prev_point.position) /
        (curr_point.time_from_start - prev_point.time_from_start);
    prev_point = curr_point;
  }
  points_.front().velocity = points_.at(1).velocity;
}

void TrajectoryExt::recomputeAcceleration() {
  // iterate over points, compute numerical derivatives
  TrajectoryExtPoint prev_point = points_.front();
  for (int i = 1; i < points_.size(); i++) {
    TrajectoryExtPoint curr_point = points_.at(i);

    points_.at(i).acceleration =
        (curr_point.velocity - prev_point.velocity) /
        (curr_point.time_from_start - prev_point.time_from_start);
    prev_point = curr_point;
  }
  points_.front().acceleration = points_.at(1).acceleration;

  for (int i = 1; i < points_.size(); i++) {
    // Attitude
    Eigen::Vector3d thrust =
        points_.at(i).acceleration + 9.81 * Eigen::Vector3d::UnitZ();
    Eigen::Vector3d I_eZ_I(0.0, 0.0, 1.0);
    Eigen::Quaterniond q_pitch_roll =
        Eigen::Quaterniond::FromTwoVectors(I_eZ_I, thrust);

    Eigen::Vector3d linvel_body =
        q_pitch_roll.inverse() * points_.at(i).velocity;
    double heading = 0.0;
    if (yawing_enabled_) {
      //      heading = std::atan2(linvel_body.y(), linvel_body.x());
      heading =
          std::atan2(points_.at(i).velocity.y(), points_.at(i).velocity.x());
    }

    Eigen::Quaterniond q_heading = Eigen::Quaterniond(
        Eigen::AngleAxisd(heading, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q_att = q_pitch_roll * q_heading;
    q_att.normalize();
    points_.at(i).attitude = q_att;

    // Inputs
    points_.at(i).collective_thrust = thrust.norm();
    // compute bodyrates
    double time_step =
        points_.at(1).time_from_start - points_.at(0).time_from_start;
    Eigen::Vector3d thrust_1 =
        points_.at(i - 1).acceleration + 9.81 * Eigen::Vector3d::UnitZ();
    Eigen::Vector3d thrust_2 =
        points_.at(i).acceleration + 9.81 * Eigen::Vector3d::UnitZ();
    thrust_1.normalize();
    thrust_2.normalize();
    Eigen::Vector3d crossProd =
        thrust_1.cross(thrust_2);  // direction of omega, in inertial axes
    Eigen::Vector3d angular_rates_wf = Eigen::Vector3d::Zero();
    if (crossProd.norm() > 0.0) {
      angular_rates_wf =
          std::acos(std::min(1.0, std::max(-1.0, thrust_1.dot(thrust_2)))) /
          time_step * crossProd / (crossProd.norm() + 0.0000001);
    }
    points_.at(i).bodyrates = q_att.inverse() * angular_rates_wf;
  }
  points_.front().attitude = points_.at(1).attitude;
  points_.front().bodyrates = points_.at(1).bodyrates;
  points_.front().collective_thrust = points_.at(1).collective_thrust;
}

bool TrajectoryExt::setConstantArcLengthSpeed(const double &speed,
                                               const int &traj_len,
                                               const double &traj_dt) {
  //  std::printf("setConstantArcLengthSpeed\n");
  // this only works when the trajectory is already fitted with a polynomial
  // (workaround...?)
  if (poly_coeff_.empty()) {
    std::printf("polynomial coefficients unknown!");
    return false;
  }

  // we can reuse the previous first point
  // this assumes that the first point already correctly fulfills the continuity
  // constraints
  std::vector<TrajectoryExtPoint> new_points;
  new_points.push_back(points_.front());
  // iterate through the polynomial, find points that are equally spaced
  double start_time = 0.0;
  double end_time = points_.back().time_from_start;
  int steps = 100;
  double dt = (end_time - start_time) / steps;

  int j = 1;  // this index iterates through the evaluation points

  double t_curr = start_time + dt;
  Eigen::Vector3d pos_prev = evaluatePoly(t_curr, 0);
  Eigen::Vector3d pos_curr;
  double acc_arc_length = 0.0;
  while (t_curr <= end_time) {
    // finely sample the polynomial
    // as soon as point is found that has accumulated arc length of x, add it to
    // the points
    pos_curr = evaluatePoly(t_curr, 0);

    acc_arc_length += (pos_curr - pos_prev).norm();
    if (acc_arc_length >= speed * j * traj_dt) {
      // add this point to the resampled points
      TrajectoryExtPoint temp_point;
      temp_point.time_from_start = j * traj_dt;
      temp_point.position = pos_curr;
      new_points.push_back(temp_point);
      j += 1;
    }
    if (new_points.size() >= points_.size()) {
      break;
    }
    pos_prev = pos_curr;
    t_curr += dt;
  }

  // in case we don't have enough points, extrapolate with constant velocity
  if (j < 2) {
    std::printf("not enough points to extrapolate, won't adapt trajectory!\n");
    return false;
  }

  while (new_points.size() < points_.size()) {
    TrajectoryExtPoint temp_point;
    temp_point.time_from_start = j * traj_dt;
    temp_point.position =
        new_points.at(j - 1).position +
        (new_points.at(j - 1).position - new_points.at(j - 2).position);
    new_points.push_back(temp_point);
    j += 1;
  }
  // refit the polynomial coefficients to these points
  points_.clear();
  for (auto &point : new_points) {
    points_.push_back(point);
  }
  fitPolynomialCoeffs(poly_order_, continuity_order_);
  resamplePointsFromPolyCoeffs();
  return true;
}
