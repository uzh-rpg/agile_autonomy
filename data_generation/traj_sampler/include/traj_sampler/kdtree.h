#pragma once

#include <random>

#include <Open3D/Geometry/KDTreeFlann.h>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/IO/ClassIO/PointCloudIO.h>

class KdTreeSampling {
 public:
  KdTreeSampling(const std::string pointcloud_filename, const double crash_dist,
                 const double crash_penalty, Eigen::Vector3d drone_dimensions);

  bool query_kdtree(const double *state_array, double *accumulated_cost_array,
                    const int traj_len, const int query_every_nth_point,
                    const bool &use_attitude) const;

 private:
  open3d::geometry::KDTreeFlann kd_tree_;
  Eigen::MatrixXd points_;
  double crash_dist_;
  double crash_penalty_;
  Eigen::Vector3d drone_dimensions_;
  void parse_pointcloud(const std::string pointcloud_filename);
  bool searchRadius(const Eigen::Vector3d &query_point,
                    const Eigen::Quaterniond &attitude,
                    const bool &use_attitude, const double radius,
                    double *min_distance) const;
};
