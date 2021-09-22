#include "traj_sampler/kdtree.h"

#include <iostream>

#include "example-utils.hpp"
#include "tinyply.h"

KdTreeSampling::KdTreeSampling(const std::string pointcloud_filename,
                               const double crash_dist,
                               const double crash_penalty,
                               Eigen::Vector3d drone_dimensions)
    : crash_dist_(crash_dist),
      crash_penalty_(crash_penalty),
      drone_dimensions_(drone_dimensions) {
  parse_pointcloud(pointcloud_filename);
}

void KdTreeSampling::parse_pointcloud(const std::string pointcloud_filename) {
  std::ifstream csvFile;
  csvFile.open(pointcloud_filename.c_str());

  if (!csvFile.is_open()) {
    std::cout << "Path Wrong!!!!" << std::endl;
    std::cout << pointcloud_filename << std::endl;
    exit(EXIT_FAILURE);
  }

  open3d::geometry::PointCloud pointcloud;

  std::cout << "Now Reading: " << pointcloud_filename << std::endl;

  std::unique_ptr<std::istream> file_stream;
  std::vector<uint8_t> byte_buffer;

  try {
    // For most files < 1gb, pre-loading the entire file upfront and wrapping it
    // into a stream is a net win for parsing speed, about 40% faster.
    bool preload_into_memory = true;
    if (preload_into_memory) {
      byte_buffer = read_file_binary(pointcloud_filename);
      file_stream.reset(
          new memory_stream((char *)byte_buffer.data(), byte_buffer.size()));
    } else {
      file_stream.reset(
          new std::ifstream(pointcloud_filename, std::ios::binary));
    }

    if (!file_stream || file_stream->fail())
      throw std::runtime_error("file_stream failed to open " +
                               pointcloud_filename);

    file_stream->seekg(0, std::ios::end);
    const double size_mb = file_stream->tellg() * double(1e-6);
    file_stream->seekg(0, std::ios::beg);

    PlyFile file;
    file.parse_header(*file_stream);

    // Because most people have their own mesh types, tinyply treats parsed data
    // as structured/typed byte buffers. See examples below on how to marry your
    // own application-specific data structures with this one.
    std::shared_ptr<PlyData> vertices, normals, colors, texcoords, faces,
        tripstrip;

    // The header information can be used to programmatically extract properties
    // on elements known to exist in the header prior to reading the data. For
    // brevity of this sample, properties like vertex position are hard-coded:
    try {
      vertices =
          file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception &e) {
      //      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      normals =
          file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    } catch (const std::exception &e) {
      //      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      colors = file.request_properties_from_element(
          "vertex", {"red", "green", "blue", "alpha"});
    } catch (const std::exception &e) {
      //      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      colors =
          file.request_properties_from_element("vertex", {"r", "g", "b", "a"});
    } catch (const std::exception &e) {
      //      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      texcoords = file.request_properties_from_element("vertex", {"u", "v"});
    } catch (const std::exception &e) {
      //      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.
    try {
      faces =
          file.request_properties_from_element("face", {"vertex_indices"}, 3);
    } catch (const std::exception &e) {
      //      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    // Tristrips must always be read with a 0 list size hint (unless you know
    // exactly how many elements are specifically in the file, which is
    // unlikely);
    try {
      tripstrip = file.request_properties_from_element("tristrips",
                                                       {"vertex_indices"}, 0);
    } catch (const std::exception &e) {
      //      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    manual_timer read_timer;

    read_timer.start();
    file.read(*file_stream);
    read_timer.stop();

    const double parsing_time = read_timer.get() / 1000.f;
    std::cout << "\tparsing " << size_mb << "mb in " << parsing_time
              << " seconds [" << (size_mb / parsing_time) << " MBps]"
              << std::endl;

    if (vertices)
      std::cout << "\tRead " << vertices->count << " total vertices "
                << std::endl;

    const size_t numVerticesBytes = vertices->buffer.size_bytes();
    std::vector<float3> verts(vertices->count);
    std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);

    int idx = 0;
    for (auto point_tinyply : verts) {
      if (idx == 0) {
        points_ = Eigen::Vector3d(static_cast<double>(point_tinyply.x),
                                  static_cast<double>(point_tinyply.y),
                                  static_cast<double>(point_tinyply.z));
      } else {
        points_.conservativeResize(points_.rows(), points_.cols() + 1);
        points_.col(points_.cols() - 1) =
            Eigen::Vector3d(static_cast<double>(point_tinyply.x),
                            static_cast<double>(point_tinyply.y),
                            static_cast<double>(point_tinyply.z));
      }
      idx += 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }

  kd_tree_.SetMatrixData(points_);
  std::cout << "Completed pointcloud parsing!" << std::endl;
}

/// search closest points around current state
/// returns true if a point is found and a minimum distance
bool KdTreeSampling::searchRadius(const Eigen::Vector3d &query_point,
                                  const Eigen::Quaterniond &attitude,
                                  const bool &use_attitude, const double radius,
                                  double *min_distance) const {
  std::vector<int> indices;
  std::vector<double> distances_squared;
  *min_distance = std::numeric_limits<double>::max();

  // get all points within some radius from the query position
  kd_tree_.SearchRadius(query_point, radius, indices, distances_squared);
  //  kd_tree_.SearchKNN(query_point, 1, indices, distances_squared);
  if (indices.size() == 0) {
    // no points are found within the query radius
    return false;
  }

  if (use_attitude) {
    // we iterate through all found points and check if they actually touch the
    // drone
    for (auto close_point_idx : indices) {
      // get point, check if within drone body
      Eigen::Vector3d close_point = points_.col(close_point_idx);

      // project point on each body axis and check distance
      Eigen::Vector3d close_point_body =
          attitude.inverse() * (close_point - query_point);

      if (std::abs(close_point_body.x()) <= std::abs(drone_dimensions_.x()) &&
          std::abs(close_point_body.y()) <= std::abs(drone_dimensions_.y()) &&
          std::abs(close_point_body.z()) <= std::abs(drone_dimensions_.z())) {
        // point is in collision
        *min_distance = 0.0;
        return true;
      }
    }
  }

  *min_distance = std::sqrt(*std::min_element(std::begin(distances_squared),
                                              std::end(distances_squared)));

  return true;
}

/// checks if trajectory is in collision.
/// As soon as point in collision is detected, adds penalty to cost, returns
/// early.
bool KdTreeSampling::query_kdtree(const double *state_array,
                                  double *accumulated_cost_array,
                                  const int traj_len,
                                  const int query_every_nth_point,
                                  const bool &use_attitude) const {
  double min_distance_traj = std::numeric_limits<double>::max();
  for (int j = 0; j <= traj_len; j = j + query_every_nth_point) {
    Eigen::Vector3d query_point(static_cast<double>(state_array[(13 * j + 0)]),
                                static_cast<double>(state_array[(13 * j + 1)]),
                                static_cast<double>(state_array[(13 * j + 2)]));
    Eigen::Quaterniond attitude(
        static_cast<double>(state_array[(13 * j + 9)]),
        static_cast<double>(state_array[(13 * j + 10)]),
        static_cast<double>(state_array[(13 * j + 11)]),
        static_cast<double>(state_array[(13 * j + 12)]));

    // do query for this point
    double query_radius = use_attitude
                              ? (crash_dist_ + drone_dimensions_.norm())
                              : drone_dimensions_.minCoeff();

    if (!use_attitude && j < 2) {
      query_radius *= 2.0;
    }

    double min_distance_point;
    bool found_point_within_radius = searchRadius(
        query_point, attitude, use_attitude, query_radius, &min_distance_point);

    if (found_point_within_radius) {
      if (use_attitude) {
        if (min_distance_point == 0.0) {
          accumulated_cost_array[0] += static_cast<double>(crash_penalty_);
          return true;
        } else {
          accumulated_cost_array[0] +=
              static_cast<double>((query_radius - min_distance_point) /
                                  query_radius * crash_penalty_);
        }
      } else {
        accumulated_cost_array[0] += static_cast<double>(crash_penalty_);
        return true;
      }
    }
  }
  return false;
}