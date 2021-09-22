#include "agile_autonomy/flightmare_bridge.h"

#include "quadrotor_common/parameter_helper.h"

#include <experimental/filesystem>

// Flightmare dependencies
#include <rpgq_simulator/implementation/objects/quadrotor_vehicle/quad_and_rgb_camera.h>
#include <rpgq_simulator/tools/env_changer.h>
#include <rpgq_simulator/tools/point_cloud_generator.h>
#include <rpgq_simulator/visualization/flightmare_bridge.hpp>
#include <rpgq_simulator/visualization/flightmare_message_types.hpp>

namespace flightmare_bridge {
FlightmareBridge::FlightmareBridge(const ros::NodeHandle& nh,
                                   const ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {
  if (!loadParameters()) {
    ROS_ERROR("[%s] Failed to load all parameters in FlightmareBridge!",
              ros::this_node::getName().c_str());
    ros::shutdown();
  }

  // Flightmare setup
  // quad_ ID can be any real number between
  // 0 ~ 25, each ID corresponding to a unique name
  RPGQ::QuadrotorID quad_ID = 1;
  std::string quad_name = RPGQ::QuadrotorName(quad_ID);
  // create quadrotor with a stereo RGB camera attached.
  quad_stereo_ = std::make_shared<RPGQ::Simulator::QuadStereoRGBCamera>(
      quad_name, nullptr, 1000000);

  // configure the camera
  double hor_fov_radians = (M_PI * (rgb_fov_deg_ / 180.0));
  double flightmare_fov =
      2. * std::atan(std::tan(hor_fov_radians / 2) * img_rows_ / img_cols_);
  flightmare_fov = (flightmare_fov / M_PI) * 180.0;
  std::cout << "Vertical Fov is " << flightmare_fov << std::endl;
  left_rgb_cam_ = quad_stereo_->GetLeftRGBCamera();
  left_rgb_cam_->EnableOpticalFlow(false);
  left_rgb_cam_->EnableCategorySegment(false);
  left_rgb_cam_->EnableDepth(true);
  left_rgb_cam_->SetWidth(img_cols_);
  left_rgb_cam_->SetHeight(img_rows_);
  left_rgb_cam_->SetFov(flightmare_fov);

  right_rgb_cam_ = quad_stereo_->GetRightRGBCamera();
  right_rgb_cam_->EnableOpticalFlow(false);
  right_rgb_cam_->EnableCategorySegment(false);
  right_rgb_cam_->EnableDepth(false);
  right_rgb_cam_->SetWidth(img_cols_);
  right_rgb_cam_->SetHeight(img_rows_);
  right_rgb_cam_->SetFov(flightmare_fov);

  // set the relative position of the camera with respect to quadrotor center
  // mass
  Eigen::Vector3d B_r_BCr(0.0, 0.0, 0.1);
  Eigen::Vector3d B_r_BCl(0.0, stereo_baseline_, 0.1);
  // set the relative rotation of the camera
  Eigen::Matrix3d R_BCr;
  Eigen::Matrix3d R_BCl;
  R_BCr = Eigen::AngleAxisd(0.0 * M_PI, Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(-pitch_angle_deg_ / 180.0 * M_PI,
                            Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(-0.5 * M_PI, Eigen::Vector3d::UnitZ());
  R_BCl = Eigen::AngleAxisd(0.0 * M_PI, Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(-pitch_angle_deg_ / 180.0 * M_PI,
                            Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(-0.5 * M_PI, Eigen::Vector3d::UnitZ());
  right_rgb_cam_->SetRelPose(B_r_BCr, R_BCr);
  left_rgb_cam_->SetRelPose(B_r_BCl, R_BCl);

  // configure the quadrotor
  quad_ = quad_stereo_->GetQuad();
  Eigen::Vector3d quad_position{0.0, 0.0, 3.0};
  quad_->SetPos(quad_position);
  quad_->SetQuat(Eigen::Quaterniond(std::cos(0.5 * M_PI_2), 0.0, 0.0,
                                    std::sin(0.5 * M_PI_2)));
  quad_->SetSize(Eigen::Vector3d(0.01, 0.01, 0.01));

  scene_id_ = env_idx_;

  // create flightmare birdge and connect sockets..
  flightmareBridge_ptr_ = RPGQ::Simulator::FlightmareBridge::getInstance();

  flightmareBridge_ptr_->initializeConnections();
  flightmareBridge_ptr_->addQuadStereoRGB(quad_stereo_);

  if (!unity_is_ready_) {
    ROS_INFO("Connecting to unity...");
    // connect to unity.
    // please open the Unity3D standalone.
    double time_out_count = 0;
    double sleep_useconds = 1.0 * 1e5;
    const double connection_time_out = 99.0;  // seconds
    while (!flightmare_ready_ && ros::ok()) {
      if (flightmareBridge_ptr_ != nullptr) {
        // connect unity
        flightmareBridge_ptr_->setScene(scene_id_);
        flightmare_ready_ = flightmareBridge_ptr_->connectUnity();
        usleep(sleep_useconds);
      }
      if (time_out_count / 1e6 > connection_time_out) {
        std::cout << "Flightmare connection failed, time out." << std::endl;
        break;
      }
      // sleep
      usleep(sleep_useconds);
      // increase time out counter
      time_out_count += sleep_useconds;
    }
  }
  // wait 1 seconds. until to environment is fully loaded.
  usleep(1 * 1e6);
  // wait until point cloud is created.
  // Unity is frozen as long as point cloud is created
  // check if it's possible to send and receive a frame again and then
  // continue
  RPGQ::FlightmareTypes::ImgID send_id = 1;
  flightmareBridge_ptr_->getRender(send_id);
  RPGQ::Simulator::RenderMessage_t unity_output;
  RPGQ::FlightmareTypes::ImgID receive_id = 0;
  while (send_id != receive_id) {
    receive_id = flightmareBridge_ptr_->handleOutput(unity_output);
  }
  generator_ = std::default_random_engine(random_device_());
  // Unity is ready when image is received
  ROS_INFO("Unity is ready!");
  unity_is_ready_ = true;

  sgm_.reset(new sgm_gpu::SgmGpu(pnh_, img_cols_, img_rows_));

  // Image Transport Publisher
  image_transport::ImageTransport it(pnh_);
  unity_rgb_pub_ = it.advertise("unity_rgb", 1);
  unity_depth_pub_ = it.advertise("unity_depth", 1);
  sgm_depth_pub_ = it.advertise("sgm_depth", 1);

  // ROS publishers and subscribers
  tree_spacing_sub_ = nh_.subscribe(
      "tree_spacing", 1, &FlightmareBridge::treeSpacingCallback, this);
  object_spacing_sub_ = nh_.subscribe(
      "object_spacing", 1, &FlightmareBridge::objectSpacingCallback, this);
  remove_objects_sub_ = nh_.subscribe(
      "remove_objects", 1, &FlightmareBridge::removeObjectsCallback, this);
}

void FlightmareBridge::disconnect() {
  flightmareBridge_ptr_->disconnectUnity();
}

void FlightmareBridge::spawnObjects(
    const quadrotor_common::TrajectoryPoint& start_state) {
  if (spawn_trees_ || spawn_objects_) {
    RPGQ::Simulator::rmObjects();
    RPGQ::Simulator::rmTrees();
  }
  ROS_INFO("Find elevation");
  if (spawn_trees_) {
    ROS_INFO("Place trees");
    // Remove the previous trees if they exist
    RPGQ::Simulator::rmObjects();
    RPGQ::Simulator::rmTrees();
    RPGQ::Simulator::TreeMessage_t tree_msg;
    // compute the requested tree density for Poisson
    double density = 1.0 / (avg_tree_spacing_ * avg_tree_spacing_);
    int num_trees =
        static_cast<int>(bounding_box_[0] * bounding_box_[1] * density);
    // draw sample from poisson distribution
    std::poisson_distribution<int> poisson_dist(num_trees);
    tree_msg.density = static_cast<double>(poisson_dist(generator_));
    ROS_INFO("Spawning [%d] trees, poisson mode is [%d].",
             static_cast<int>(tree_msg.density), num_trees);
    if (seed_ == 0) {
      // generate random seed
      std::uniform_int_distribution<int> distribution(1, 1 << 20);
      tree_msg.seed = distribution(generator_);
      ROS_INFO("Generated random seed [%d]", tree_msg.seed);
    } else {
      tree_msg.seed = seed_ + rollout_idx_ % 10;
      ROS_INFO("Incrementing seed to %d.", tree_msg.seed);
      rollout_idx_ += 1;
    }
    tree_msg.bounding_origin[0] =
        start_state.position.x() + bounding_box_origin_[0];
    tree_msg.bounding_origin[1] =
        start_state.position.y() + bounding_box_origin_[1];
    tree_msg.bounding_area[0] = bounding_box_[0];
    tree_msg.bounding_area[1] = bounding_box_[1];
    RPGQ::Simulator::placeTrees(tree_msg);
  }

  if (spawn_objects_) {
    ROS_INFO("Place objects");
    // Remove the previous trees if they exist
    RPGQ::Simulator::rmObjects();
    RPGQ::Simulator::ObjectMessage_t obj_msg;
    if (seed_ == 0) {
      // generate random seed
      std::uniform_int_distribution<int> distribution(1, 1 << 20);
      obj_msg.seed = distribution(generator_);
      ROS_INFO("Generated random seed [%d]", obj_msg.seed);
    } else {
      seed_ += 1;
      ROS_INFO("Incrementing seed to %d.", seed_);
      obj_msg.seed = seed_;
    }
    obj_msg.name = object_names_;
    // compute the requested tree density for Poisson
    double density =
        1.0 / (avg_object_spacing_ * avg_object_spacing_ * avg_object_spacing_);
    int num_objects = static_cast<int>(bounding_box_[0] * bounding_box_[1] *
                                       bounding_box_[2] * density);
    // draw sample from poisson distribution
    std::poisson_distribution<int> poisson_dist(num_objects);
    obj_msg.density = static_cast<double>(poisson_dist(generator_));
    ROS_INFO("Spawning [%d] objects, poisson mode is [%d].",
             static_cast<int>(obj_msg.density), num_objects);
    obj_msg.rand_size = rand_width_;
    obj_msg.scale_min = min_object_scale_;
    obj_msg.scale_max = max_object_scale_;
    obj_msg.rot_min = min_object_angles_;
    obj_msg.rot_max = max_object_angles_;
    obj_msg.bounding_origin[0] =
        start_state.position.x() + bounding_box_origin_[0];
    obj_msg.bounding_origin[1] =
        start_state.position.y() + bounding_box_origin_[1];
    obj_msg.bounding_origin[2] =
        start_state.position.z() + bounding_box_origin_[2];
    obj_msg.bounding_area[0] = bounding_box_[0];
    obj_msg.bounding_area[1] = bounding_box_[1];
    obj_msg.bounding_area[2] = bounding_box_[2];
    RPGQ::Simulator::placeObjects(obj_msg);
    ROS_INFO("Waiting 5 seconds after object spawning...");
    usleep(5 * 1e6);
  }
}

void FlightmareBridge::removeObjectsCallback(
    const std_msgs::EmptyConstPtr& msg) {
  RPGQ::Simulator::rmObjects();
  RPGQ::Simulator::rmTrees();
};

void FlightmareBridge::generatePointcloud(
    const Eigen::Ref<Eigen::Vector3d>& min_corner,
    const Eigen::Ref<Eigen::Vector3d>& max_corner,
    const std::string& curr_data_dir) {
  ROS_INFO("Start creating pointcloud");
  RPGQ::Simulator::PointCloudMessage_t pcd_msg;
  pcd_msg.scene_id = scene_id_;
  pcd_msg.bounding_box_scale =
      std::vector<double>{(max_corner.x() - min_corner.x()) + 20,
                          (max_corner.y() - min_corner.y()) + 20,
                          (max_corner.z() - min_corner.z()) + 10};
  ROS_INFO("Scale pointcloud: [%.2f, %.2f, %.2f]",
           pcd_msg.bounding_box_scale.at(0), pcd_msg.bounding_box_scale.at(1),
           pcd_msg.bounding_box_scale.at(2));
  pcd_msg.bounding_box_origin = std::vector<double>{
      (max_corner.x() + min_corner.x()) / 2.0,
      (max_corner.y() + min_corner.y()) / 2.0,
      (max_corner.z() + min_corner.z()) / 2.0};
  ROS_INFO("Origin pointcloud: [%.2f, %.2f, %.2f]",
           pcd_msg.bounding_box_origin.at(0), pcd_msg.bounding_box_origin.at(1),
           pcd_msg.bounding_box_origin.at(2));

  pcd_msg.path = curr_data_dir + "/";
  pcd_msg.file_name = "pointcloud-unity";
  pcd_msg.unity_ground_offset = 0.0;
  pcd_msg.resolution_above_ground = pc_resolution_;
  pcd_msg.resolution_below_ground = pc_resolution_;

  RPGQ::Simulator::pointCloudGenerator(pcd_msg);

  // render Unity until point cloud exists
  RPGQ::FlightmareTypes::ImgID send_id = 1;
  RPGQ::Simulator::RenderMessage_t unity_output;
  while (!std::experimental::filesystem::exists(pcd_msg.path +
                                                pcd_msg.file_name + ".ply")) {
    usleep(1 * 1e6);
  }

  usleep(5 * 1e6);

  ROS_INFO("Pointcloud saved");
}

void FlightmareBridge::getImageFromUnity(
    const quadrotor_common::QuadStateEstimate& state_estimate,
    cv::Mat* left_frame, cv::Mat* right_frame, cv::Mat* gt_depth_frame) {
  // 1) set Pose camera in unity
  Eigen::Vector3d quad_pos = state_estimate.position;
  quad_->SetPos(quad_pos);
  quad_->SetQuat(state_estimate.orientation);

  // 2) Send request for frame
  RPGQ::FlightmareTypes::ImgID send_id =
      state_estimate.timestamp.toNSec() % 4294967296;
  flightmareBridge_ptr_->getRender(send_id);
  RPGQ::Simulator::RenderMessage_t unity_output;
  RPGQ::FlightmareTypes::ImgID receive_id = 0;

  // 3) Wait until frame is ready
  while (send_id != receive_id) {
    receive_id = flightmareBridge_ptr_->handleOutput(unity_output);
  }

  // 4) Receive frame
  left_rgb_cam_->GetRGBImage(*left_frame);
  left_rgb_cam_->GetDepthmap(*gt_depth_frame);
  right_rgb_cam_->GetRGBImage(*right_frame);
}

void FlightmareBridge::computeDepthImage(const cv::Mat& left_frame,
                                         const cv::Mat& right_frame,
                                         cv::Mat* const depth) {
  ros::WallTime start_disp_comp = ros::WallTime::now();
  cv::Mat disparity(img_rows_, img_cols_, CV_8UC1);
  if (perform_sgm_) {
    sgm_->computeDisparity(left_frame, right_frame, &disparity);
  }
  disparity.convertTo(disparity, CV_32FC1);

  // compute depth from disparity
  cv::Mat depth_float(img_rows_, img_cols_, CV_32FC1);

  float f = (img_cols_ / 2.0) / std::tan((M_PI * (rgb_fov_deg_ / 180.0)) / 2.0);
  //  depth = static_cast<float>(stereo_baseline_) * f / disparity;
  for (int r = 0; r < img_rows_; ++r) {
    for (int c = 0; c < img_cols_; ++c) {
      if (disparity.at<float>(r, c) == 0.0f) {
        depth_float.at<float>(r, c) = 0.0f;
        depth->at<unsigned short>(r, c) = 0;
      } else if (disparity.at<float>(r, c) == 255.0f) {
        depth_float.at<float>(r, c) = 0.0f;
        depth->at<unsigned short>(r, c) = 0;
      } else {
        depth_float.at<float>(r, c) = static_cast<float>(stereo_baseline_) * f /
                                      disparity.at<float>(r, c);
        depth->at<unsigned short>(r, c) = static_cast<unsigned short>(
            1000.0 * static_cast<float>(stereo_baseline_) * f /
            disparity.at<float>(r, c));
      }
    }
  }
  double disp_comp_duration = (ros::WallTime::now() - start_disp_comp).toSec();
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

void FlightmareBridge::getImages(
    const quadrotor_common::QuadStateEstimate& state_estimate,
    const std::string curr_data_dir, const int frame_counter) {
  cv::Mat left_rgb_img;
  cv::Mat right_rgb_img;
  cv::Mat gt_depth_img;
  cv::Mat gt_depth_uint16(img_rows_, img_cols_, CV_16UC1);
  getImageFromUnity(state_estimate, &left_rgb_img, &right_rgb_img,
                    &gt_depth_img);

  std::string ty = type2str(gt_depth_img.type());
  for (int r = 0; r < img_rows_; ++r) {
    for (int c = 0; c < img_cols_; ++c) {
      gt_depth_uint16.at<unsigned short>(r, c) = static_cast<unsigned short>(
          1000.0 * std::min(65.0f, gt_depth_img.at<float>(r, c)));
    }
  }

  // compute disparity image
  cv::Mat depth_uint16(img_rows_, img_cols_, CV_16UC1);
  computeDepthImage(left_rgb_img, right_rgb_img, &depth_uint16);

  // overlay the speed on the left image for debugging / visualization
  cv::Mat rviz_rgb_img = left_rgb_img.clone();
  double speed = state_estimate.velocity.norm();
  std::stringstream stream;
  stream << std::fixed << "Speed: " << std::setprecision(2) << speed;
  std::string speed_string = stream.str();
  cv::putText(rviz_rgb_img, speed_string, cv::Point(20, 20),  // Coordinates
              cv::FONT_HERSHEY_COMPLEX_SMALL,                 // Font
              1.0,                        // Scale. 2.0 = 2x bigger
              cv::Scalar(255, 255, 255),  // BGR Color
              1);                         // Line Thickness (Optional)

  // publish to rviz
  sensor_msgs::ImagePtr rgb_msg =
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", rviz_rgb_img).toImageMsg();
  rgb_msg->header.stamp = state_estimate.timestamp;
  unity_rgb_pub_.publish(rgb_msg);
  sensor_msgs::ImagePtr depth_msg =
      cv_bridge::CvImage(std_msgs::Header(), "mono16", gt_depth_uint16)
          .toImageMsg();
  depth_msg->header.stamp = state_estimate.timestamp;
  unity_depth_pub_.publish(depth_msg);
  sensor_msgs::ImagePtr sgm_depth_msg =
      cv_bridge::CvImage(std_msgs::Header(), "mono16", depth_uint16)
          .toImageMsg();
  sgm_depth_msg->header.stamp = state_estimate.timestamp;
  sgm_depth_pub_.publish(sgm_depth_msg);

  if (!curr_data_dir.empty()) {
    std::ostringstream ss;
    ss << std::setw(8) << std::setfill('0') << frame_counter;
    std::string s2(ss.str());

    // save image to disk
    std::string left_rgb_img_filename =
        curr_data_dir + "/img/frame_left_" + s2 + ".png";
    std::string right_rgb_img_filename =
        curr_data_dir + "/img/frame_right_" + s2 + ".png";
    std::string depth_img_filename =
        curr_data_dir + "/img/depth_" + s2 + ".tif";
    std::string gt_depth_img_filename =
        curr_data_dir + "/img/gt_depth_" + s2 + ".tif";

    // Unity images
    // 5) Save frame
    cv::imwrite(left_rgb_img_filename, left_rgb_img);
    cv::imwrite(right_rgb_img_filename, right_rgb_img);
    cv::imwrite(depth_img_filename, depth_uint16);
    cv::imwrite(gt_depth_img_filename, gt_depth_uint16);
  }
}

void FlightmareBridge::treeSpacingCallback(
    const std_msgs::Float32ConstPtr& msg) {
  avg_tree_spacing_ = static_cast<double>(msg->data);
}

void FlightmareBridge::objectSpacingCallback(
    const std_msgs::Float32ConstPtr& msg) {
  avg_object_spacing_ = static_cast<double>(msg->data);
}

bool FlightmareBridge::loadParameters() {
  if (!quadrotor_common::getParam("unity/spawn_trees", spawn_trees_, true))
    return false;

  if (!quadrotor_common::getParam("unity/avg_tree_spacing", avg_tree_spacing_,
                                  5.0))
    return false;

  if (!quadrotor_common::getParam("unity/perform_sgm", perform_sgm_, true))
    return false;

  if (!quadrotor_common::getParam("unity/spawn_objects", spawn_objects_, true))
    return false;

  if (!quadrotor_common::getParam("unity/avg_object_spacing",
                                  avg_object_spacing_, 5.0))
    return false;
  if (!quadrotor_common::getParam("unity/rand_width", rand_width_, 5.0))
    return false;

  if (!quadrotor_common::getParam("unity/seed", seed_, 0)) return false;

  if (!quadrotor_common::getParam("unity/env_idx", env_idx_, 0)) return false;

  if (!pnh_.getParam("unity/bounding_box", bounding_box_)) return false;

  if (!pnh_.getParam("unity/bounding_box_origin", bounding_box_origin_))
    return false;

  if (!pnh_.getParam("unity/min_object_scale", min_object_scale_)) return false;

  if (!pnh_.getParam("unity/max_object_scale", max_object_scale_)) return false;

  if (!pnh_.getParam("unity/min_object_angles", min_object_angles_))
    return false;

  if (!pnh_.getParam("unity/max_object_angles", max_object_angles_))
    return false;

  if (!pnh_.getParam("unity/object_names", object_names_)) return false;

  if (!quadrotor_common::getParam("unity/pointcloud_resolution", pc_resolution_,
                                  0.3))
    return false;

  // Camera parameters
  if (!quadrotor_common::getParam("camera/fov", rgb_fov_deg_, 90.0))
    return false;
  if (!quadrotor_common::getParam("camera/width", img_cols_, 320)) return false;
  if (!quadrotor_common::getParam("camera/height", img_rows_, 240))
    return false;
  if (!quadrotor_common::getParam("camera/baseline", stereo_baseline_, 0.1))
    return false;
  if (!quadrotor_common::getParam("camera/pitch_angle_deg", pitch_angle_deg_,
                                  0.0))
    return false;

  return true;
}
}  // namespace flightmare_bridge
