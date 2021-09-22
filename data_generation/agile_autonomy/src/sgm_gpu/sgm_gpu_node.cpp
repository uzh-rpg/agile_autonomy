/***********************************************************************
  Copyright (C) 2020 Hironori Fujimoto

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
 
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************/

#include "sgm_gpu_node.h"

namespace sgm_gpu
{

SgmGpuNode::SgmGpuNode()
{
  node_handle_.reset(new ros::NodeHandle());
  private_node_handle_.reset(new ros::NodeHandle("~"));

  image_transport_.reset(new image_transport::ImageTransport(*node_handle_));

  sgm_.reset(new SgmGpu(*private_node_handle_));

  disparity_pub_ = private_node_handle_->advertise<stereo_msgs::DisparityImage>("disparity", 1);

  // Subscribe left and right Image topic
  std::string left_base_topic = node_handle_->resolveName("left_image");
  std::string right_base_topic = node_handle_->resolveName("right_image");
  left_image_sub_.subscribe(*image_transport_, left_base_topic, 10);
  right_image_sub_.subscribe(*image_transport_, right_base_topic, 10);

  // Find CameraInfo topic from corresponded Image topic and subscribe it
  std::string left_info_topic = image_transport::getCameraInfoTopic(left_base_topic);
  std::string right_info_topic = image_transport::getCameraInfoTopic(right_base_topic);
  left_info_sub_.subscribe(*node_handle_, left_info_topic, 10);
  right_info_sub_.subscribe(*node_handle_, right_info_topic, 10);

  stereo_synchronizer_.reset(
    new StereoSynchronizer(left_image_sub_, right_image_sub_, left_info_sub_, right_info_sub_, 10)
  );
  stereo_synchronizer_->registerCallback(&SgmGpuNode::stereoCallback, this);
}

void SgmGpuNode::stereoCallback(
  const sensor_msgs::ImageConstPtr &left_image,
  const sensor_msgs::ImageConstPtr &right_image,
  const sensor_msgs::CameraInfoConstPtr &left_info,
  const sensor_msgs::CameraInfoConstPtr &right_info
)
{
  if (disparity_pub_.getNumSubscribers() == 0)
    return;

  stereo_msgs::DisparityImage disparity;
  sgm_->computeDisparity(*left_image, *right_image, *left_info, *right_info, disparity);

  disparity_pub_.publish(disparity);
}

} // namespace sgm_gpu
