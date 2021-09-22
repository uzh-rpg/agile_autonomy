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
#include "sgm_gpu/sgm_gpu.h"

#include "sgm_gpu/cost_aggregation.h"
#include "sgm_gpu/costs.h"
#include "sgm_gpu/hamming_cost.h"
#include "sgm_gpu/left_right_consistency.h"
#include "sgm_gpu/median_filter.h"

#include <cv_bridge/cv_bridge.h>
//#include <image_geometry/stereo_camera_model.h>

namespace sgm_gpu {
// Variables which have CUDA-related type are put here
//   in order to include sgm_gpu.h from non-CUDA package
cudaStream_t stream1_;
cudaStream_t stream2_;
cudaStream_t stream3_;

dim3 BLOCK_SIZE_;
dim3 grid_size_;

SgmGpu::SgmGpu(const ros::NodeHandle &parent_node_handle, const int cols,
               const int rows)
    : memory_allocated_(false), cols_(cols), rows_(rows) {
  private_node_handle_.reset(
      new ros::NodeHandle(parent_node_handle, "libsgm_gpu"));
  // Get parameters used in SGM algorithm
  p1_ = 6;   // static_cast<uint8_t>(private_node_handle_->param("p1", 6));
  p2_ = 96;  // static_cast<uint8_t>(private_node_handle_->param("p2", 96));
  check_consistency_ =
      true;  // private_node_handle_->param("check_consistency", true);

  // Create streams
  cudaStreamCreate(&stream1_);
  cudaStreamCreate(&stream2_);
  cudaStreamCreate(&stream3_);
}

SgmGpu::~SgmGpu() {
  freeMemory();

  cudaStreamDestroy(stream1_);
  cudaStreamDestroy(stream2_);
  cudaStreamDestroy(stream3_);
}

void SgmGpu::allocateMemory(uint32_t cols, uint32_t rows) {
  freeMemory();

  cols_ = cols;
  rows_ = rows;

  int total_pixel = cols_ * rows_;
  cudaMalloc((void **)&d_im0_, sizeof(uint8_t) * total_pixel);
  cudaMalloc((void **)&d_im1_, sizeof(uint8_t) * total_pixel);

  cudaMalloc((void **)&d_transform0_, sizeof(cost_t) * total_pixel);
  cudaMalloc((void **)&d_transform1_, sizeof(cost_t) * total_pixel);

  int cost_volume_size = total_pixel * MAX_DISPARITY;
  cudaMalloc((void **)&d_cost_, sizeof(uint8_t) * cost_volume_size);

  cudaMalloc((void **)&d_L0_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L1_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L2_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L3_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L4_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L5_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L6_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L7_, sizeof(uint8_t) * cost_volume_size);

  cudaMalloc((void **)&d_s_, sizeof(uint16_t) * cost_volume_size);

  cudaMalloc((void **)&d_disparity_, sizeof(uint8_t) * total_pixel);
  cudaMalloc((void **)&d_disparity_filtered_uchar_,
             sizeof(uint8_t) * total_pixel);
  cudaMalloc((void **)&d_disparity_right_, sizeof(uint8_t) * total_pixel);
  cudaMalloc((void **)&d_disparity_right_filtered_uchar_,
             sizeof(uint8_t) * total_pixel);

  memory_allocated_ = true;
}

void SgmGpu::freeMemory() {
  if (!memory_allocated_) return;

  cudaFree(d_im0_);
  cudaFree(d_im1_);
  cudaFree(d_transform0_);
  cudaFree(d_transform1_);
  cudaFree(d_L0_);
  cudaFree(d_L1_);
  cudaFree(d_L2_);
  cudaFree(d_L3_);
  cudaFree(d_L4_);
  cudaFree(d_L5_);
  cudaFree(d_L6_);
  cudaFree(d_L7_);
  cudaFree(d_disparity_);
  cudaFree(d_disparity_filtered_uchar_);
  cudaFree(d_disparity_right_);
  cudaFree(d_disparity_right_filtered_uchar_);
  cudaFree(d_cost_);
  cudaFree(d_s_);

  memory_allocated_ = false;
}

bool SgmGpu::computeDisparity(const sensor_msgs::Image &left_image,
                              const sensor_msgs::Image &right_image,
                              const sensor_msgs::CameraInfo &left_camera_info,
                              const sensor_msgs::CameraInfo &right_camera_info,
                              stereo_msgs::DisparityImage &disparity_msg) {
  if (left_image.width != right_image.width ||
      left_image.height != right_image.height) {
    ROS_ERROR_STREAM_NAMED(
        "libsgm_gpu",
        "Image dimension of left and right are not same: \n"
            << "Left: " << left_image.width << "x" << left_image.height << "\n"
            << "Right: " << right_image.width << "x" << right_image.height);
    return false;
  }

  if (left_image.encoding != right_image.encoding) {
    ROS_ERROR_STREAM_NAMED("libsgm_gpu",
                           "Image encoding of left and right are not same: \n"
                               << "Left: " << left_image.encoding << "\n"
                               << "Right: " << right_image.encoding);
    return false;
  }

  // Convert to 8 bit grayscale image
  cv_bridge::CvImagePtr left_mono8 =
      cv_bridge::toCvCopy(left_image, sensor_msgs::image_encodings::MONO8);
  cv_bridge::CvImagePtr right_mono8 =
      cv_bridge::toCvCopy(right_image, sensor_msgs::image_encodings::MONO8);

  // Resize images to their width and height divisible by 4 for limit of CUDA
  // code
  resizeToDivisibleBy4(left_mono8->image, right_mono8->image);

  // Reallocate memory if needed
  bool size_changed =
      (cols_ != left_mono8->image.cols || rows_ != left_mono8->image.rows);
  if (!memory_allocated_ || size_changed)
    allocateMemory(left_mono8->image.cols, left_mono8->image.rows);

  // Copy image to GPU device
  size_t mono8_image_size = left_mono8->image.total() * sizeof(uint8_t);
  cudaMemcpyAsync(d_im0_, left_mono8->image.ptr<uint8_t>(), mono8_image_size,
                  cudaMemcpyHostToDevice, stream1_);
  cudaMemcpyAsync(d_im1_, right_mono8->image.ptr<uint8_t>(), mono8_image_size,
                  cudaMemcpyHostToDevice, stream1_);

  BLOCK_SIZE_.x = 32;
  BLOCK_SIZE_.y = 32;

  grid_size_.x = (cols_ + BLOCK_SIZE_.x - 1) / BLOCK_SIZE_.x;
  grid_size_.y = (rows_ + BLOCK_SIZE_.y - 1) / BLOCK_SIZE_.y;

  CenterSymmetricCensusKernelSM2<<<grid_size_, BLOCK_SIZE_, 0, stream1_>>>(
      d_im0_, d_im1_, d_transform0_, d_transform1_, rows_, cols_);

  cudaStreamSynchronize(stream1_);
  HammingDistanceCostKernel<<<rows_, MAX_DISPARITY, 0, stream1_>>>(
      d_transform0_, d_transform1_, d_cost_, rows_, cols_);

  const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE / WARP_SIZE;
  const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ / WARP_SIZE;

  // Cost Aggregation
  CostAggregationKernelLeftToRight<<<(rows_ + PIXELS_PER_BLOCK_HORIZ - 1) /
                                         PIXELS_PER_BLOCK_HORIZ,
                                     COSTAGG_BLOCKSIZE_HORIZ, 0, stream2_>>>(
      d_cost_, d_L0_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
      d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_,
      d_L6_);
  CostAggregationKernelRightToLeft<<<(rows_ + PIXELS_PER_BLOCK_HORIZ - 1) /
                                         PIXELS_PER_BLOCK_HORIZ,
                                     COSTAGG_BLOCKSIZE_HORIZ, 0, stream3_>>>(
      d_cost_, d_L1_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
      d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_,
      d_L6_);
  CostAggregationKernelUpToDown<<<(cols_ + PIXELS_PER_BLOCK - 1) /
                                      PIXELS_PER_BLOCK,
                                  COSTAGG_BLOCKSIZE, 0, stream1_>>>(
      d_cost_, d_L2_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
      d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_,
      d_L6_);
  CostAggregationKernelDownToUp<<<(cols_ + PIXELS_PER_BLOCK - 1) /
                                      PIXELS_PER_BLOCK,
                                  COSTAGG_BLOCKSIZE, 0, stream1_>>>(
      d_cost_, d_L3_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
      d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_,
      d_L6_);
  CostAggregationKernelDiagonalDownUpLeftRight<<<
      (cols_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0,
      stream1_>>>(d_cost_, d_L4_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
                  d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_,
                  d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalUpDownLeftRight<<<
      (cols_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0,
      stream1_>>>(d_cost_, d_L5_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
                  d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_,
                  d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalDownUpRightLeft<<<
      (cols_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0,
      stream1_>>>(d_cost_, d_L6_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
                  d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_,
                  d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalUpDownRightLeft<<<
      (cols_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0,
      stream1_>>>(d_cost_, d_L7_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
                  d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_,
                  d_L4_, d_L5_, d_L6_);

  int total_pixel = rows_ * cols_;
  MedianFilter3x3<<<(total_pixel + MAX_DISPARITY - 1) / MAX_DISPARITY,
                    MAX_DISPARITY, 0, stream1_>>>(
      d_disparity_, d_disparity_filtered_uchar_, rows_, cols_);

  if (check_consistency_) {
    ChooseRightDisparity<<<grid_size_, BLOCK_SIZE_, 0, stream1_>>>(
        d_disparity_right_, d_s_, rows_, cols_);
    MedianFilter3x3<<<(total_pixel + MAX_DISPARITY - 1) / MAX_DISPARITY,
                      MAX_DISPARITY, 0, stream1_>>>(
        d_disparity_right_, d_disparity_right_filtered_uchar_, rows_, cols_);

    LeftRightConsistencyCheck<<<grid_size_, BLOCK_SIZE_, 0, stream1_>>>(
        d_disparity_filtered_uchar_, d_disparity_right_filtered_uchar_, rows_,
        cols_);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    ROS_ERROR_NAMED("libsgm_gpu", "%s %d\n", cudaGetErrorString(err), err);
    return false;
  }

  cudaDeviceSynchronize();
  cv::Mat disparity(rows_, cols_, CV_8UC1);
  cudaMemcpy(disparity.data, d_disparity_filtered_uchar_,
             sizeof(uint8_t) * total_pixel, cudaMemcpyDeviceToHost);

  // Restore image size if resized to be divisible by 4
  if (cols_ != left_image.width || rows_ != left_image.height) {
    cv::Size input_size(left_image.width, left_image.height);
    cv::resize(disparity, disparity, input_size, 0, 0, cv::INTER_AREA);
  }

  //  convertToMsg(disparity, left_camera_info, right_camera_info,
  //  disparity_msg);

  return true;
}

bool SgmGpu::computeDisparity(const cv::Mat &left_image,
                              const cv::Mat &right_image,
                              cv::Mat *disparity_out) {
  // Convert images to grayscale
  cv::Mat left_mono8, right_mono8;
  if (left_image.channels() > 1) {
    cv::cvtColor(left_image, left_mono8, CV_RGB2GRAY);
  }

  if (right_image.channels() > 1) {
    cv::cvtColor(right_image, right_mono8, CV_RGB2GRAY);
  }

  // Resize images to their width and height divisible by 4 for limit of CUDA
  // code
  resizeToDivisibleBy4(left_mono8, right_mono8);

  // Reallocate memory if needed
  bool size_changed = (cols_ != left_mono8.cols || rows_ != left_mono8.rows);
  if (!memory_allocated_ || size_changed)
    allocateMemory(left_mono8.cols, left_mono8.rows);

  // Copy image to GPU device
  size_t mono8_image_size = left_mono8.total() * sizeof(uint8_t);
  cudaMemcpyAsync(d_im0_, left_mono8.ptr<uint8_t>(), mono8_image_size,
                  cudaMemcpyHostToDevice, stream1_);
  cudaMemcpyAsync(d_im1_, right_mono8.ptr<uint8_t>(), mono8_image_size,
                  cudaMemcpyHostToDevice, stream1_);

  BLOCK_SIZE_.x = 32;
  BLOCK_SIZE_.y = 32;

  grid_size_.x = (cols_ + BLOCK_SIZE_.x - 1) / BLOCK_SIZE_.x;
  grid_size_.y = (rows_ + BLOCK_SIZE_.y - 1) / BLOCK_SIZE_.y;

  CenterSymmetricCensusKernelSM2<<<grid_size_, BLOCK_SIZE_, 0, stream1_>>>(
      d_im0_, d_im1_, d_transform0_, d_transform1_, rows_, cols_);

  cudaStreamSynchronize(stream1_);
  HammingDistanceCostKernel<<<rows_, MAX_DISPARITY, 0, stream1_>>>(
      d_transform0_, d_transform1_, d_cost_, rows_, cols_);

  const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE / WARP_SIZE;
  const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ / WARP_SIZE;

  // Cost Aggregation
  CostAggregationKernelLeftToRight<<<(rows_ + PIXELS_PER_BLOCK_HORIZ - 1) /
                                         PIXELS_PER_BLOCK_HORIZ,
                                     COSTAGG_BLOCKSIZE_HORIZ, 0, stream2_>>>(
      d_cost_, d_L0_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
      d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_,
      d_L6_);
  CostAggregationKernelRightToLeft<<<(rows_ + PIXELS_PER_BLOCK_HORIZ - 1) /
                                         PIXELS_PER_BLOCK_HORIZ,
                                     COSTAGG_BLOCKSIZE_HORIZ, 0, stream3_>>>(
      d_cost_, d_L1_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
      d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_,
      d_L6_);
  CostAggregationKernelUpToDown<<<(cols_ + PIXELS_PER_BLOCK - 1) /
                                      PIXELS_PER_BLOCK,
                                  COSTAGG_BLOCKSIZE, 0, stream1_>>>(
      d_cost_, d_L2_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
      d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_,
      d_L6_);
  CostAggregationKernelDownToUp<<<(cols_ + PIXELS_PER_BLOCK - 1) /
                                      PIXELS_PER_BLOCK,
                                  COSTAGG_BLOCKSIZE, 0, stream1_>>>(
      d_cost_, d_L3_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
      d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_,
      d_L6_);
  CostAggregationKernelDiagonalDownUpLeftRight<<<
      (cols_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0,
      stream1_>>>(d_cost_, d_L4_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
                  d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_,
                  d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalUpDownLeftRight<<<
      (cols_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0,
      stream1_>>>(d_cost_, d_L5_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
                  d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_,
                  d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalDownUpRightLeft<<<
      (cols_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0,
      stream1_>>>(d_cost_, d_L6_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
                  d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_,
                  d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalUpDownRightLeft<<<
      (cols_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0,
      stream1_>>>(d_cost_, d_L7_, d_s_, p1_, p2_, rows_, cols_, d_transform0_,
                  d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_,
                  d_L4_, d_L5_, d_L6_);

  int total_pixel = rows_ * cols_;
  MedianFilter3x3<<<(total_pixel + MAX_DISPARITY - 1) / MAX_DISPARITY,
                    MAX_DISPARITY, 0, stream1_>>>(
      d_disparity_, d_disparity_filtered_uchar_, rows_, cols_);

  if (check_consistency_) {
    ChooseRightDisparity<<<grid_size_, BLOCK_SIZE_, 0, stream1_>>>(
        d_disparity_right_, d_s_, rows_, cols_);
    MedianFilter3x3<<<(total_pixel + MAX_DISPARITY - 1) / MAX_DISPARITY,
                      MAX_DISPARITY, 0, stream1_>>>(
        d_disparity_right_, d_disparity_right_filtered_uchar_, rows_, cols_);

    LeftRightConsistencyCheck<<<grid_size_, BLOCK_SIZE_, 0, stream1_>>>(
        d_disparity_filtered_uchar_, d_disparity_right_filtered_uchar_, rows_,
        cols_);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    ROS_ERROR_NAMED("libsgm_gpu", "%s %d\n", cudaGetErrorString(err), err);
    return false;
  }

  cudaDeviceSynchronize();
  cv::Mat disparity(rows_, cols_, CV_8UC1);
  cudaMemcpy(disparity.data, d_disparity_filtered_uchar_,
             sizeof(uint8_t) * total_pixel, cudaMemcpyDeviceToHost);
  cudaMemcpy(disparity_out->data, d_disparity_filtered_uchar_,
             sizeof(uint8_t) * total_pixel, cudaMemcpyDeviceToHost);

  // Restore image size if resized to be divisible by 4
  if (cols_ != left_image.cols || rows_ != left_image.rows) {
    cv::Size input_size(left_image.cols, left_image.rows);
    cv::resize(disparity, disparity, input_size, 0, 0, cv::INTER_AREA);
  }

  //  convertToMsg(disparity, left_camera_info, right_camera_info,
  //  disparity_msg);

  return true;
}

void SgmGpu::resizeToDivisibleBy4(cv::Mat &left_image, cv::Mat &right_image) {
  bool need_resize = false;
  cv::Size original_size, resized_size;

  original_size = cv::Size(left_image.cols, left_image.rows);
  resized_size = original_size;
  if (original_size.width % 4 != 0) {
    need_resize = true;
    resized_size.width = (original_size.width / 4 + 1) * 4;
  }
  if (original_size.height % 4 != 0) {
    need_resize = true;
    resized_size.height = (original_size.height / 4 + 1) * 4;
  }

  if (need_resize) {
    cv::resize(left_image, left_image, resized_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(right_image, right_image, resized_size, 0, 0, cv::INTER_LINEAR);
  }
}

// void SgmGpu::convertToMsg(const cv::Mat_<unsigned char> &disparity,
//                          const sensor_msgs::CameraInfo &left_camera_info,
//                          const sensor_msgs::CameraInfo &right_camera_info,
//                          stereo_msgs::DisparityImage &disparity_msg) {
//  cv::Mat disparity_32f;
//  disparity.convertTo(disparity_32f, CV_32F);
//  cv_bridge::CvImage disparity_converter(
//      left_camera_info.header, sensor_msgs::image_encodings::TYPE_32FC1,
//      disparity_32f);
//  disparity_converter.toImageMsg(disparity_msg.image);
//
//  disparity_msg.header = left_camera_info.header;
//
//  image_geometry::StereoCameraModel stereo_model;
//  stereo_model.fromCameraInfo(left_camera_info, right_camera_info);
//  disparity_msg.f = stereo_model.left().fx();
//  disparity_msg.T = stereo_model.baseline();
//
//  disparity_msg.min_disparity = 0.0;
//  disparity_msg.max_disparity = MAX_DISPARITY;
//  disparity_msg.delta_d = 1.0;
//}

}  // namespace sgm_gpu
