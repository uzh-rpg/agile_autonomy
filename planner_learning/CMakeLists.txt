cmake_minimum_required(VERSION 2.8.3)
project(planner_learning)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
add_compile_options(-O3)

#find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)

#cs_add_library(acrobatic_sequence src/acrobatic_sequence.cpp)

#cs_add_executable(fpv_aggressive_trajectories src/fpv_aggressive_trajectories.cpp)
#target_link_libraries(fpv_aggressive_trajectories acrobatic_sequence)
#cs_add_executable(odometry_republisher src/odometry_republisher.cpp)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  catkin_simple REQUIRED
)

catkin_python_setup()
catkin_simple()


cs_install()
cs_export()
