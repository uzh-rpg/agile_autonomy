cmake_minimum_required(VERSION 2.8.3)
project(example_vision)

find_package(catkin_simple REQUIRED COMPONENTS)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

catkin_package(
  LIBRARIES
  CATKIN_DEPENDS 
)
catkin_simple(ALL_DEPS_REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# add files
file(GLOB_RECURSE EXAMPLE_VISION_SOURCE
  "*.h"
  "*.c"
  "*.hpp"
  "*.cpp"
)

cs_add_executable(camera_control 
  src/camera_control.cpp)

#
target_link_libraries(
  camera_control
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  zmq
  zmqpp
)
cs_install()


