project(agile_autonomy_utils)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple REQUIRED)

add_definitions(-std=c++17)

catkin_simple()

cs_add_library(visualize src/visualize.cpp)

cs_add_library(trajectory_ext src/trajectory_ext.cpp)

cs_add_library(logging src/logging.cpp)

cs_install()

cs_export()
