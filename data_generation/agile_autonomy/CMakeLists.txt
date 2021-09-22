cmake_minimum_required(VERSION 2.8.3)
project(agile_autonomy)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
add_compile_options(-O3)

find_package(catkin_simple REQUIRED COMPONENTS)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_62,code=sm_62
        -gencode=arch=compute_75,code=sm_75
        )

message("CUDA Libs: ${CUDA_LIBRARIES}")
message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
message("CUDA curand library: ${CUDA_curand_LIBRARY}")

catkin_package(
        LIBRARIES
        CATKIN_DEPENDS
)
catkin_simple(ALL_DEPS_REQUIRED)

cuda_add_library(sgm_gpu
        src/sgm_gpu/sgm_gpu.cu
        src/sgm_gpu/costs.cu
        src/sgm_gpu/hamming_cost.cu
        src/sgm_gpu/left_right_consistency.cu
        src/sgm_gpu/median_filter.cu)

cs_add_library(flightmare_bridge src/flightmare_bridge.cpp)
target_link_libraries(flightmare_bridge stdc++fs
        sgm_gpu
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        zmq
        zmqpp)

cs_add_executable(agile_autonomy src/agile_autonomy.cpp)
target_include_directories(agile_autonomy PUBLIC
        ../../../rpg_mpc/model/quadrotor_mpc_codegen/
        ../../../rpg_mpc/externals/qpoases
        ../../../rpg_mpc/externals/qpoases/INCLUDE
        ../../../rpg_mpc/externals/qpoases/SRC)
target_link_libraries(
        agile_autonomy stdc++fs
        flightmare_bridge
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
)

cs_install()
cs_export()
