project(traj_sampler)
cmake_minimum_required(VERSION 3.0.0)
message("Detected cmake version ${CMAKE_VERSION}")

find_package(catkin_simple REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# find all open3D stuff
message("Searching open3D in /usr/local/lib/cmake/")
find_package(Open3D HINTS /usr/local/lib/cmake/)
list(APPEND Open3D_LIBRARIES dl)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message("Found OpenMP ${OpenMP_CXX_FOUND}  ${OpenMP_VERSION} ${OpenMP_CXX_VERSION_MAJOR} ${Open3D_VERSION} OpenMP::OpenMP_CXX")
    get_cmake_property(_variableNames VARIABLES)
    list(SORT _variableNames)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach ()
endif ()


# Open3D
if (Open3D_FOUND)
    message("Found Open3D ${Open3D_VERSION}")
    # link_directories must be before add_executable
    link_directories(${Open3D_LIBRARY_DIRS})
else ()
    message("Open3D not found")
endif ()

add_definitions(-std=c++17)

catkin_simple()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACK -DEIGEN_USE_LAPACK -Ofast")

cs_add_library(kdtree src/kdtree.cpp
        src/tinyply.cpp)
target_include_directories(kdtree PUBLIC ${Open3D_INCLUDE_DIRS})
target_link_libraries(kdtree ${Open3D_LIBRARIES}
        OpenMP::OpenMP_CXX)

cs_add_library(traj_sampler src/traj_sampler.cpp)
target_include_directories(traj_sampler PUBLIC
        ../../../rpg_mpc/model/quadrotor_mpc_codegen/
        ../../../rpg_mpc/externals/qpoases
        ../../../rpg_mpc/externals/qpoases/INCLUDE
        ../../../rpg_mpc/externals/qpoases/SRC)
target_link_libraries(traj_sampler
        kdtree
        ${LAPACK_LIBRARIES}
        ${catkin_LIBRARIES})

cs_add_executable(generate_label src/generate_label.cpp)
target_link_libraries(generate_label traj_sampler
        ${catkin_LIBRARIES}
        stdc++fs)

cs_install()
cs_export()
