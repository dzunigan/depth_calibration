cmake_minimum_required(VERSION 2.8.3)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

if(NOT ROS_BUILD_TYPE)
  set(ROS_BUILD_TYPE Release)
endif()

message("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall -Wextra")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
else(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
  set(MRPT_DONT_USE_DBG_LIBS ON)
endif(CMAKE_BUILD_TYPE STREQUAL "Release")
message(STATUS "Flags: " ${CMAKE_CXX_FLAGS})

## Dependencies
set(GFLAGS_PREFER_EXPORTED_GFLAGS_CMAKE_CONFIGURATION FALSE CACHE BOOL "Prefer exported GFlags?")

#find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(Eigen3 REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(MRPT REQUIRED base opengl gui)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

find_package(Git)

include(GenerateVersionDefinitions)

include_directories(include ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIR} ${GLOG_INCLUDE_DIRS} ${MRPT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## Header files
file(GLOB_RECURSE HEADER_FILES include/*.hpp include/*.h)
add_custom_target(header_files SOURCES ${HEADER_FILES})

rosbuild_add_executable(show_distortion
src/show_distortion.cpp
)
rosbuild_add_boost_directories()
rosbuild_link_boost(show_distortion filesystem system)

target_link_libraries(show_distortion
${GFLAGS_LIBRARY}
${GLOG_LIBRARIES}
${MRPT_LIBRARIES}
${OpenCV_LIBRARIES}
)

rosbuild_add_executable(calibrate
src/calibrate.cpp
)
rosbuild_add_boost_directories()
rosbuild_link_boost(calibrate filesystem system)

target_link_libraries(calibrate
${GFLAGS_LIBRARY}
${GLOG_LIBRARIES}
${MRPT_LIBRARIES}
${OpenCV_LIBRARIES}
)

rosbuild_add_executable(show_calibration
src/show_calibration.cpp
)
rosbuild_add_boost_directories()
rosbuild_link_boost(show_calibration filesystem system)

target_link_libraries(show_calibration
${GFLAGS_LIBRARY}
${GLOG_LIBRARIES}
${MRPT_LIBRARIES}
${OpenCV_LIBRARIES}
)

rosbuild_add_executable(local_distortion
src/local_distortion.cpp
)
rosbuild_add_boost_directories()
rosbuild_link_boost(local_distortion filesystem system)

target_link_libraries(local_distortion
${GFLAGS_LIBRARY}
${GLOG_LIBRARIES}
${MRPT_LIBRARIES}
${OpenCV_LIBRARIES}
)

rosbuild_add_executable(global_error
src/global_error.cpp
)
rosbuild_add_boost_directories()
rosbuild_link_boost(global_error filesystem system)

target_link_libraries(global_error
${GFLAGS_LIBRARY}
${GLOG_LIBRARIES}
${MRPT_LIBRARIES}
${OpenCV_LIBRARIES}
${PCL_LIBRARIES}
)

rosbuild_add_executable(show_correction
src/show_correction.cpp
)
rosbuild_add_boost_directories()
rosbuild_link_boost(show_correction filesystem system)

target_link_libraries(show_correction
${GFLAGS_LIBRARY}
${GLOG_LIBRARIES}
${MRPT_LIBRARIES}
${OpenCV_LIBRARIES}
)

## Unit tests
rosbuild_add_executable(test_plane_transform
test/test_plane_transform.cpp
)
rosbuild_add_boost_directories()
rosbuild_link_boost(test_plane_transform filesystem system)

rosbuild_add_executable(test_plane_estimation
test/test_plane_estimation.cpp
)
rosbuild_add_boost_directories()
rosbuild_link_boost(test_plane_estimation filesystem system)

