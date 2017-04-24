cmake_minimum_required(VERSION 2.8.3)
project(apriltagn_ros)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  image_transport
  roscpp
  sensor_msgs
  geometry_msgs
  tf
  cv_bridge
  message_generation
  nodelet
  std_msgs
  cmake_modules
)

find_package(Eigen3 REQUIRED)


message("Eigen   " ${Eigen_INCLUDE_DIRS})
message("Eigen3  " ${Eigen3_INCLUDE_DIRS})
message("Eigen3  " ${EIGEN3_INCLUDE_DIRS})

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS image_transport roscpp sensor_msgs geometry_msgs tf cv_bridge std_msgs
  #  DEPENDS Eigen
)

###########
## Build ##
###########

include_directories(include include/common)
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_library(apriltag src/apriltag.c
src/apriltag_quad_thresh.c
src/common/zarray.c
src/common/pam.c
src/common/image_u8.c
src/common/matd.c
src/common/pjpeg.c
src/common/zmaxheap.c
src/common/homography.c
src/common/image_u8x3.c
src/common/string_util.c
src/common/zhash.c
src/common/image_f32.c
src/common/unionfind.c
src/common/pnm.c
src/common/workerpool.c
src/common/pjpeg-idct.c
src/common/getopt.c
src/common/svd22.c
src/common/image_u8x4.c
src/common/time_util.c
src/common/g2d.c)
add_library(tag16h5 src/tag16h5.c)
add_library(tag25h7 src/tag25h7.c)
add_library(tag25h9 src/tag25h9.c)
add_library(tag36artoolkit src/tag36artoolkit.c)
add_library(tag36h10 src/tag36h10.c)
add_library(tag36h11 src/tag36h11.c)

add_executable(apriltag_detector
  src/apriltag_detector.cpp
)
target_link_libraries(apriltag_detector
  ${catkin_LIBRARIES} 
  apriltag
  tag16h5  tag25h7  tag25h9  tag36artoolkit  tag36h10  tag36h11
)
#add_dependencies(apriltag_detector apriltags_ros_generate_messages_cpp)


#add_library(apriltag_detector_nodelet src/apriltag_detector_nodelet.cpp)
#target_link_libraries(apriltag_detector_nodelet
#  ${catkin_LIBRARIES}
#  apriltag_detector
#)
#
#add_executable(apriltag_detector_node src/apriltag_detector_node.cpp)
#target_link_libraries(apriltag_detector_node
#  ${catkin_LIBRARIES}
#  apriltag_detector
#)
