/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.

This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <iostream>

#include <opencv2/opencv.hpp>

#include <apriltag.h>
#include <tag36h11.h>
#include <tag36h10.h>
#include <tag36artoolkit.h>
#include <tag25h9.h>
#include <tag25h7.h>
#include <common/getopt.h>

#include <ros/ros.h>
#include <ros/node_handle.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <image_transport/camera_common.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <geometry_msgs/PoseArray.h>

using namespace std;
//using namespace cv;

#define SHOW_IMAGE 1

class AprilTagDetector
{
  getopt_t *getopt;
  cv::Mat gray;
  apriltag_family_t *tf_;
  apriltag_detector_t *td_;
	ros::NodeHandle node_;
  double tag_size_;
  std::string famname_,
              image_topic_;
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo> sync_a_;

  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;

  ros::Publisher pose_pub_;

  public:
    AprilTagDetector(const ros::NodeHandle);
    ~AprilTagDetector();
    Eigen::Matrix4d getRelativeTransform(double, double, double, double, double [4][2]) const;
    void imageCallback (const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&);
};


AprilTagDetector::AprilTagDetector(ros::NodeHandle nh)
  :sync_a_(1),
   node_(nh)   
{
    node_.param<std::string>("family", famname_, std::string("tag36h11"));

    // Initialize tag detector with options
    if ( !famname_.compare("tag36h11"))
    {
        tf_ = tag36h11_create();
    }
    else if (!famname_.compare("tag36h10"))
        tf_ = tag36h10_create();
    else if (!famname_.compare("tag36artoolkit"))
        tf_ = tag36artoolkit_create();
    else if (!famname_.compare("tag25h9"))
        tf_ = tag25h9_create();
    else if (!famname_.compare("tag25h7"))
        tf_ = tag25h7_create();
    else {
        ROS_ERROR("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
        exit(-1);
    }


//    node_.param<uint32_t>("border", tf_->black_border, (uint32_t)1);
    tf_->black_border = (uint32_t) node_.param("border", 1);
    
    td_ = apriltag_detector_create();
    apriltag_detector_add_family(td_, tf_);
    node_.param<int>("debug", td_->debug, 0);
    //node_.param<bool>("quiet",
    node_.param<int>("threads", td_->nthreads, 4);
    node_.param<float>("decimate", td_->quad_decimate, 1.0);
    //node_.param<double>("blur", td_-> , 0.0);
    node_.param<int>("refine_edges", td_->refine_edges, 1);
    node_.param<int>("refine_decode", td_->refine_decode, 0);
    node_.param<int>("refine_pose", td_->refine_pose, 0);

    node_.param<double>("tag_size", tag_size_, 1);
//  node_.param<double>("tag_size", tag_size_, 0.163513);

    node_.param<std::string>("image_topic", image_topic_, "image");
    image_sub_.subscribe(node_, image_topic_.c_str(), 1);
    info_sub_.subscribe(node_, image_transport::getCameraInfoTopic(image_topic_.c_str()), 1);
    
    sync_a_.connectInput(image_sub_, info_sub_);
    sync_a_.registerCallback(boost::bind(&AprilTagDetector::imageCallback, this, _1, _2));

    pose_pub_ = node_.advertise<geometry_msgs::PoseArray>("tag_detections_pose", 1);
}

AprilTagDetector::~AprilTagDetector()
{
    apriltag_detector_destroy(td_);
    if (!famname_.compare("tag36h11"))
        tag36h11_destroy(tf_);
    else if (!famname_.compare("tag36h10"))
        tag36h10_destroy(tf_);
    else if (!famname_.compare("tag36artoolkit"))
        tag36artoolkit_destroy(tf_);
    else if (!famname_.compare("tag25h9"))
        tag25h9_destroy(tf_);
    else if (!famname_.compare("tag25h7"))
        tag25h7_destroy(tf_);
}


Eigen::Matrix4d AprilTagDetector::getRelativeTransform(double fx, double fy, double px, double py, double p[4][2]) const {
  std::vector<cv::Point3f> objPts;
  std::vector<cv::Point2f> imgPts;
  double s = tag_size_/2.;
  objPts.push_back(cv::Point3f(-s,-s, 0));
  objPts.push_back(cv::Point3f( s,-s, 0));
  objPts.push_back(cv::Point3f( s, s, 0));
  objPts.push_back(cv::Point3f(-s, s, 0));

  std::pair<float, float> p1( p[0][0], p[0][1]);
  std::pair<float, float> p2( p[1][0], p[1][1]);
  std::pair<float, float> p3( p[2][0], p[2][1]);
  std::pair<float, float> p4( p[3][0], p[3][1]);
  imgPts.push_back(cv::Point2f(p1.first, p1.second));
  imgPts.push_back(cv::Point2f(p2.first, p2.second));
  imgPts.push_back(cv::Point2f(p3.first, p3.second));
  imgPts.push_back(cv::Point2f(p4.first, p4.second));

  cv::Mat rvec, tvec;
  cv::Matx33f cameraMatrix(
                           fx, 0, px,
                           0, fy, py,
                           0,  0,  1);
  cv::Vec4f distParam(0,0,0,0); // all 0?
  cv::solvePnP(objPts, imgPts, cameraMatrix, distParam, rvec, tvec);
  cv::Matx33d r;
  cv::Rodrigues(rvec, r);
  Eigen::Matrix3d wRo;
  wRo << r(0,0), r(0,1), r(0,2), r(1,0), r(1,1), r(1,2), r(2,0), r(2,1), r(2,2);

  Eigen::Matrix4d T; 
  T.topLeftCorner(3,3) = wRo;
  T.col(3).head(3) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
  T.row(3) << 0,0,0,1;

  return T;
}

void AprilTagDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info)
{
  cv_bridge::CvImagePtr cv_ptr;
  try{
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e){
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat gray;
  cv::cvtColor(cv_ptr->image, gray, CV_BGR2GRAY);
  // Make an image_u8_t header for the Mat data
  image_u8_t im = { .width = gray.cols,
      .height = gray.rows,
      .stride = gray.cols,
      .buf = gray.data
  };

  zarray_t *detections = apriltag_detector_detect(td_, &im);
  ROS_DEBUG("%d tag detected", zarray_size(detections));

  static tf::TransformBroadcaster br;
  tf::Transform transform;

  double fx = cam_info->P[0];
  double fy = cam_info->P[5];
  double px = cam_info->P[2];
  double py = cam_info->P[6];


  geometry_msgs::PoseArray tag_pose_array;
  tag_pose_array.header.frame_id = "camera";

  bool tag_seen[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Draw detection outlines
  for (int i = 0; i < zarray_size(detections); i++)
  {
      
      apriltag_detection_t *det;
      zarray_get(detections, i, &det);
      Eigen::Matrix4d transform = getRelativeTransform(fx, fy, px, py, det->p);
#if SHOW_IMAGE > 0
      cv::line(gray, cv::Point(det->p[0][0], det->p[0][1]),
               cv::Point(det->p[1][0], det->p[1][1]),
               cv::Scalar(0, 0xff, 0), 2);
      cv::line(gray, cv::Point(det->p[0][0], det->p[0][1]),
               cv::Point(det->p[3][0], det->p[3][1]),
               cv::Scalar(0, 0, 0xff), 2);
      cv::line(gray, cv::Point(det->p[1][0], det->p[1][1]),
               cv::Point(det->p[2][0], det->p[2][1]),
               cv::Scalar(0xff, 0, 0), 2);
      cv::line(gray, cv::Point(det->p[2][0], det->p[2][1]),
               cv::Point(det->p[3][0], det->p[3][1]),
               cv::Scalar(0xff, 0, 0), 2);
#endif
      Eigen::Matrix3d rot = transform.block(0, 0, 3, 3);
      Eigen::Quaternion<double> rot_quaternion;
      rot_quaternion = rot;

      geometry_msgs::PoseStamped tag_pose;
      tag_pose.pose.position.x = transform(0, 3);
      tag_pose.pose.position.y = transform(1, 3);
      tag_pose.pose.position.z = transform(2, 3);
      tag_pose.pose.orientation.x = rot_quaternion.x();
      tag_pose.pose.orientation.y = rot_quaternion.y();
      tag_pose.pose.orientation.z = rot_quaternion.z();
      tag_pose.pose.orientation.w = rot_quaternion.w();
      tag_pose.header = cv_ptr->header;

      tag_pose_array.poses.push_back(tag_pose.pose);
      tag_seen[det->id] = true;
#if SHOW_IMAGE > 0
      stringstream ss;
      ss << det->id;
      cv::String text = ss.str();
      int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
      double fontscale = 1.0;
      int baseline;
      cv::Size textsize = cv::getTextSize(text, fontface, fontscale, 2,
                                      &baseline);
      cv::putText(gray, text, cv::Point(det->c[0]-textsize.width/2,
                                 det->c[1]+textsize.height/2),
              fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
#endif
  }
  zarray_destroy(detections);

  if(tag_seen[1] && tag_seen[2])
    pose_pub_.publish(tag_pose_array);
#if SHOW_IMAGE > 0
  if(!gray.empty())
    cv::imshow("Tag Detections", gray);
  cv::waitKey(1);
#endif
}


int main(int argc, char *argv[])
{
  ros::init(argc, argv, "apriltag_detector");
  ros::NodeHandle nh;
  AprilTagDetector ad(nh);

  
  while (ros::ok())
  {
    ros::spinOnce();
  }
  
  return 0;
}
