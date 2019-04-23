#ifndef SENSORS_HPP_
#define SENSORS_HPP_

// STL
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// ROS
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>

// CV Bridge
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "util/math.hpp"
#include "types.hpp"

class LaserScanner {
public:
    LaserScanner()
        : angle_min_(-1.5708), angle_max_(1.56466), angle_increment_(0.00613592),
          range_min_(0.02), range_max_(5.6)
    {
        T_wf_.linear() = Eigen::Matrix3d::Identity();
        T_wf_.translation() = Eigen::Vector3d(0.2062, 0.0, 0.28);
    }

    ~LaserScanner()
    { }

    static const double& angleMin() { return getInstance().angle_min_; }
    static const double& angleMax() { return getInstance().angle_max_; }
    static const double& angleIncrement() { return getInstance().angle_increment_; }

    static const double& rangeMin() { return getInstance().range_min_; }
    static const double& rangeMax() { return getInstance().range_max_; }

    static const Eigen::Isometry3d& T_wf() { return getInstance().T_wf_; }

    static laser_scan_t process(const sensor_msgs::LaserScanConstPtr& laser_scan) {
        laser_scan_t scan_data;

        scan_data.timestamp = timestamp(laser_scan->header.stamp);
        std::size_t idx = 0;
        for (double angle = angleMin(); angle <= angleMax(); angle += angleIncrement()) {
            double range = laser_scan->ranges[idx];
            if (range >= rangeMin() && range <= rangeMax())
                scan_data.points.push_back(PolarToCartesian<double>(range, angle));
            idx++;
        }

        return scan_data;
    }

   static std::vector<point3d_t> transform(const laser_scan_t& scan_data) {
       std::vector<point3d_t> points3d(scan_data.points.size());

       const Eigen::Isometry3d& T = T_wf();
       for (std::size_t idx = 0; idx < scan_data.points.size(); ++idx) {
           const point2d_t& point2d = scan_data.points[idx];
           points3d[idx] = T * point3d_t(point2d(0), point2d(1), 0.0);
       }

       return points3d;
   }

private:
    double angle_min_;
    double angle_max_;
    double angle_increment_;

    double range_min_;
    double range_max_;

    Eigen::Isometry3d T_wf_;

    static LaserScanner& getInstance() {
        static LaserScanner instance;
        return instance;
    }
};

class DepthCamera {
public:
    DepthCamera()
        : hfov_(58.59), vfov_(45.64),
          depth_min_(0.5), depth_max_(7.5)
    {
/* Camera Up */
        T_wf_.linear() = (Eigen::AngleAxisd(-1.5010, Eigen::Vector3d::UnitZ())
                          * Eigen::AngleAxisd(0.006192, Eigen::Vector3d::UnitY())
                          * Eigen::AngleAxisd(-2.323130, Eigen::Vector3d::UnitX()))
                         .toRotationMatrix();
        T_wf_.translation() = Eigen::Vector3d(0.09, 0.015, 1.572821);
/* Camera Down */
/*
        T_wf_.linear() = (Eigen::AngleAxisd(-1.5446, Eigen::Vector3d::UnitZ())
                          * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY())
                          * Eigen::AngleAxisd(-1.3708, Eigen::Vector3d::UnitX()))
                         .toRotationMatrix();
        T_wf_.translation() = Eigen::Vector3d(0.06, 0.015, 0.985);
*/
    }

    ~DepthCamera()
    { }

    static const double& horizontalFOV() { return getInstance().hfov_; }
    static const double& verticalFOV() { return getInstance().vfov_; }

    static const double& depthMin() { return getInstance().depth_min_; }
    static const double& depthMax() { return getInstance().depth_max_; }

    static const Eigen::Isometry3d& T_wf() { return getInstance().T_wf_; }

    static depth_image_t process(const sensor_msgs::ImageConstPtr& depth_image) {
       depth_image_t depth;

       RUNTIME_ASSERT(depth_image->encoding == sensor_msgs::image_encodings::TYPE_32FC1);

       cv_bridge::CvImageConstPtr cv_ptr;
       cv_ptr = cv_bridge::toCvShare(depth_image);
       RUNTIME_ASSERT(cv_ptr->image.type() == CV_32FC1);

       depth.timestamp = timestamp(depth_image->header.stamp);
       depth.image.resize(depth_image->height, depth_image->width);
       for (std::uint32_t i = 0; i < depth_image->height; ++i) {
           for (std::uint32_t j = 0; j < depth_image->width; ++j) {
               depth.image(i, j) = cv_ptr->image.at<float>(i, j);
           }
       }

       return depth;
    }

    static std::vector<point3d_t> transform(const depth_image_t& depth) {
            const unsigned int width = depth.image.cols();
            const unsigned int height = depth.image.rows();

            // Camera params
            const double cx = width / 2 - 0.5;
            const double cy = height / 2 - 0.5;
            const double inv_fx = (2.0 * std::tan(DegToRad(horizontalFOV() / 2.0))) / static_cast<double>(width);
            const double inv_fy = (2.0 * std::tan(DegToRad(verticalFOV() / 2.0))) / static_cast<double>(height);

            // Organized point cloud
            std::vector<point3d_t> cloud;
            for (unsigned int i = 0; i < height; ++i) {
                for (unsigned int j = 0; j < width; ++j) {
                    const float z = depth.image(i, j);

                    if (IsNaN(z) || IsInf(z)) continue;
                    if (z < depthMin() || z > depthMax()) continue;

                    const float y = (i - cy) * z * inv_fy;
                    const float x = (j - cx) * z * inv_fx;

                    point3d_t point3d = T_wf() * point3d_t(x, y, z);
                    if (point3d(2) < 0.1) continue;

                    cloud.push_back(point3d);
                }
            }

        return cloud;
    }

private:
    double hfov_;
    double vfov_;

    double depth_min_;
    double depth_max_;

    Eigen::Isometry3d T_wf_;

    static DepthCamera& getInstance() {
        static DepthCamera instance;
        return instance;
    }
};

#endif // SENSORS_HPP_
