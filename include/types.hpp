#ifndef TYPES_HPP_
#define TYPES_HPP_

// STL
#include <cstddef>
#include <cstdint>
#include <vector>

// Eigen
#include <Eigen/Core>

// ROS
#include <ros/ros.h>

#include "util/alignment.h"

// Types
using timestamp_t = std::uint64_t;

using point2d_t = Eigen::Vector2d;
using point3d_t = Eigen::Vector3d;

struct laser_scan_t {
    timestamp_t timestamp;
    std::vector<point2d_t> points;

    laser_scan_t()
        : timestamp(0), points()
    { }

    laser_scan_t(const timestamp_t timestamp, const std::vector<point2d_t>& points)
        : timestamp(timestamp), points(points)
    { }
};

struct depth_image_t {
    timestamp_t timestamp;
    Eigen::MatrixXf image;

    depth_image_t()
        : timestamp(0), image()
    { }

    depth_image_t(const timestamp_t timestamp, const Eigen::MatrixXf& image)
        : timestamp(timestamp), image(image)
    { }
};

inline timestamp_t timestamp(const ros::Time& t) {
    return static_cast<timestamp_t>(t.nsec) + static_cast<timestamp_t>(t.sec) * static_cast<timestamp_t>(1000000000ull);
}

inline bool operator<(const laser_scan_t& lhs, const laser_scan_t& rhs) {
    return (lhs.timestamp < rhs.timestamp);
}

inline bool operator<(const laser_scan_t& lhs, const timestamp_t rhs) {
    return (lhs.timestamp < rhs);
}

inline bool operator<(const depth_image_t& lhs, const depth_image_t& rhs) {
    return (lhs.timestamp < rhs.timestamp);
}

inline bool operator<(const depth_image_t& lhs, const timestamp_t rhs) {
    return (lhs.timestamp < rhs);
}

#endif // TYPES_HPP_
