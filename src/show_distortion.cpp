
#define PROGRAM_NAME \
    "show_distortion"

#define FLAGS_CASES                                                                                \
    FLAG_CASE(uint64, pixel_skip, 3, "Pixel decimation")                                           \
    FLAG_CASE(double, start, 0.0, "Start time [s]")                                                \
    FLAG_CASE(double, end, -1.0, "End time [s]")

#define ARGS_CASES                                                                                 \
    ARG_CASE(rosbag)                                                                               \
    ARG_CASE(laser_topic)                                                                          \
    ARG_CASE(depth_topic)

#include "mrpt_scene.hpp"

// STL
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

// Boost
#include <boost/filesystem/operations.hpp>
#include <boost/math/special_functions/sign.hpp>

// ROS
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>

#include "util/alignment.h"
#include "util/args.hpp"
#include "util/endian.hpp"
#include "util/macros.h"
#include "util/math.hpp"
#include "util/statistical.hpp"
#include "associate.hpp"
#include "calibration.hpp"
#include "estimators.hpp"
#include "ransac.hpp"
#include "sensors.hpp"
#include "types.hpp"

using Line_RANSAC = RANSAC<LineEstimator>;

void ValidateArgs() {
    RUNTIME_ASSERT(boost::filesystem::is_regular_file(ARGS_rosbag));
}

void ValidateFlags() {
    if (FLAGS_end < FLAGS_start) FLAGS_end = std::numeric_limits<double>::max();
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> Rz(T yaw_radians) {
    const T cos_yaw = std::cos(yaw_radians);
    const T sin_yaw = std::sin(yaw_radians);

    Eigen::Matrix<T, 3, 3> rotation;
    rotation << cos_yaw, -sin_yaw, 0.0,
                sin_yaw, cos_yaw, 0.0,
                0.0, 0.0, 1.0;
    return rotation;
}

int main(int argc, char* argv[]) {

    // Handle help flag
    if (args::HelpRequired(argc, argv)) {
        args::ShowHelp();
        return 0;
    }

    // Parse input flags
    args::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    // Check number of args
    if (argc-1 != args::NumArgs()) {
        args::ShowHelp();
        return -1;
    }

    // Parse input args
    args::ParseCommandLineArgs(argc, argv);

    // Validate input arguments
    ValidateFlags();
    ValidateArgs();

    MRPTScene scene;

    RANSACOptions ransac_options;
    ransac_options.max_error = 0.005;
    ransac_options.min_inlier_ratio = 0.25;
    ransac_options.confidence = 0.9999;
    ransac_options.min_num_trials =  100;
    ransac_options.max_num_trials = 1000;

    rosbag::Bag bag;
    bag.open(ARGS_rosbag, rosbag::bagmode::Read);

    std::vector<std::string> laser_topic = {ARGS_laser_topic};
    rosbag::View laser_view(bag, rosbag::TopicQuery(laser_topic));

    std::vector<laser_scan_t> laser_scans;
    for (rosbag::View::iterator it = laser_view.begin(); it != laser_view.end(); ++it) {
        const rosbag::MessageInstance& m = *it;
        sensor_msgs::LaserScanConstPtr scan = m.instantiate<sensor_msgs::LaserScan>();
        if (scan != nullptr) {
            if (scan->header.stamp.toSec() < FLAGS_start) continue;
            if (scan->header.stamp.toSec() >= FLAGS_end) continue;
            laser_scan_t scan_data = LaserScanner::process(scan);
            laser_scans.push_back(scan_data);
        }
    }

    RUNTIME_ASSERT(!laser_scans.empty());
    std::sort(laser_scans.begin(), laser_scans.end());

    std::vector<std::string> depth_topic = {ARGS_depth_topic};
    rosbag::View depth_view(bag, rosbag::TopicQuery(depth_topic));

    std::vector<depth_image_t> depth_images;
    for (rosbag::View::iterator it = depth_view.begin(); it != depth_view.end(); ++it) {
        const rosbag::MessageInstance& m = *it;
        sensor_msgs::ImageConstPtr image = m.instantiate<sensor_msgs::Image>();
        if (image != nullptr) {
            if (image->header.stamp.toSec() < FLAGS_start) continue;
            if (image->header.stamp.toSec() >= FLAGS_end) continue;
            depth_image_t depth_data = DepthCamera::process(image);
            depth_images.push_back(depth_data);
        }
    }

    RUNTIME_ASSERT(!depth_images.empty());
    std::sort(depth_images.begin(), depth_images.end());

    // Camera params
    const unsigned int width = depth_images.front().image.cols();
    const unsigned int height = depth_images.front().image.rows();

    const double cx = width / 2 - 0.5;
    const double cy = height / 2 - 0.5;
    const double inv_fx = (2.0 * std::tan(DegToRad(DepthCamera::horizontalFOV() / 2.0))) / static_cast<double>(width);
    const double inv_fy = (2.0 * std::tan(DegToRad(DepthCamera::verticalFOV() / 2.0))) / static_cast<double>(height);

    std::unordered_map<std::size_t, std::size_t> associations = associate(laser_scans, depth_images, 15000000ull);
    //COUT_LOG(associations.size());

{
    double target_distance = 1.0;
    double best = std::numeric_limits<double>::max();
    bool found = false;

    Eigen::Vector3d line;
    laser_scan_t scan;
    depth_image_t depth_image;

    //std::vector<calibration_observation_ptr> observations;
    matrix<std::vector<calibration_observation_ptr>> pixels(height / (FLAGS_pixel_skip + 1),
                                                            width / (FLAGS_pixel_skip + 1));
    for (const auto& entry : associations) {
        const laser_scan_t& scan_data = laser_scans.at(entry.first);

        Line_RANSAC::Report ransac_report;

        Line_RANSAC ransac(ransac_options);
        ransac_report = ransac.Estimate(scan_data.points);
        if (!ransac_report.success) continue;

        std::vector<point2d_t> inliers;
        RUNTIME_ASSERT(scan_data.points.size() == ransac_report.inlier_mask.size());
        for (std::size_t k = 0; k < scan_data.points.size(); ++k) {
            if (ransac_report.inlier_mask.at(k)) {
                inliers.push_back(scan_data.points[k]);
            }
        }

        std::vector<LineEstimator::M_t> models = LineEstimator::Estimate(inliers);
        if (models.empty()) continue;

        Eigen::Vector4d p(models.front()(0), models.front()(1), 0.0, models.front()(2));
        p = transform_plane(hesseNormalForm(p), LaserScanner::T_wf());

        if (std::abs(target_distance - std::abs(p(3))) < best ) {
            found = true;
            best = std::abs(target_distance - std::abs(p(3)));

            line = models.front();
            scan = scan_data;
            depth_image = depth_images.at(entry.second);
        }
    }

    double yaw = -boost::math::sign(line(1)) * std::acos(line.head<2>().dot(Eigen::Vector2d::UnitX()) / line.head<2>().norm());
    Eigen::Matrix3d R = Rz<double>(yaw);

    //for (const point3d_t& point : LaserScanner::transform(scan))
        //scene.addPoint(R * point, MRPTScene::COLOR_BLACK);
    for (double y = -2.31; y < 2.31; y += 0.01) // 2.31 = max_distance * tan(hfov / 2)
        scene.addPoint(Eigen::Vector3d(std::abs(line(2)) + 0.2062, y, 0.28), MRPTScene::COLOR_BLACK);

    for (const point3d_t& point : DepthCamera::transform(depth_image))
        scene.addPoint(R * point, MRPTScene::COLOR_BLUE);
}

{
    double target_distance = 2.0;
    double best = std::numeric_limits<double>::max();
    bool found = false;

    Eigen::Vector3d line;
    laser_scan_t scan;
    depth_image_t depth_image;

    //std::vector<calibration_observation_ptr> observations;
    matrix<std::vector<calibration_observation_ptr>> pixels(height / (FLAGS_pixel_skip + 1),
                                                            width / (FLAGS_pixel_skip + 1));
    for (const auto& entry : associations) {
        const laser_scan_t& scan_data = laser_scans.at(entry.first);

        Line_RANSAC::Report ransac_report;

        Line_RANSAC ransac(ransac_options);
        ransac_report = ransac.Estimate(scan_data.points);
        if (!ransac_report.success) continue;

        std::vector<point2d_t> inliers;
        RUNTIME_ASSERT(scan_data.points.size() == ransac_report.inlier_mask.size());
        for (std::size_t k = 0; k < scan_data.points.size(); ++k) {
            if (ransac_report.inlier_mask.at(k)) {
                inliers.push_back(scan_data.points[k]);
            }
        }

        std::vector<LineEstimator::M_t> models = LineEstimator::Estimate(inliers);
        if (models.empty()) continue;

        Eigen::Vector4d p(models.front()(0), models.front()(1), 0.0, models.front()(2));
        p = transform_plane(hesseNormalForm(p), LaserScanner::T_wf());

        if (std::abs(target_distance - std::abs(p(3))) < best ) {
            found = true;
            best = std::abs(target_distance - std::abs(p(3)));

            line = models.front();
            scan = scan_data;
            depth_image = depth_images.at(entry.second);
        }
    }

    double yaw = -boost::math::sign(line(1)) * std::acos(line.head<2>().dot(Eigen::Vector2d::UnitX()) / line.head<2>().norm());
    Eigen::Matrix3d R = Rz<double>(yaw);

    //for (const point3d_t& point : LaserScanner::transform(scan))
        //scene.addPoint(R * point, MRPTScene::COLOR_BLACK);
    for (double y = -2.31; y < 2.31; y += 0.01) // 2.31 = max_distance * tan(hfov / 2)
        scene.addPoint(Eigen::Vector3d(std::abs(line(2)) + 0.2062, y, 0.28), MRPTScene::COLOR_BLACK);

    for (const point3d_t& point : DepthCamera::transform(depth_image))
        scene.addPoint(R * point, MRPTScene::COLOR_BLUE);
}

{
    double target_distance = 3.0;
    double best = std::numeric_limits<double>::max();
    bool found = false;

    Eigen::Vector3d line;
    laser_scan_t scan;
    depth_image_t depth_image;

    //std::vector<calibration_observation_ptr> observations;
    matrix<std::vector<calibration_observation_ptr>> pixels(height / (FLAGS_pixel_skip + 1),
                                                            width / (FLAGS_pixel_skip + 1));
    for (const auto& entry : associations) {
        const laser_scan_t& scan_data = laser_scans.at(entry.first);

        Line_RANSAC::Report ransac_report;

        Line_RANSAC ransac(ransac_options);
        ransac_report = ransac.Estimate(scan_data.points);
        if (!ransac_report.success) continue;

        std::vector<point2d_t> inliers;
        RUNTIME_ASSERT(scan_data.points.size() == ransac_report.inlier_mask.size());
        for (std::size_t k = 0; k < scan_data.points.size(); ++k) {
            if (ransac_report.inlier_mask.at(k)) {
                inliers.push_back(scan_data.points[k]);
            }
        }

        std::vector<LineEstimator::M_t> models = LineEstimator::Estimate(inliers);
        if (models.empty()) continue;

        Eigen::Vector4d p(models.front()(0), models.front()(1), 0.0, models.front()(2));
        p = transform_plane(hesseNormalForm(p), LaserScanner::T_wf());

        if (std::abs(target_distance - std::abs(p(3))) < best ) {
            found = true;
            best = std::abs(target_distance - std::abs(p(3)));

            line = models.front();
            scan = scan_data;
            depth_image = depth_images.at(entry.second);
        }
    }

    double yaw = -boost::math::sign(line(1)) * std::acos(line.head<2>().dot(Eigen::Vector2d::UnitX()) / line.head<2>().norm());
    Eigen::Matrix3d R = Rz<double>(yaw);

    //for (const point3d_t& point : LaserScanner::transform(scan))
        //scene.addPoint(R * point, MRPTScene::COLOR_BLACK);
    for (double y = -2.31; y < 2.31; y += 0.01) // 2.31 = max_distance * tan(hfov / 2)
        scene.addPoint(Eigen::Vector3d(std::abs(line(2)) + 0.2062, y, 0.28), MRPTScene::COLOR_BLACK);

    for (const point3d_t& point : DepthCamera::transform(depth_image))
        scene.addPoint(R * point, MRPTScene::COLOR_BLUE);
}

    scene.repaint();
    scene.waitForKey();
    scene.clear();

    return 0;
}
