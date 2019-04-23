
#define PROGRAM_NAME \
    "show_distortion"

#define FLAGS_CASES                                                                                \
    FLAG_CASE(uint64, pixel_skip, 3, "Pixel decimation")                                           \
    FLAG_CASE(double, start, 0.0, "Start time [s]")                                                \
    FLAG_CASE(double, end, -1.0, "End time [s]")

#define ARGS_CASES                                                                                 \
    ARG_CASE(rosbag)                                                                               \
    ARG_CASE(laser_topic)                                                                          \
    ARG_CASE(depth_topic)                                                                          \
    ARG_CASE(output_file)

#include "mrpt_scene.hpp"

// STL
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

// Boost
#include <boost/filesystem/operations.hpp>

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
    RUNTIME_ASSERT(!boost::filesystem::exists(ARGS_output_file));
}

void ValidateFlags() {
    if (FLAGS_end < FLAGS_start) FLAGS_end = std::numeric_limits<double>::max();
}

/*
std::uint64_t hash(std::uint64_t a, std::uint64_t b) {
    RUNTIME_ASSERT(a < 4294967295ull);
    RUNTIME_ASSERT(b < 4294967295ull);
    return (a << 32) + b;
}

void inv_hash(std::uint64_t hash, std::uint32_t& a, std::uint32_t& b) {
    a = static_cast<std::uint32_t>(hash >> 32);
    b = static_cast<std::uint32_t>(hash & 4294967295ull);
}
*/

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

//            std::cout << scan_data.timestamp << std::endl;
/*
            for (const Eigen::Vector2d& point : scan_data.points)
                scene.addPoint(LaserScanner::transform(point), MRPTScene::COLOR_BLACK);
*/

/*
            Line_RANSAC::Report ransac_report;

            Line_RANSAC ransac(ransac_options);
            ransac_report = ransac.Estimate(scan_data.points);
            RUNTIME_ASSERT(ransac_report.success);

            RUNTIME_ASSERT(scan_data.points.size() == ransac_report.inlier_mask.size());
            std::vector<point2d_t> inliers;
            std::vector<point3d_t> points3d = LaserScanner::transform(scan_data);
            for (std::size_t k = 0; k < scan_data.points.size(); ++k) {
                if (ransac_report.inlier_mask.at(k)) {
                    inliers.push_back(scan_data.points[k]);
                    scene.addPoint(points3d[k], MRPTScene::COLOR_BLACK);
                }
            }

            std::cout << "Num. inliers: " << ransac_report.support.num_inliers << std::endl;
            std::cout << ransac_report.model.transpose() << std::endl;

            std::vector<LineEstimator::M_t> models = LineEstimator::Estimate(inliers);
            std::cout << "Models: " << models.size() << std::endl;
            for (const LineEstimator::M_t& m : models)
                std::cout << m.transpose() << std::endl;

            scene.repaint();
            scene.waitForKey();
            scene.clear();
*/
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

/*
            std::vector<point3d_t> camera_points = DepthCamera::transform(depth_data);
            for (const point3d_t& point : camera_points)
                scene.addPoint(point, MRPTScene::COLOR_BLACK);

            scene.repaint();
            scene.waitForKey();
            scene.clear();
*/
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

    //std::vector<calibration_observation_ptr> observations;
    matrix<std::vector<calibration_observation_ptr>> pixels(height / (FLAGS_pixel_skip + 1),
                                                            width / (FLAGS_pixel_skip + 1));
    for (const auto& entry : associations) {
/*
        for (const point3d_t& point : LaserScanner::transform(laser_scans.at(entry.first)))
            scene.addPoint(point, MRPTScene::COLOR_BLACK);
        for (const point3d_t& point : DepthCamera::transform(depth_images.at(entry.second)))
            scene.addPoint(point, MRPTScene::COLOR_BLUE);
*/

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
        p = transform_plane(p, DepthCamera::T_wf().inverse());

        if (std::abs(p(3)) < 1.0 || std::abs(p(3)) > 4.0) continue;

        const depth_image_t& depth_data = depth_images.at(entry.second);
        RUNTIME_ASSERT(depth_data.image.cols() == width);
        RUNTIME_ASSERT(depth_data.image.rows() == height);

//        unsigned int i = 30 * (FLAGS_pixel_skip + 1), j = 40 * (FLAGS_pixel_skip + 1);

        for (unsigned int i = 0; i < height; i += FLAGS_pixel_skip + 1) {
            for (unsigned int j = 0; j < width; j += FLAGS_pixel_skip + 1) {
                const double z = depth_data.image(i, j);
                if (IsNaN(z) || IsInf(z)) continue;
                if (z < 1.0 || z > 4.0) continue; // TODO Bug yielding incorrect estimations!

                const double coef_x = (j - cx) * inv_fx;
                const double coef_y = (i - cy) * inv_fy;

                point3d_t point3d = DepthCamera::T_wf() * point3d_t(coef_x * z, coef_y * z, z);
                if (point3d(2) < 0.1) continue;

                const double z_ = -p(3) / p.head<3>().dot(Eigen::Vector3d(coef_x, coef_y, 1.0));

                //std::cout << z << "," << z_ << std::endl;
                calibration_observation_ptr obs_ptr = std::make_shared<calibration_observation_t>(z, z_);
                pixels.at(i / (FLAGS_pixel_skip + 1), j / (FLAGS_pixel_skip + 1)).push_back(obs_ptr);
            }
        }
/*
        for (const point3d_t& point : LaserScanner::transform(scan_data))
            scene.addPoint(point, MRPTScene::COLOR_BLACK);
        for (unsigned int i = 0; i < height; ++i) {
            for (unsigned int j = 0; j < width; ++j) {
                const double z = depth_data.image(i, j);
                if (IsNaN(z) || IsInf(z)) continue;

                const double coef_x = (j - cx) * inv_fx;
                const double coef_y = (i - cy) * inv_fy;

                const double z_ = -p(3) / p.head<3>().dot(Eigen::Vector3d(coef_x, coef_y, 1.0));

                scene.addPoint(DepthCamera::T_wf() * Eigen::Vector3d(coef_x*z_, coef_y*z_, z_),
                               MRPTScene::COLOR_BLUE);
            }
        }

        scene.repaint();
        scene.waitForKey();
        scene.clear();
*/
    }

    NoiseStdvFunc f = noise_calibration(pixels, 0.1, 1.0, 4.0);
    //NoiseStdvFunc f = single_noise_calibration(observations, 0.1, 1.0, 4.0);
    //std::cout << f.coeff.transpose() << std::endl;

    matrix<calibration_t> calibration(pixels.rows, pixels.cols);

    std::size_t k = 0;
    for (std::size_t i = 0; i < pixels.rows; ++i) {
        for (std::size_t j = 0; j < pixels.cols; ++j) {
            if (!pixels.at(i, j).empty()) {
                // TODO and stdv measurements > 2/sqrt(3)?
                RUNTIME_ASSERT(!pixels.at(i, j).empty());

                std::uint64_t z_cnt = pixels.at(i, j).size();
                measurement_t z_min = std::numeric_limits<measurement_t>::max();
                measurement_t z_max = std::numeric_limits<measurement_t>::min();
                for (std::size_t idx = 0; idx < pixels.at(i, j).size(); ++idx) {
                    const measurement_t z = pixels.at(i, j)[idx]->measurement;
                    if (z < z_min) z_min = z;
                    if (z > z_max) z_max = z;
                }

                std::uint8_t meaningful = 0;
                Eigen::Vector3d coeff = Eigen::Vector3d::Zero();
                if (pixels.at(i, j).size() > 25) {
                    meaningful = 1;
                    coeff = depth_calibration(pixels.at(i, j), f);
                }

                calibration.at(i, j) = calibration_t(meaningful, coeff(0), coeff(1), coeff(2), z_min, z_max, z_cnt);
                ++k;
            } else
                calibration.at(i, j) = calibration_t(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0);
        }
    }

    // Write calibration
    std::ios::openmode mode = std::ios::out | std::ios::binary;

    std::ofstream output_stream(ARGS_output_file, mode);
    RUNTIME_ASSERT(output_stream.is_open());

    WriteBinaryLittleEndian<std::size_t>(&output_stream, calibration.rows);
    WriteBinaryLittleEndian<std::size_t>(&output_stream, calibration.cols);
    for (std::size_t i = 0; i < calibration.rows; ++i) {
        for (std::size_t j = 0; j < calibration.cols; ++j) {
            WriteBinaryLittleEndian<std::uint8_t>(&output_stream, calibration.at(i, j).meaningful);
            WriteBinaryLittleEndian<double>(&output_stream, calibration.at(i, j).a);
            WriteBinaryLittleEndian<double>(&output_stream, calibration.at(i, j).b);
            WriteBinaryLittleEndian<double>(&output_stream, calibration.at(i, j).c);
            WriteBinaryLittleEndian<double>(&output_stream, calibration.at(i, j).z_min);
            WriteBinaryLittleEndian<double>(&output_stream, calibration.at(i, j).z_max);
            WriteBinaryLittleEndian<std::uint64_t>(&output_stream, calibration.at(i, j).z_cnt);
        }
    }

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
        p = transform_plane(p, DepthCamera::T_wf().inverse());

        if (std::abs(p(3)) < 1.0 || std::abs(p(3)) > 4.0) continue;

        const depth_image_t& depth_data = depth_images.at(entry.second);

        for (unsigned int i = 0; i < height; i += FLAGS_pixel_skip + 1) {
            for (unsigned int j = 0; j < width; j += FLAGS_pixel_skip + 1) {
                const double z = depth_data.image(i, j);
                if (IsNaN(z) || IsInf(z)) continue;
                if (z < 1.0 || z > 4.0) continue; // TODO Bug yielding incorrect estimations!
                if (!calibration.at(i / (FLAGS_pixel_skip + 1), j / (FLAGS_pixel_skip + 1)).meaningful) continue;

                const double z_ = calibration.at(i / (FLAGS_pixel_skip + 1), j / (FLAGS_pixel_skip + 1)).correct(z);

                Eigen::Vector3d p((j - cx) * inv_fx, (i - cy) * inv_fy, 1.0);

                point3d_t point3d = DepthCamera::T_wf() * (p * z);
                if (point3d(2) < 0.1) continue;

                scene.addPoint(DepthCamera::T_wf() * (p * z),
                               MRPTScene::COLOR_RED);

                scene.addPoint(DepthCamera::T_wf() * (p * z_),
                               MRPTScene::COLOR_BLUE);
            }
        }

        for (const point3d_t& point : LaserScanner::transform(scan_data))
            scene.addPoint(point, MRPTScene::COLOR_BLACK);

        scene.repaint();
        scene.waitForKey();
        scene.clear();
    }

    return 0;
}
