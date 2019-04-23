
#define PROGRAM_NAME \
    "local_distortion"

#define FLAGS_CASES                                                                                \
    FLAG_CASE(double, start, 0.0, "Start time [s]")                                                \
    FLAG_CASE(double, end, -1.0, "End time [s]")                                                   \
    FLAG_CASE(bool, compensate, true, "Compensate depth errors")


#define ARGS_CASES                                                                                 \
    ARG_CASE(rosbag)                                                                               \
    ARG_CASE(laser_topic)                                                                          \
    ARG_CASE(depth_topic)                                                                          \
    ARG_CASE(calibration)

// STL
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
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
using Plane_RANSAC = RANSAC<PlaneEstimator>;

void ValidateArgs() {
    RUNTIME_ASSERT(boost::filesystem::is_regular_file(ARGS_rosbag));
    RUNTIME_ASSERT(boost::filesystem::is_regular_file(ARGS_calibration));
}

void ValidateFlags() {
    if (FLAGS_end < FLAGS_start) FLAGS_end = std::numeric_limits<double>::max();
}

/*
double interpolate(const double a, const double b, const double t) {
    RUNTIME_ASSERT(t >= 0.0 && t <= 1.0);

    return a + t*(b - a);
}

void depthCorrection(depth_image_t& depth, std::shared_ptr<matrix<calibration_t>> calibration) {
    if (calibration == nullptr) return;

    const double di = static_cast<double>(depth.image.rows()) / static_cast<double>(calibration->rows);
    const double dj = static_cast<double>(depth.image.cols()) / static_cast<double>(calibration->cols);

    for (std::size_t i = 0; i < depth.image.rows(); ++i) {
        for (std::size_t j = 0; j < depth.image.cols(); ++j) {
            const double z = depth.image(i, j);
            if (IsNaN(z) || IsInf(z)) continue;

            std::size_t i0 = i / di;
            std::size_t j0 = j / dj;

            // Bounds checking
            if ((i0 + 1) < calibration->rows && (j0 + 1) < calibration->cols) {
                std::size_t i1 = i0 + 1;
                std::size_t j1 = j0 + 1;

                if (!(calibration->at(i0, j0).meaningful && calibration->at(i1, j0).meaningful &&
                    calibration->at(i0, j1).meaningful && calibration->at(i1, j1).meaningful)) continue;

                const double a = calibration->at(i0, j0).evaluate(z);
                const double b = calibration->at(i1, j0).evaluate(z);
                const double c = calibration->at(i0, j1).evaluate(z);
                const double d = calibration->at(i1, j1).evaluate(z);

                const double ti = (static_cast<double>(i) - i0*di) / di;
                const double ab = interpolate(a, b, ti);
                const double cd = interpolate(c, d, ti);

                depth.image(i, j) = z - interpolate(ab, cd, (static_cast<double>(j) - j0*dj) / dj);
            }
        }
    }
}

void getPointCloud(const depth_image_t& depth, pcl::PointCloud<point_t>::Ptr cloud) {
    RUNTIME_ASSERT(cloud != nullptr);

    const double nand = std::numeric_limits<double>::quiet_NaN();

    const unsigned int width = depth.image.cols();
    const unsigned int height = depth.image.rows();

    // Camera params
    const double cx = width / 2 - 0.5;
    const double cy = height / 2 - 0.5;
    const double inv_fx = (2.0 * std::tan(DegToRad(DepthCamera::horizontalFOV() / 2.0))) / static_cast<double>(width);
    const double inv_fy = (2.0 * std::tan(DegToRad(DepthCamera::verticalFOV() / 2.0))) / static_cast<double>(height);

    // Organized point cloud
    cloud->width = width;
    cloud->height = height;
    cloud->resize(width * height);

    cloud->is_dense = false;
    for (unsigned int i = 0; i < height; ++i) {
        for (unsigned int j = 0; j < width; ++j) {
            const double z = depth.image(i, j);

            if (IsNaN(z) || IsInf(z) ||
                z < DepthCamera::depthMin() || z > DepthCamera::depthMax()) {
                cloud->at(j, i) = point_t(nand, nand, nand);
                continue;
            }

            const double y = (i - cy) * z * inv_fy;
            const double x = (j - cx) * z * inv_fx;

            point3d_t point3d = DepthCamera::T_wf() * point3d_t(x, y, z);
            if (point3d(2) < 0.1) {
                cloud->at(j, i) = point_t(nand, nand, nand);
                continue;
            }

            cloud->at(j, i) = point_t(point3d(0), point3d(1), point3d(2));
        }
    }
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

    // Read calibration
    std::shared_ptr<matrix<calibration_t>> calibration = read_calib(ARGS_calibration);

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

    for (const auto& entry : associations) {
        const laser_scan_t& scan_data = laser_scans.at(entry.first);

        Eigen::Vector3d line = Eigen::Vector3d::Zero();
        {
            RANSACOptions ransac_options;
            ransac_options.max_error = 0.005;
            ransac_options.min_inlier_ratio = 0.25;
            ransac_options.confidence = 0.9999;
            ransac_options.min_num_trials =  100;
            ransac_options.max_num_trials = 1000;

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
            line = models.front();
        }

        Eigen::Vector4d p(line(0), line(1), 0.0, line(2));
        p = transform_plane(hesseNormalForm(p), LaserScanner::T_wf());
        p = transform_plane(p, DepthCamera::T_wf().inverse());

        if (std::abs(p(3)) < 1.0 || std::abs(p(3)) > 4.0) continue;

        depth_image_t& depth_data = depth_images.at(entry.second);
        RUNTIME_ASSERT(depth_data.image.cols() == width);
        RUNTIME_ASSERT(depth_data.image.rows() == height);

        std::vector<Eigen::Vector3d> points;
        for (unsigned int i = 0; i < height; i += 4) {
            for (unsigned int j = 0; j < width; j += 4) {
                const double z = depth_data.image(i, j);
                if (IsNaN(z) || IsInf(z)) continue;
                if (z < 1.0 || z > 4.0) continue; // TODO Bug yielding incorrect estimations!

                const double coef_x = (j - cx) * inv_fx;
                const double coef_y = (i - cy) * inv_fy;

                point3d_t l(coef_x, coef_y, 1.0);

                point3d_t point3d = DepthCamera::T_wf() * (l * z);
                if (point3d(2) < 0.1) continue;

                const double z_truth = -p(3) / p.head<3>().dot(l);

                if (calibration->at(i / 4, j / 4).meaningful) {
                    if (FLAGS_compensate) {
                        const double z_ = calibration->at(i / 4, j / 4).correct(z);
                        points.push_back(l * z_);
                    } else {
                        points.push_back(l * z);
                    }
                }
            }
        }

        if (points.size() < 10) continue;

        Eigen::Vector4d plane = Eigen::Vector4d::Zero();
        {
            RANSACOptions ransac_options;
            ransac_options.max_error = 0.03;
            ransac_options.min_inlier_ratio = 0.2;
            ransac_options.confidence = 0.9999;
            ransac_options.min_num_trials =  100;
            ransac_options.max_num_trials = 1000;

            Plane_RANSAC::Report ransac_report;

            Plane_RANSAC ransac(ransac_options);
            ransac_report = ransac.Estimate(points);
            if (!ransac_report.success) continue;

            std::vector<point3d_t> inliers;
            RUNTIME_ASSERT(points.size() == ransac_report.inlier_mask.size());
            for (std::size_t k = 0; k < points.size(); ++k) {
                if (ransac_report.inlier_mask.at(k)) {
                    inliers.push_back(points[k]);
                }
            }

            if (inliers.size() < ransac_options.min_inlier_ratio * points.size()) continue;

/*
            std::vector<PlaneEstimator::M_t> models = PlaneEstimator::Estimate(inliers);
            if (models.empty()) continue;
            plane = models.front();
*/
            plane = ransac_report.model;
        }

        double sum = 0.0;
        std::size_t k = 0;

        for (unsigned int i = 0; i < height; i += 4) {
            for (unsigned int j = 0; j < width; j += 4) {
                const double z = depth_data.image(i, j);
                if (IsNaN(z) || IsInf(z)) continue;
                if (z < 1.0 || z > 4.0) continue; // TODO Bug yielding incorrect estimations!

                const double coef_x = (j - cx) * inv_fx;
                const double coef_y = (i - cy) * inv_fy;

                point3d_t l(coef_x, coef_y, 1.0);

                point3d_t point3d = DepthCamera::T_wf() * (l * z);
                if (point3d(2) < 0.1) continue;

                const double z_truth = -p(3) / p.head<3>().dot(l);

                if (calibration->at(i / 4, j / 4).meaningful) {
                    double e = std::numeric_limits<double>::quiet_NaN();
                    if (FLAGS_compensate) {
                        const double z_ = calibration->at(i / 4, j / 4).correct(z);
                        e = (plane.head<3>().dot(l * z_) + plane(3)) / plane.head<3>().norm();
                    } else {
                        e = (plane.head<3>().dot(l * z) + plane(3)) / plane.head<3>().norm();
                    }

                    sum += e*e;
                    k++;
                }
            }
        }

        if (k > 0) {
            const double error = std::sqrt(sum / static_cast<double>(k));
            std::cout << std::abs(p(3)) << "," << error << std::endl;
        }
    }

    return 0;
}
