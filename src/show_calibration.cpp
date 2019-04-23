
#define PROGRAM_NAME \
    "show_calibration"

#define FLAGS_CASES                                                                                \

#define ARGS_CASES                                                                                 \
    ARG_CASE(calibration)

// STL
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <limits>

// Boost
#include <boost/filesystem/operations.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "util/alignment.h"
#include "util/args.hpp"

#include "calibration.hpp"

void ValidateArgs() {
    RUNTIME_ASSERT(boost::filesystem::is_regular_file(ARGS_calibration));
}

void ValidateFlags() {
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

    std::shared_ptr<matrix<calibration_t>> calibration = read_calib(ARGS_calibration);

    std::size_t max_cnt = 0;
    double min_z = 0.0, max_z = 0.0;
    for (std::size_t i = 0; i < calibration->rows; ++i) {
        for (std::size_t j = 0; j < calibration->cols; ++j) {
            if (calibration->at(i, j).z_min > min_z)
                min_z = calibration->at(i, j).z_min;
            if (calibration->at(i, j).z_max > max_z)
                max_z = calibration->at(i, j).z_max;
            if (calibration->at(i, j).z_cnt > max_cnt)
                max_cnt = calibration->at(i, j).z_cnt;
        }
    }

/*
    std::cout << "Max min depth: " << min_z << std::endl;
    std::cout << "Max max depth: " << max_z << std::endl;
*/
    std::cout << "Max depth: " << std::max(min_z, max_z) << std::endl;
    std::cout << "Max count: " << max_cnt << std::endl;

    const double scale_minz = 255.0 / std::max(min_z, max_z);
    const double scale_maxz = 255.0 / std::max(min_z, max_z);
    const double scale_cnt = 255.0 / static_cast<double>(max_cnt);

    cv::Mat img_calib(calibration->rows, calibration->cols, CV_8U);
    cv::Mat img_minmax(calibration->rows, calibration->cols, CV_8UC3);
    cv::Mat img_cnt(calibration->rows, calibration->cols, CV_8U);
    for (std::size_t i = 0; i < calibration->rows; ++i) {
        for (std::size_t j = 0; j < calibration->cols; ++j) {
            img_calib.at<std::uint8_t>(i, j) = 255 * calibration->at(i, j).meaningful;
            img_minmax.at<cv::Vec3b>(i, j)[0] = calibration->at(i, j).z_min * scale_minz;
            img_minmax.at<cv::Vec3b>(i, j)[2] = calibration->at(i, j).z_max * scale_maxz;
            img_cnt.at<std::uint8_t>(i, j) = calibration->at(i, j).z_cnt * scale_cnt;
        }
    }

    cv::namedWindow("Calibrated", CV_WINDOW_KEEPRATIO);
    cv::imshow("Calibrated", img_calib);

    cv::namedWindow("Depth range", CV_WINDOW_KEEPRATIO);
    cv::imshow("Depth range", img_minmax);

    cv::namedWindow("Count", CV_WINDOW_KEEPRATIO);
    cv::imshow("Count", img_cnt);

    cv::waitKey();

    return 0;
}