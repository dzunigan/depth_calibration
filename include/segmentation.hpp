// STL
#include <cstddef>
#include <memory>
#include <vector>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/fast_bilateral.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>

// Eigen
#include <Eigen/Core>

#include "util/alignment.h"
#include "util/math.hpp"

inline unsigned int planeSegmentation(pcl::PointCloud<point_t>::Ptr cloud, std::shared_ptr<std::vector<int>> segments, std::shared_ptr<std::vector<Eigen::Vector4d>> params) {
    double sigma_s = 10.0;
    double sigma_r = 0.005;

    double max_change_factor = 0.02;
    double smoothing_size = 10.0;

    double inliers_ratio = 0.10;
    double angular_th = 0.5;
    double distance_th = 0.1;
    double max_curvature = 0.1;

    // Filter point cloud
    pcl::FastBilateralFilter<point_t> bf;
    bf.setSigmaS(sigma_s);
    bf.setSigmaR(sigma_r);

    pcl::PointCloud<point_t>::Ptr filtered_cloud(new pcl::PointCloud<point_t>);
    bf.setInputCloud(cloud);
    bf.filter(*filtered_cloud);
    filtered_cloud = cloud;

    // Compute normals
    pcl::IntegralImageNormalEstimation<point_t, pcl::Normal> ne;
    ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
    //ne.setNormalEstimationMethod(ne.SIMPLE_3D_GRADIENT);
    //ne.setNormalEstimationMethod(ne.AVERAGE_DEPTH_CHANGE);
    //ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(max_change_factor);
    ne.setNormalSmoothingSize(smoothing_size);
    ne.setDepthDependentSmoothing(true);

    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>);
    ne.setInputCloud(filtered_cloud);
    ne.compute(*normal_cloud);

    // Plane segmentation
    pcl::OrganizedMultiPlaneSegmentation<point_t, pcl::Normal, pcl::Label> mps;
    mps.setMinInliers(filtered_cloud->size()*inliers_ratio);
    mps.setAngularThreshold(angular_th);
    mps.setDistanceThreshold(distance_th);
    mps.setMaximumCurvature(max_curvature);
    mps.setInputNormals(normal_cloud);
    mps.setInputCloud(filtered_cloud);

    std::vector<pcl::PlanarRegion<point_t>, Eigen::aligned_allocator<pcl::PlanarRegion<point_t>>> regions;
    //mps.segmentAndRefine(regions);
    std::vector<pcl::ModelCoefficients> model_coefficients;
    std::vector<pcl::PointIndices> inliers;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> label_indices;
    std::vector<pcl::PointIndices> boundary_indices;
    mps.segmentAndRefine(regions, model_coefficients, inliers, labels, label_indices, boundary_indices);

    if (segments) {
        segments->resize(cloud->width * cloud->height, -1);
        for(std::size_t i = 0; i < inliers.size(); ++i) {
            const pcl::PointIndices &indices = inliers[i];
            for (int index : indices.indices)
                (*segments)[index] = i;
        }
    }

    if (params) {
        params->resize(model_coefficients.size());
        for (std::size_t i = 0; i < model_coefficients.size(); ++i) {
            const pcl::ModelCoefficients &coefficients = model_coefficients[i];
            (*params)[i] = hesseNormalForm(Eigen::Vector4d(coefficients.values.at(0), coefficients.values.at(1),
                                                           coefficients.values.at(2), coefficients.values.at(3)));
        }
    }

    return model_coefficients.size();
}
