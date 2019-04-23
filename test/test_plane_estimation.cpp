
// STL
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "util/alignment.h"
#include "util/macros.h"
#include "util/math.hpp"
#include "util/random.hpp"

#include "estimators.hpp"

int main() {

    SetPRNGSeed(0);

    std::vector<Eigen::Vector3d> points = {Eigen::Vector3d(RandomReal<double>(-10, 10), RandomReal<double>(-10, 10), 5.0),
                                           Eigen::Vector3d(RandomReal<double>(-10, 10), RandomReal<double>(-10, 10), 5.0),
                                           Eigen::Vector3d(RandomReal<double>(-10, 10), RandomReal<double>(-10, 10), 5.0)};

    Eigen::Vector4d p(0.0, 0.0, 1.0, -5.0);

    for (const Eigen::Vector3d& point : points)
        RUNTIME_ASSERT(is_negligible(point.dot(p.head<3>()) + p(3)));

    std::vector<Eigen::Vector4d> planes = PlaneEstimator::Estimate(points);
    if (planes.empty()) throw std::runtime_error("No plane found!");
    Eigen::Vector4d plane = planes.front();

    RUNTIME_ASSERT(is_negligible(
        std::acos( p.head<3>().dot(plane.head<3>()) / (p.head<3>().norm()*plane.head<3>().norm()) )
    ));
    RUNTIME_ASSERT(std::abs(plane(3) - p(3)) < 1e-4);

    std::cout << "OK!" << std::endl;

    return 0;
}
