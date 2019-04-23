
// STL
#include <iostream>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "util/alignment.h"
#include "util/macros.h"
#include "util/math.hpp"
#include "util/random.hpp"

int main() {

    SetPRNGSeed(0);

    std::vector<Eigen::Vector3d> points = {Eigen::Vector3d(RandomReal<double>(-10, 10), RandomReal<double>(-10, 10), 5.0),
                                           Eigen::Vector3d(RandomReal<double>(-10, 10), RandomReal<double>(-10, 10), 5.0),
                                           Eigen::Vector3d(RandomReal<double>(-10, 10), RandomReal<double>(-10, 10), 5.0)};

    Eigen::Vector4d p(0.0, 0.0, 1.0, -5.0);

    for (const Eigen::Vector3d& point : points)
        RUNTIME_ASSERT(is_negligible(point.dot(p.head<3>()) + p(3)));

    Eigen::Isometry3d T;
    T.linear() = (Eigen::AngleAxisd(RandomReal<double>(-1, 1), Eigen::Vector3d::UnitZ())
                  * Eigen::AngleAxisd(RandomReal<double>(-1, 1), Eigen::Vector3d::UnitY())
                  * Eigen::AngleAxisd(RandomReal<double>(-1, 1), Eigen::Vector3d::UnitX()))
                 .toRotationMatrix();
    T.translation() = Eigen::Vector3d(RandomReal<double>(-100, 100),
                                      RandomReal<double>(-100, 100),
                                      RandomReal<double>(-100, 100));

    Eigen::Vector4d p_ = transform_plane(p, T);
    if (!is_negligible(p_(3)))
        RUNTIME_ASSERT(p_(3) < 0.0);

    for (Eigen::Vector3d& point : points) {
        point = T * point;
        RUNTIME_ASSERT(is_negligible(point.dot(p_.head<3>()) + p_(3)));
    }

    std::cout << "OK!" << std::endl;

    return 0;
}
