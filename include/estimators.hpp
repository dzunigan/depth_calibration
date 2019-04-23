
// STL
#include <cmath>
#include <cstddef>
#include <vector>

// Boost
#include <boost/math/special_functions/sign.hpp>

// Eigen
#include <Eigen/Core>

#include "util/macros.h"
#include "util/math.hpp"

class LineEstimator {
public:
    typedef Eigen::Vector2d X_t;
    typedef Eigen::Vector3d M_t;

    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 2;

    // Estimate the ground plane from a set of
    // plane observations.
    //
    // @param x   Observed plane parameters, in Hesse normal.
    //
    // @return    Estimated ground plane parameters
    static std::vector<M_t> Estimate(const std::vector<X_t>& x);

    // Calculate the residuals of a given set of plane observations
    // given a reference model.
    //
    // Residuals are defined as the squared sine of the angle between normals,
    // and the squared difference between the perpendicular distance to origin.
    //
    // @param x          Set of planar observations.
    // @param m          Model representing the plane parameters in Hesse normal form.
    // @param residuals1 Output vector of residuals.
    static void Residuals(const std::vector<X_t>& x, const M_t& m,
                          std::vector<double>* residuals);
};

class PlaneEstimator {
public:
    typedef Eigen::Vector3d X_t;
    typedef Eigen::Vector4d M_t;

    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 3;

    // Estimate a plane from a set of 3D points
    //
    // @param x   Observed plane points.
    //
    // @return    Estimated plane parameters (in Hessian normal form)
    static std::vector<M_t> Estimate(const std::vector<X_t>& x);

    // Calculate the residuals of a set of 3D points,
    // given a plane model.
    //
    // Residuals are defined as the squared perpendicular distance to the plane.
    //
    // @param x          Set of 3D points.
    // @param m          Model representing the plane, in Hessian normal form.
    // @param residuals1 Output vector of residuals.
    static void Residuals(const std::vector<X_t>& x, const M_t& m,
                          std::vector<double>* residuals);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

std::vector<LineEstimator::M_t> LineEstimator::Estimate(const std::vector<X_t>& x) {
    RUNTIME_ASSERT(x.size() >= kMinNumSamples);

    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();

    const std::size_t n = x.size();
    for (std::size_t k = 0; k < n; ++k) {
        const X_t& x_k = x[k];

        Eigen::Matrix<double, 1, 3> Q;
        Q << x_k(0), x_k(1), 1.0;

        M += Q.transpose()*Q;
    }

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    W(0, 0) = 1.0; W(1, 1) = 1.0;

    const double m00 = M(0, 0),
            m01 = 0.5*(M(0, 1) + M(1, 0)),
            m02 = 0.5*(M(0, 2) + M(2, 0)),
            m11 = M(1, 1),
            m12 = 0.5*(M(1, 2) + M(2, 1)),
            m22 = M(2, 2);

    using std::pow;
    const double a = m22,
            b = - pow(m02,2) - pow(m12,2) + m00*m22 + m11*m22,
            c = - m22*pow(m01,2) + 2*m01*m02*m12 - m11*pow(m02,2) - m00*pow(m12,2) + m00*m11*m22;

    // solve: a*s^2 + b*s + c = 0
    double x0, x1;
    const int real_solutions = solveQuadratic(a, b, c, x0, x1);

    std::vector<M_t> models;

    if (real_solutions == 0)
        return models;

    std::vector<double> lambdas;
    lambdas.push_back(x0);
    if (real_solutions > 1)
        lambdas.push_back(x1);

    M_t solution; // Unique solution
    bool solution_found = false;
    double solution_cost = std::numeric_limits<double>::max();

    for (const double lambda : lambdas) {
        Eigen::MatrixXd phi;
        int kernel_size = solveNullspace(M + lambda*W, phi);

        if (kernel_size != 1) continue;

        phi /=  phi.block<2, 1>(0, 0).norm(); // Normalize solution
        if (!is_negligible(phi(2, 0))) phi *= -boost::math::sign(phi(2, 0));

        M_t candidate_solution;
        candidate_solution << phi(0, 0), phi(1, 0), phi(2, 0);

        std::vector<double> residuals;
        Residuals(x, candidate_solution, &residuals);

        const double cost = std::accumulate(residuals.begin(), residuals.end(), 0.0);
        if (cost < solution_cost) {
            solution = candidate_solution;
            solution_found = true;
            solution_cost = cost;
        }
    }

    if (solution_found)
        models.push_back(solution);
    
    return models;
}

void LineEstimator::Residuals(const std::vector<X_t>& x, const M_t& m, std::vector<double>* residuals) {
    const std::size_t n = x.size();
    residuals->resize(n);

    for (std::size_t k = 0; k < n; ++k) {
        const X_t &x_k = x[k];
        const double e = std::abs(m.dot(Eigen::Vector3d(x_k(0), x_k(1), 1.0))) / m.head<2>().norm();

        (*residuals)[k] = e*e;
    }
}

std::vector<PlaneEstimator::M_t> PlaneEstimator::Estimate(const std::vector<X_t>& x) {
    RUNTIME_ASSERT(x.size() >= kMinNumSamples);

    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();

    const std::size_t n = x.size();
    for (std::size_t k = 0; k < n; ++k) {
        const X_t &x_k = x[k];

        Eigen::Matrix<double, 1, 4> Q;
        Q << x_k(0), x_k(1), x_k(2), 1.0;

        M += Q.transpose()*Q;
    }

    Eigen::Matrix4d W = Eigen::Matrix4d::Zero();
    W(0, 0) = 1; W(1, 1) = 1; W(2, 2) = 1;

    const double m00 = M(0, 0),
            m01 = 0.5*(M(0, 1) + M(1, 0)),
            m02 = 0.5*(M(0, 2) + M(2, 0)),
            m03 = 0.5*(M(0, 3) + M(3, 0)),
            m11 = M(1, 1),
            m12 = 0.5*(M(1, 2) + M(2, 1)),
            m13 = 0.5*(M(1, 3) + M(3, 1)),
            m22 = M(2, 2),
            m23 = 0.5*(M(2, 3) + M(3, 2)),
            m33 = M(3, 3);

    // m33*s^3 + (- m03^2 - m13^2 - m23^2 + m00*m33 + m11*m33 + m22*m33)*s^2 + (2*m01*m03*m13 - m03^2*m11 - m00*m23^2 - m03^2*m22 - m11*m23^2 - m01^2*m33 - m02^2*m33 - m13^2*m22 - m12^2*m33 - m00*m13^2 + 2*m02*m03*m23 + m00*m11*m33 + 2*m12*m13*m23 + m00*m22*m33 + m11*m22*m33)*s + m01^2*m23^2 - m22*m33*m01^2 + 2*m33*m01*m02*m12 - 2*m01*m02*m13*m23 - 2*m01*m03*m12*m23 + 2*m22*m01*m03*m13 + m02^2*m13^2 - m11*m33*m02^2 - 2*m02*m03*m12*m13 + 2*m11*m02*m03*m23 + m03^2*m12^2 - m11*m22*m03^2 - m00*m33*m12^2 + 2*m00*m12*m13*m23 - m00*m22*m13^2 - m00*m11*m23^2 + m00*m11*m22*m33

    using std::pow;
    const double a = m33,
            b = -pow(m03, 2) - pow(m13, 2) - pow(m23, 2) + m00*m33 + m11*m33 + m22*m33,
            c = 2.0*m01*m03*m13 - pow(m03, 2)*m11 - m00*pow(m23, 2) - pow(m03, 2)*m22 - m11*pow(m23, 2) - pow(m01, 2)*m33 - pow(m02, 2)*m33 - pow(m13, 2)*m22 - pow(m12, 2)*m33 - m00*pow(m13, 2) + 2.0*m02*m03*m23 + m00*m11*m33 + 2.0*m12*m13*m23 + m00*m22*m33 + m11*m22*m33,
            d = pow(m01, 2)*pow(m23, 2) - m22*m33*pow(m01, 2) + 2.0*m33*m01*m02*m12 - 2.0*m01*m02*m13*m23 - 2.0*m01*m03*m12*m23 + 2.0*m22*m01*m03*m13 + pow(m02, 2)*pow(m13, 2) - m11*m33*pow(m02, 2) - 2.0*m02*m03*m12*m13 + 2.0*m11*m02*m03*m23 + pow(m03, 2)*pow(m12, 2) - m11*m22*pow(m03, 2) - m00*m33*pow(m12, 2) + 2.0*m00*m12*m13*m23 - m00*m22*pow(m13, 2) - m00*m11*pow(m23, 2) + m00*m11*m22*m33;

    // solve: a*s^3 + b*s^2 + c*s + d = 0
    double roots[3];
    const int num_solutions = SolveCubicReals(a, b, c, d, roots);

    std::vector<M_t> models;

    M_t solution; // Unique solution
    bool solution_found = false;
    double solution_cost = std::numeric_limits<double>::max();

    for (int i = 0; i < num_solutions; ++i) {
        const double lambda = roots[i];

        Eigen::MatrixXd phi;
        int kernel_size = solveNullspace(M + lambda*W, phi);

        if (kernel_size != 1) return models; // TODO Bug returning non-minimal solution (the true solution may be discarded for being non-singular) [numerical instability?]
        phi *= (is_negligible(phi(3, 0)) ? 1.0 : -boost::math::sign(phi(3, 0))) / phi.block<3, 1>(0, 0).norm();

        M_t candidate_solution;
        candidate_solution = phi;

        std::vector<double> residuals;
        Residuals(x, candidate_solution, &residuals);

        double cost = std::accumulate(residuals.begin(), residuals.end(), 0.0);
        if (cost < solution_cost) {
            solution = hesseNormalForm(candidate_solution);
            solution_found = true;
            solution_cost = cost;
        }
    }

    if (solution_found)
        models.push_back(solution);

    return models;
}

void PlaneEstimator::Residuals(const std::vector<X_t>& x, const M_t& m, std::vector<double>* residuals) {
    const std::size_t n = x.size();
    residuals->resize(n);

    for (std::size_t k = 0; k < n; ++k) {
        const X_t &x_k = x[k];
        const double e = m.dot(Eigen::Vector4d(x_k(0), x_k(1), x_k(2), 1.0)) / m.head<3>().norm();

        (*residuals)[k] = e*e;
    }
}
