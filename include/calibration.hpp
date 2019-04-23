
// STL
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "util/endian.hpp"
#include "util/macros.h"
#include "util/statistical.hpp"

#include <iostream> // Debug

using measurement_t = double;

struct calibration_observation_t {
    measurement_t measurement;
    measurement_t reference;

    calibration_observation_t()
        : measurement(std::numeric_limits<measurement_t>::quiet_NaN()),
          reference(std::numeric_limits<measurement_t>::quiet_NaN())
    { }

    calibration_observation_t(const measurement_t measurement, const measurement_t reference)
        : measurement(measurement), reference(reference)
    { }
};

using calibration_observation_ptr = std::shared_ptr<calibration_observation_t>;

inline bool operator<(const calibration_observation_t& lhs, const calibration_observation_t& rhs) {
    return (lhs.measurement < rhs.measurement);
}

inline bool operator<(const calibration_observation_ptr& lhs, const calibration_observation_ptr& rhs) {
    return (lhs->measurement < rhs->measurement);
}

template<typename T>
struct matrix {
    std::size_t rows, cols;
    std::vector<std::vector<T>> data;

    matrix(std::size_t rows, std::size_t cols)
        : rows(rows), cols(cols)
    {
        data.resize(rows);
        for (auto& v : data)
            v.resize(cols);
    }

    T& at(std::size_t i, std::size_t j) {
        return data.at(i).at(j);
    }
};

struct calibration_t {
    std::uint8_t meaningful;

    double a, b, c;

    double z_min, z_max;
    std::uint64_t z_cnt;

    calibration_t()
        : meaningful(0), a(0.0), b(0.0), c(0.0),
          z_min(0.0), z_max(0.0), z_cnt(0)
    { }

    calibration_t(const std::uint8_t meaningful, const double a, const double b, const double c,
                  const double z_min, const double z_max, const std::uint64_t  z_cnt)
        : meaningful(meaningful), a(a), b(b), c(c),
          z_min(z_min), z_max(z_max), z_cnt(z_cnt)
    { }

    double correct(double z) const {
        RUNTIME_ASSERT(meaningful);
        return z - (a*z*z + b*z + c);
    }

    double evaluate(double z) const {
        return (a*z*z + b*z + c);
    }
};

struct NoiseStdvFunc {
    Eigen::Vector3d coeff;

    NoiseStdvFunc(const Eigen::Vector3d& coeff)
        : coeff(coeff)
    { }

    double at (const double x) const {
        return x*x*coeff(0) + x* coeff(1) + coeff(2);
    }
};

NoiseStdvFunc noise_calibration(matrix<std::vector<calibration_observation_ptr>>& observations, measurement_t discretization = 0.1, measurement_t min = 1.0, measurement_t max = 4.0) {
    RUNTIME_ASSERT(min < max);

    std::size_t N = static_cast<std::size_t>(std::ceil((max - min) / discretization));
    std::vector<std::vector<measurement_t>> biases(N);
    for (std::size_t i = 0; i < observations.rows; ++i) {
        for (std::size_t j = 0; j < observations.cols; ++j) {
            // Discretize measurements
            std::vector<std::vector<measurement_t>> px_biases(N);
            for (const calibration_observation_ptr obs : observations.at(i, j)) {
                if (obs->measurement < min || obs->measurement > max) continue;
                std::size_t idx = static_cast<std::size_t>((obs->measurement - min) / discretization);
                px_biases.at(idx).push_back(obs->measurement - obs->reference);
            }

            // Subtract bias mean
            for (std::vector<measurement_t>& bin : px_biases) {
                if (bin.size() < 3) {
                    bin.clear();
                    continue;
                }

                measurement_t mean = vector_mean(bin);
                for (measurement_t& elem : bin)
                    elem -= mean;
            }

            // Add normalized biases
            for (std::size_t k = 0; k < N; ++k) {
                biases[k].insert(biases[k].end(), px_biases[k].cbegin(), px_biases[k].cend());
            }
        }
    }

    Eigen::MatrixXd A(N, 3);
    Eigen::VectorXd b(N);

    std::size_t k = 0;
    for (std::size_t idx = 0; idx < N; ++idx) {
        if (biases.at(idx).size() < 12) continue;

        const double z = (idx + 0.5) * discretization + min;
        const double stdv = vector_stdv(biases.at(idx), 0.0);

        //std::cout << z << "," << stdv << std::endl;

        A.row(k) = Eigen::RowVector3d(z*z, z, 1.0);
        b(k) = stdv;
        k++;
    }

    RUNTIME_ASSERT(k > 3);
    A.conservativeResize(k, Eigen::NoChange);
    b.conservativeResize(k);

    return NoiseStdvFunc(A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b));
}

NoiseStdvFunc single_noise_calibration(std::vector<calibration_observation_ptr>& observations, measurement_t discretization = 0.1, measurement_t min = 1.0, measurement_t max = 4.0) {
    RUNTIME_ASSERT(min < max);

    std::size_t N = static_cast<std::size_t>(std::ceil((max - min) / discretization));
    std::vector<std::vector<measurement_t>> biases(N);

    for (const calibration_observation_ptr obs : observations) {
        if (obs->measurement < min || obs->measurement > max) continue;
        std::size_t idx = static_cast<std::size_t>((obs->measurement - min) / discretization);
        biases.at(idx).push_back(obs->measurement - obs->reference);
    }

    // Subtract bias mean
    for (std::vector<measurement_t>& bin : biases) {
        if (bin.size() < 3) continue;

        measurement_t mean = vector_mean(bin);
        for (measurement_t& elem : bin)
            elem -= mean;
    }

    Eigen::MatrixXd A(N, 3);
    Eigen::VectorXd b(N);

    std::size_t k = 0;
    for (std::size_t idx = 0; idx < N; ++idx) {
        if (biases.at(idx).size() < 5) continue;

        const double z = (idx + 0.5) * discretization + min;
        const double stdv = vector_stdv(biases.at(idx), 0.0);

        //std::cout << z << "," << stdv << std::endl;

        A.row(k) = Eigen::RowVector3d(z*z, z, 1.0);
        b(k) = stdv;
        k++;
    }

    RUNTIME_ASSERT(k > 3);
    A.conservativeResize(k, Eigen::NoChange);
    b.conservativeResize(k);

    return NoiseStdvFunc(A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b));
}

Eigen::Vector3d depth_calibration(std::vector<calibration_observation_ptr>& observations, const NoiseStdvFunc& stdv) {
    RUNTIME_ASSERT(!observations.empty());

    std::size_t N = observations.size();

    Eigen::MatrixXd A(N, 3);
    Eigen::VectorXd b(N);

    for (std::size_t idx = 0; idx < N; ++idx) {
        const double z = observations[idx]->measurement;
        const double z_ = observations[idx]->reference;

        const double w = 1.0 / stdv.at(z);

        //std::cout << z << "," << z_ << "," << w << std::endl;

        A.row(idx) = Eigen::RowVector3d(z*z*w, z*w, w);
        b(idx) = (z - z_)*w;
    }

    return A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}

void write_calib(const std::string& file, std::shared_ptr<matrix<calibration_t>> calibration) {
    // Write calibration
    RUNTIME_ASSERT(calibration);
    std::ios::openmode mode = std::ios::out | std::ios::binary;

    std::ofstream output_stream(file, mode);
    RUNTIME_ASSERT(output_stream.is_open());

    WriteBinaryLittleEndian<std::size_t>(&output_stream, calibration->rows);
    WriteBinaryLittleEndian<std::size_t>(&output_stream, calibration->cols);
    for (std::size_t i = 0; i < calibration->rows; ++i) {
        for (std::size_t j = 0; j < calibration->cols; ++j) {
            WriteBinaryLittleEndian<std::uint8_t>(&output_stream, calibration->at(i, j).meaningful);
            WriteBinaryLittleEndian<double>(&output_stream, calibration->at(i, j).a);
            WriteBinaryLittleEndian<double>(&output_stream, calibration->at(i, j).b);
            WriteBinaryLittleEndian<double>(&output_stream, calibration->at(i, j).c);
            WriteBinaryLittleEndian<double>(&output_stream, calibration->at(i, j).z_min);
            WriteBinaryLittleEndian<double>(&output_stream, calibration->at(i, j).z_max);
            WriteBinaryLittleEndian<std::uint64_t>(&output_stream, calibration->at(i, j).z_cnt);
        }
    }
}

std::shared_ptr<matrix<calibration_t>> read_calib(const std::string& file) {
    // Read calibration
    std::shared_ptr<matrix<calibration_t>> calibration;

    std::ios::openmode mode = std::ios::in | std::ios::binary;

    std::ifstream input_stream(file, mode);
    RUNTIME_ASSERT(input_stream.is_open());

    std::size_t rows = ReadBinaryLittleEndian<std::size_t>(&input_stream);
    std::size_t cols = ReadBinaryLittleEndian<std::size_t>(&input_stream);

    calibration = std::make_shared<matrix<calibration_t>>(rows, cols);

    for (std::size_t i = 0; i < calibration->rows; ++i) {
        for (std::size_t j = 0; j < calibration->cols; ++j) {
            calibration->at(i, j).meaningful = ReadBinaryLittleEndian<std::uint8_t>(&input_stream);
            calibration->at(i, j).a = ReadBinaryLittleEndian<double>(&input_stream);
            calibration->at(i, j).b = ReadBinaryLittleEndian<double>(&input_stream);
            calibration->at(i, j).c = ReadBinaryLittleEndian<double>(&input_stream);
            calibration->at(i, j).z_min = ReadBinaryLittleEndian<double>(&input_stream);
            calibration->at(i, j).z_max = ReadBinaryLittleEndian<double>(&input_stream);
            calibration->at(i, j).z_cnt = ReadBinaryLittleEndian<std::uint64_t>(&input_stream);
        }
    }

    return calibration;
}
