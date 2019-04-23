// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RANSAC_HPP_
#define RANSAC_HPP_

//#include <cfloat>
//#include <random>
//#include <stdexcept>
//#include <vector>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// Eigen
#include <Eigen/Core>

#include "random_sampler.hpp"
#include "support_measurement.hpp"

#include "util/macros.h"

struct RANSACOptions {
  // Maximum error for a sample to be considered as an inlier.
  double max_error = 0.0;

  // A priori assumed minimum inlier ratio, which determines the maximum number
  // of iterations. Only applies if smaller than `max_num_trials`.
  double min_inlier_ratio = 0.1;

  // Abort the iteration if minimum probability that one sample is free from
  // outliers is reached.
  double confidence = 0.99;

  // Number of random trials to estimate model from random subset.
  std::size_t min_num_trials = 0;
  std::size_t max_num_trials = std::numeric_limits<std::size_t>::max();

  void Check() const {
    RUNTIME_ASSERT(max_error > 0.0);
    RUNTIME_ASSERT(min_inlier_ratio >= 0.0 && min_inlier_ratio <= 1.0);
    RUNTIME_ASSERT(confidence >= 0.0 && confidence <= 1.0);
    RUNTIME_ASSERT(min_num_trials <= max_num_trials);
  }
};

template <typename Estimator, typename SupportMeasurer = InlierSupportMeasurer,
          typename Sampler = RandomSampler>
class RANSAC {
 public:
  struct Report {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Whether the estimation was successful.
    bool success = false;

    // The number of RANSAC trials / iterations.
    std::size_t num_trials = 0;

    // The support of the estimated model.
    typename SupportMeasurer::Support support;

    // Boolean mask which is true if a sample is an inlier.
    std::vector<char> inlier_mask;

    // The estimated model.
    typename Estimator::M_t model;
  };

  explicit RANSAC(const RANSACOptions& options);

  // Determine the maximum number of trials required to sample at least one
  // outlier-free random set of samples with the specified confidence,
  // given the inlier ratio.
  //
  // @param num_inliers    The number of inliers.
  // @param num_samples    The total number of samples.
  // @param confidence     Confidence that one sample is outlier-free.
  //
  // @return               The required number of iterations.
  static std::size_t ComputeNumTrials(const std::size_t num_inliers,
                                 const std::size_t num_samples,
                                 const double confidence);

  // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
  //
  // @param X              Independent variables.
  // @param Y              Dependent variables.
  //
  // @return               The report with the results of the estimation.
  Report Estimate(const std::vector<typename Estimator::X_t>& X);

  // Objects used in RANSAC procedure. Access useful to define custom behavior
  // through options or e.g. to compute residuals.
  Estimator estimator;
  Sampler sampler;
  SupportMeasurer support_measurer;

 protected:
  RANSACOptions options_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Estimator, typename SupportMeasurer, typename Sampler>
RANSAC<Estimator, SupportMeasurer, Sampler>::RANSAC(
    const RANSACOptions& options)
    : sampler(Sampler(Estimator::kMinNumSamples)), options_(options) {
  options.Check();

  // Determine max_num_trials based on assumed `min_inlier_ratio`.
  const std::size_t kNumSamples = 100000;
  const std::size_t dyn_max_num_trials = ComputeNumTrials(
      static_cast<std::size_t>(options_.min_inlier_ratio * kNumSamples), kNumSamples,
      options_.confidence);
  options_.max_num_trials =
      std::min<std::size_t>(options_.max_num_trials, dyn_max_num_trials);
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
std::size_t RANSAC<Estimator, SupportMeasurer, Sampler>::ComputeNumTrials(
    const std::size_t num_inliers, const std::size_t num_samples,
    const double confidence) {
  const double inlier_ratio = num_inliers / static_cast<double>(num_samples);

  const double nom = 1 - confidence;
  if (nom <= 0) {
    return std::numeric_limits<std::size_t>::max();
  }

  const double denom = 1 - std::pow(inlier_ratio, Estimator::kMinNumSamples);
  if (denom <= 0) {
    return 1;
  }

  return static_cast<std::size_t>(std::ceil(std::log(nom) / std::log(denom)));
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
typename RANSAC<Estimator, SupportMeasurer, Sampler>::Report
RANSAC<Estimator, SupportMeasurer, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X) {
  const std::size_t num_samples = X.size();

  Report report;
  report.success = false;
  report.num_trials = 0;

  if (num_samples < Estimator::kMinNumSamples) {
    return report;
  }

  typename SupportMeasurer::Support best_support;
  typename Estimator::M_t best_model;

  bool abort = false;

  const double max_residual = options_.max_error * options_.max_error;

  std::vector<double> residuals(num_samples);

  std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);

  sampler.Initialize(num_samples);

  std::size_t max_num_trials = options_.max_num_trials;
  max_num_trials = std::min<std::size_t>(max_num_trials, sampler.MaxNumSamples());
  std::size_t dyn_max_num_trials = max_num_trials;

  for (report.num_trials = 0; report.num_trials < max_num_trials;
       ++report.num_trials) {
    if (abort) {
      report.num_trials += 1;
      break;
    }

    sampler.SampleX(X, &X_rand);

    // Estimate model for current subset.
    const std::vector<typename Estimator::M_t> sample_models =
        estimator.Estimate(X_rand);

    // Iterate through all estimated models.
    for (const auto& sample_model : sample_models) {
      estimator.Residuals(X, sample_model, &residuals);
      RUNTIME_ASSERT(residuals.size() == X.size());

      const auto support = support_measurer.Evaluate(residuals, max_residual);

      // Save as best subset if better than all previous subsets.
      if (support_measurer.Compare(support, best_support)) {
        best_support = support;
        best_model = sample_model;

        dyn_max_num_trials = ComputeNumTrials(best_support.num_inliers,
                                              num_samples, options_.confidence);
      }

      if (report.num_trials >= dyn_max_num_trials &&
          report.num_trials >= options_.min_num_trials) {
        abort = true;
        break;
      }
    }
  }

  report.support = best_support;
  report.model = best_model;

  // No valid model was found.
  if (report.support.num_inliers <= estimator.kMinNumSamples) {
    return report;
  }

  report.success = true;

  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.

  estimator.Residuals(X, report.model, &residuals);
  RUNTIME_ASSERT(residuals.size() == X.size());

  report.inlier_mask.resize(num_samples);
  for (std::size_t i = 0; i < residuals.size(); ++i) {
    if (residuals[i] <= max_residual) {
      report.inlier_mask[i] = true;
    } else {
      report.inlier_mask[i] = false;
    }
  }

  return report;
}

#endif  // RANSAC_HPP_
