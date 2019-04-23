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

#ifndef RANDOM_SAMPLER_HPP_
#define RANDOM_SAMPLER_HPP_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "sampler.hpp"

#include "util/random.hpp"

#include "util/macros.h"

// Random sampler for RANSAC-based methods.
//
// Note that a separate sampler should be instantiated per thread.
class RandomSampler : public Sampler {
 public:
  explicit RandomSampler(const std::size_t num_samples);

  void Initialize(const std::size_t total_num_samples) override;

  std::size_t MaxNumSamples() override;

  std::vector<std::size_t> Sample() override;

 private:
  const std::size_t num_samples_;
  std::vector<std::size_t> sample_idxs_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

RandomSampler::RandomSampler(const std::size_t num_samples)
    : num_samples_(num_samples) {}

void RandomSampler::Initialize(const std::size_t total_num_samples) {
  RUNTIME_ASSERT(num_samples_ < total_num_samples);
  sample_idxs_.resize(total_num_samples);
  std::iota(sample_idxs_.begin(), sample_idxs_.end(), 0);
}

std::size_t RandomSampler::MaxNumSamples() {
  return std::numeric_limits<std::size_t>::max();
}

std::vector<std::size_t> RandomSampler::Sample() {
  Shuffle(static_cast<uint32_t>(num_samples_), &sample_idxs_);

  std::vector<std::size_t> sampled_idxs(num_samples_);
  for (std::size_t i = 0; i < num_samples_; ++i) {
    sampled_idxs[i] = sample_idxs_[i];
  }

  return sampled_idxs;
}

#endif  // RANDOM_SAMPLER_HPP_
