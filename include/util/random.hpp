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

#ifndef UTIL_RANDOM_HPP_
#define UTIL_RANDOM_HPP_

#include <algorithm>
#include <chrono>
#include <random>

#include "util/macros.h"

extern thread_local std::mt19937* PRNG;

static const unsigned kRandomPRNGSeed = std::numeric_limits<unsigned>::max();

// Initialize the PRNG with the given seed.
//
// @param seed   The seed for the PRNG. If the seed is -1, the current time
//               is used as the seed.
void SetPRNGSeed(unsigned seed = kRandomPRNGSeed);

// Generate uniformly distributed random integer number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomInteger(const T min, const T max);

// Generate uniformly distributed random real number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomReal(const T min, const T max);

// Generate Gaussian distributed random real number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomGaussian(const T mean, const T stddev);

// Fisher-Yates shuffling.
//
// Note that the vector may not contain more values than UINT32_MAX. This
// restriction comes from the fact that the 32-bit version of the
// Mersenne Twister PRNG is significantly faster.
//
// @param elems            Vector of elements to shuffle.
// @param num_to_shuffle   Optional parameter, specifying the number of first
//                         N elements in the vector to shuffle.
template <typename T>
void Shuffle(const uint32_t num_to_shuffle, std::vector<T>* elems);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

thread_local std::mt19937* PRNG = nullptr;

void SetPRNGSeed(unsigned seed) {
  // Overwrite existing PRNG
  if (PRNG != nullptr) {
    delete PRNG;
  }

  if (seed == kRandomPRNGSeed) {
    seed = static_cast<unsigned>(
        std::chrono::system_clock::now().time_since_epoch().count());
  }

  PRNG = new std::mt19937(seed);
}

template <typename T>
T RandomInteger(const T min, const T max) {
  if (PRNG == nullptr) {
    SetPRNGSeed();
  }

  std::uniform_int_distribution<T> distribution(min, max);

  return distribution(*PRNG);
}

template <typename T>
T RandomReal(const T min, const T max) {
  if (PRNG == nullptr) {
    SetPRNGSeed();
  }

  std::uniform_real_distribution<T> distribution(min, max);

  return distribution(*PRNG);
}

template <typename T>
T RandomGaussian(const T mean, const T stddev) {
  if (PRNG == nullptr) {
    SetPRNGSeed();
  }

  std::normal_distribution<T> distribution(mean, stddev);
  return distribution(*PRNG);
}

template <typename T>
void Shuffle(const uint32_t num_to_shuffle, std::vector<T>* elems) {
  RUNTIME_ASSERT(num_to_shuffle <= elems->size());
  const uint32_t last_idx = static_cast<uint32_t>(elems->size() - 1);
  for (uint32_t i = 0; i < num_to_shuffle; ++i) {
    const auto j = RandomInteger<uint32_t>(i, last_idx);
    std::swap((*elems)[i], (*elems)[j]);
  }
}

#endif  // UTIL_RANDOM_HPP_
