
#ifndef UTIL_STATISTICAL_HPP_
#define UTIL_STATISTICAL_HPP_

// STL
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <vector>

#include "util/macros.h"

template<typename T>
T vector_median(std::vector<T>& v) {
    const std::size_t N = v.size();
    RUNTIME_ASSERT(N > 1);

    if (N == 1) return v[0];

    const std::size_t m = v.size()/2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    if (N & 1)
        return v[m];
    else
        return (v[m-1] + v[m]) / static_cast<T>(2.0);
}

template<typename T>
T vector_mad(std::vector<T>& v) {
    const std::size_t N = v.size();
    RUNTIME_ASSERT(N > 1);

    const T m = vector_median(v);

    std::vector<T> abs_dev(N);
    for (std::size_t i = 0; i < N; ++i) {
        abs_dev[i] = std::abs(v[i] - m);
    }

    return vector_median(abs_dev);
}

template<typename T>
T vector_mean(const std::vector<T>& v) {
    RUNTIME_ASSERT(v.size() > 1);

    T mean = 0.0;
    for (const T& elem : v)
        mean += elem;
    return mean / static_cast<T>(v.size());
}

template<typename T>
T vector_stdv(const std::vector<T>& v) {
    RUNTIME_ASSERT(v.size() > 1);

    T s = 0.0;
    const T mean = vector_mean(v);
    for (const T& elem : v)
        s += (elem - mean)*(elem - mean);
    return std::sqrt(s / static_cast<T>(v.size()));
}

template<typename T>
T vector_stdv(const std::vector<T>& v, const T mean) {
    RUNTIME_ASSERT(v.size() > 1);

    T s = 0.0;
    for (const T& elem : v)
        s += (elem - mean)*(elem - mean);
    return std::sqrt(s / static_cast<T>(v.size()));
}

#endif // UTIL_STATISTICAL_HPP_
