
#ifndef ASSOCIATE_HPP_
#define ASSOCIATE_HPP_

// STL
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>
#include <unordered_map>

#include "types.hpp"

timestamp_t abs_diff(const timestamp_t a, const timestamp_t b) {
    if (a < b) return (b - a);
    else return (a - b);
}

template<typename A, typename B>
std::unordered_map<std::size_t, std::size_t> associate(const std::vector<A>& u, const std::vector<B>& v, const timestamp_t max_diff) {

    std::unordered_map<std::size_t, std::size_t> map;
    for (typename std::vector<A>::const_iterator u_cit = u.begin(); u_cit != u.end(); ++u_cit) {
        typename std::vector<B>::const_iterator v_cit_next = std::lower_bound(v.begin(), v.end(), u_cit->timestamp);
        if (v_cit_next == v.end()) continue;

        typename std::vector<B>::const_iterator v_cit;
        if (v_cit_next != v.begin()) {
            typename std::vector<B>::const_iterator v_cit_prev = std::prev(v_cit_next);

            v_cit = (abs_diff(u_cit->timestamp, v_cit_next->timestamp)
                      <=
                     abs_diff(u_cit->timestamp, v_cit_prev->timestamp))
                      ? v_cit_next : v_cit_prev;
        } else
            v_cit = v_cit_next;

        if (abs_diff(u_cit->timestamp, v_cit->timestamp) > max_diff)
            continue;

        std::size_t i = std::distance(u.begin(), u_cit),
                    j = std::distance(v.begin(), v_cit);

        map[i] = j;
    }

    return map;
}

#endif // ASSOCIATE_HPP_
