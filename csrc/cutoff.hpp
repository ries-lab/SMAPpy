// The dynamic cutoff of every frame of a block: median + factor * the slope of
// the 20-80 % quantile range of that frame's local maxima -- SMAP's
// getdynamiccutoff, and `smappy.detect.DynamicCutoff.__call__`, bit for bit.
//
// A quantile is the ceil(n p)-th smallest value, as SMAP's myquantilefast
// takes it: no interpolation, and three order statistics are not a sort, so
// nth_element finds them in linear time, each search starting where the last
// one left off.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace smappy {

// Cutoffs for the segments values[starts[i], starts[i+1]), i in [first, last).
// A segment with fewer than `min_count` maxima gets NaN: below ten the rule is
// a mean (`DynamicCutoff.__call__`), which the caller keeps.
inline void segment_cutoffs(const float* values, const int64_t* starts,
                            long long first, long long last, double factor,
                            long long min_count, float* out) {
    const double levels[3] = {0.2, 0.5, 0.8};
    std::vector<float> v;
    for (long long s = first; s < last; ++s) {
        const long long n = starts[s + 1] - starts[s];
        if (n < min_count || n <= 0) {
            out[s] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        v.assign(values + starts[s], values + starts[s + 1]);
        double q[3];
        long long from = 0;
        for (int k = 0; k < 3; ++k) {
            const long long rank = static_cast<long long>(std::ceil(n * levels[k])) - 1;
            std::nth_element(v.begin() + from, v.begin() + rank, v.end());
            q[k] = v[rank];
            from = rank;
        }
        // 0.8 - 0.2 as MATLAB divides by it, 0.6000000000000001 and not 0.6
        const double slope = (q[2] - q[0]) / (0.8 - 0.2);
        // apart from the sum: fused into an FMA it would round once, not twice
        volatile double rise = slope * factor;
        out[s] = static_cast<float>(q[1] + rise);
    }
}

}  // namespace smappy
