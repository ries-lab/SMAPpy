// Candidates of one filtered frame in one pass: its local maxima, the cutoff
// of each half from them, and the maxima above it -- `smappy.detect.
// find_candidates`, which called the maxima search, handed every maximum to
// Python (about 2 500 a frame of 256 x 256 to keep 30), and grouped them by
// frame and half with numpy to set and apply the cutoffs.
//
// Each frame's maxima go to a scratch buffer of the thread's, which stays in
// cache, and only the candidates leave it.  The rule for fewer than
// `min_count` maxima (a mean, `DynamicCutoff.__call__`) is numpy's float32
// mean, which this does not reproduce: such a half's maxima are handed back
// as they are, for Python to judge.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cutoff.hpp"
#include "maxima.hpp"

namespace smappy {

struct DetectParams {
    bool dynamic;          // the dynamic cutoff, else `threshold` alone
    float threshold;       // the absolute cutoff
    double factor;         // the dynamic cutoff's factor
    int split_axis;        // -1 without a split; 0 rows, 1 columns
    int split_position;    // the first row / column of the second half
    long long min_count;   // fewer maxima than this: the half is deferred
};

struct FrameScratch {
    std::vector<Maximum> maxima;
    std::vector<float> values;
};

inline int half_of(const Maximum& m, const DetectParams& p) {
    if (p.split_axis < 0) return 0;
    return (p.split_axis == 0 ? m.y : m.x) >= p.split_position ? 1 : 0;
}

// Appends frame `frame`'s candidates to `kept`, the maxima of its deferred
// halves to `deferred`, and writes each half's cutoff to cutoffs[0..halves)
// (NaN where no maxima or deferred; the threshold for an absolute cutoff).
inline void detect_frame(const float* img, int ny, int nx, int frame,
                         const DetectParams& p, FrameScratch& s,
                         std::vector<Maximum>& kept, std::vector<Maximum>& deferred,
                         float* cutoffs) {
    const int halves = p.split_axis < 0 ? 1 : 2;
    if (!p.dynamic) {
        // the threshold applied during the search, which is the cheap case
        find_maxima_frame(img, ny, nx, frame, p.threshold, kept);
        for (int h = 0; h < halves; ++h) cutoffs[h] = p.threshold;
        return;
    }
    s.maxima.clear();
    find_maxima_frame(img, ny, nx, frame, -std::numeric_limits<float>::infinity(), s.maxima);
    long long count[2] = {0, 0};
    for (int h = 0; h < halves; ++h) {
        s.values.clear();
        for (const Maximum& m : s.maxima)
            if (half_of(m, p) == h) s.values.push_back(m.value);
        count[h] = static_cast<long long>(s.values.size());
        cutoffs[h] = count[h] < p.min_count || count[h] == 0
                         ? std::numeric_limits<float>::quiet_NaN()
                         : dynamic_cutoff(s.values.data(), count[h], p.factor);
    }
    // in the search's order, row-major: the order the numpy selection kept
    for (const Maximum& m : s.maxima) {
        const int h = half_of(m, p);
        if (count[h] < p.min_count)
            deferred.push_back(m);
        else if (m.value > cutoffs[h])
            kept.push_back(m);
    }
}

}  // namespace smappy
