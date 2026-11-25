// Strict local-maximum search, following SMAP's maximumfind2.c.
//
// A pixel is a maximum when it is strictly greater than all eight neighbours,
// so plateaus yield nothing.  SMAP's single pass with short-circuit evaluation
// is kept for the ends of rows and for compilers without vector types; the
// rows themselves are searched eight pixels at a time without branches.
#pragma once

#include <cstdint>
#include <vector>

namespace smappy {

struct Maximum {
    int32_t frame, y, x;
    float value;
};

// The scalar search over pixels [x0, x1) of row y: the reference, and the
// tail of a row the vector search leaves over.
inline void find_maxima_span(const float* img, int nx, int y, int x0, int x1,
                             int frame, float threshold, std::vector<Maximum>& out) {
    const float* row = img + y * nx;
    const float* above = row - nx;
    const float* below = row + nx;
    for (int x = x0; x < x1; ++x) {
        const float v = row[x];
        if (v <= threshold) continue;
        if (v > row[x - 1] && v > row[x + 1] &&
            v > above[x - 1] && v > above[x] && v > above[x + 1] &&
            v > below[x - 1] && v > below[x] && v > below[x + 1]) {
            out.push_back({frame, y, x, v});
            ++x;  // the next pixel is smaller, so it cannot be a maximum
        }
    }
}

// Append all strict maxima of one frame (row-major, ny x nx) to `out`, in
// row-major order.  The one-pixel border is skipped, as in the original.
//
// Without a threshold (the dynamic cutoff needs every maximum) the scalar
// search's early exits are coin flips on noise, and mispredicted branches made
// it five times slower than with one (docs/performance.md).  So eight pixels
// at a time are compared with all eight neighbours and the threshold, without
// a branch: nine unaligned loads and nine compares ANDed into one mask, and a
// branch only on whether any of the eight is a maximum.  Compares, not a max
// of the neighbours: `v > n` for each keeps the scalar search's answer exactly,
// NaN included (a NaN is never a maximum, nor next to one).
inline void find_maxima_frame(const float* img, int ny, int nx, int frame,
                              float threshold, std::vector<Maximum>& out) {
#if defined(__GNUC__) || defined(__clang__)
    constexpr int W = 8;
    typedef float f8 __attribute__((vector_size(4 * W)));
    typedef int32_t i8 __attribute__((vector_size(4 * W)));
    const f8 thr = f8{} + threshold;
    const i8 lane_bit = {1, 2, 4, 8, 16, 32, 64, 128};
    // the mask's lanes as the bits of an integer (NEON has no movemask; the
    // sum is one horizontal add)
    auto bits_of = [&](i8 m) {
        const i8 b = m & lane_bit;
        unsigned s = 0;
        for (int k = 0; k < W; ++k) s += static_cast<unsigned>(b[k]);
        return s;
    };
    auto load = [](const float* p) {
        f8 v;
        __builtin_memcpy(&v, p, sizeof v);
        return v;
    };
    for (int y = 1; y < ny - 1; ++y) {
        const float* row = img + y * nx;
        const float* above = row - nx;
        const float* below = row + nx;
        int x = 1;
        for (; x + W <= nx - 1; x += W) {
            const f8 v = load(row + x);
            // with a threshold most chunks end here, at the cost of one compare
            const i8 hi = v > thr;
            if (!bits_of(hi)) continue;
            const i8 m = hi & (v > load(row + x - 1)) & (v > load(row + x + 1)) &
                         (v > load(above + x - 1)) & (v > load(above + x)) &
                         (v > load(above + x + 1)) & (v > load(below + x - 1)) &
                         (v > load(below + x)) & (v > load(below + x + 1));
            // one bit per maximum, each found by its lowest set bit: a branch
            // per maximum and not one per pixel
            for (unsigned b = bits_of(m); b; b &= b - 1) {
                const int k = __builtin_ctz(b);
                out.push_back({frame, y, x + k, v[k]});
            }
        }
        find_maxima_span(img, nx, y, x, nx - 1, frame, threshold, out);
    }
#else
    for (int y = 1; y < ny - 1; ++y)
        find_maxima_span(img, nx, y, 1, nx - 1, frame, threshold, out);
#endif
}

}  // namespace smappy
