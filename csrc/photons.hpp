// Raw camera counts into photons, and a 4Pi block's channels summed onto
// channel 0, split over threads.
//
// Both were numpy on one core in front of a threaded filter: a quarter of peak
// finding's time on 2048 x 2048 frames, and two thirds of the 4Pi channel sum's
// (docs/performance.md).  The arithmetic is numpy's, in the same order and in
// float32, so the photons are the same to the last bit: subtract, multiply,
// then divide by the excess noise -- a division, not a multiplication by the
// reciprocal, which would differ in the last place.  Each step is its own
// statement so that no compiler contracts them into a fused multiply-add.
#pragma once

#include <cstddef>
#include <cstdint>

namespace smappy {

// out[i] = ((float(raw[i]) - offset[i % period]) * gain) / excess  (the division only when excess != 1),
// for the pixels [begin, end) of a block whose per-pixel offset repeats every `period` pixels (1: one
// offset for all).  `begin` is a multiple of `period`.
template <class T>
inline void photons_range(const T* raw, float* out, std::size_t begin, std::size_t end,
                          const float* offset, std::size_t period, float gain, float excess) {
    const bool divide = excess != 1.0f;
    if (period == 1) {
        const float o = offset[0];
        for (std::size_t i = begin; i < end; ++i) {
            float v = static_cast<float>(raw[i]);
            v = v - o;
            v = v * gain;
            if (divide) v = v / excess;
            out[i] = v;
        }
        return;
    }
    for (std::size_t start = begin; start < end; start += period) {
        const T* r = raw + start;
        float* d = out + start;
        for (std::size_t p = 0; p < period; ++p) {
            float v = static_cast<float>(r[p]);
            v = v - offset[p];
            v = v * gain;
            if (divide) v = v / excess;
            d[p] = v;
        }
    }
}

// The channels of frames [begin, end) of an (n, C, h*w) block summed onto channel 0's pixels:
// channel c's pixel index[c][p] where inside[c][p], else nothing, added in channel order as numpy
// adds them (`multichannel.detection_image`); a channel with identity[c] is read pixel for pixel.
inline void channel_sum_range(const float* channels, float* out, long long begin, long long end,
                              int n_channels, std::size_t pixels, const int64_t* index,
                              const uint8_t* inside, const uint8_t* identity) {
    for (long long f = begin; f < end; ++f) {
        const float* frame = channels + static_cast<std::size_t>(f) * n_channels * pixels;
        float* d = out + static_cast<std::size_t>(f) * pixels;
        for (std::size_t p = 0; p < pixels; ++p) d[p] = frame[p];
        for (int c = 1; c < n_channels; ++c) {
            const float* ch = frame + static_cast<std::size_t>(c) * pixels;
            if (identity[c]) {
                for (std::size_t p = 0; p < pixels; ++p) d[p] = d[p] + ch[p];
                continue;
            }
            const int64_t* idx = index + static_cast<std::size_t>(c) * pixels;
            const uint8_t* in = inside + static_cast<std::size_t>(c) * pixels;
            for (std::size_t p = 0; p < pixels; ++p)
                d[p] = d[p] + (in[p] ? ch[idx[p]] : 0.0f);
        }
    }
}

}  // namespace smappy
