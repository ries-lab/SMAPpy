// The COMET drift cost function: how well localizations from different time
// windows overlap once the current drift estimate is subtracted.
//
// For every neighbour pair (i, j) the cost sums a Gaussian of the distance
// between the two drift-corrected positions, and the gradient is that sum's
// derivative with respect to each time window's drift.  The optimizer calls
// this a few hundred times over every pair in the dataset -- hundreds of
// millions of them -- so it is the whole running time of a drift correction.
//
// The maths is COMET's, stated in Python in `smappy/_comet/core/cpu_wrapper.py`
// as `_cost_and_gradient_reference`, which this must agree with.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "parallel.hpp"

namespace smappy {

// The gradient is a scatter-add into per-window bins, so threads cannot share
// one output.  Each gets its own (n_segments, 3) buffer -- a few kilobytes --
// and the buffers are summed once at the end.
struct DriftCost {
    double total = 0.0;
    std::vector<double> gradient;   // (n_segments, 3), row major
};

// `cutoff_sigmas <= 0` disables the cutoff, giving the exact kernel; a positive
// value skips pairs further apart than that many sigma, whose contribution is
// below exp(-cutoff^2/4) and which are the large majority once sigma is small.
// Templated on the caller's dtypes on purpose.  The optimizer hands over
// float32 coordinates and int32 indices and calls this a few hundred times with
// the same arrays; converting them here would copy gigabytes per evaluation,
// which is precisely what COMET's `_fast` path exists to avoid.
template <class Coord, class Index>
DriftCost cost_and_gradient(const Coord* coords, const Index* times,
                            const Index* idx_i, const Index* idx_j,
                            const double* mu, int64_t n_pairs,
                            int64_t n_segments, double sigma,
                            double sigma_factor, double cutoff_sigmas,
                            int n_threads) {
    const double s_eff = sigma * sigma_factor;
    const double sigma_sq = (2.0 * s_eff) * (2.0 * s_eff);
    const double inv_sigma_sq = 1.0 / sigma_sq;
    const double inv_sigma = 1.0 / s_eff;
    const double cutoff_sq =
        cutoff_sigmas > 0.0 ? (cutoff_sigmas * s_eff) * (cutoff_sigmas * s_eff) : 0.0;
    const bool use_cutoff = cutoff_sigmas > 0.0;

    const int threads = resolve_threads(n_threads, n_pairs);
    const std::size_t width = static_cast<std::size_t>(n_segments) * 3;
    std::vector<double> buffers(static_cast<std::size_t>(threads) * width, 0.0);
    std::vector<double> totals(static_cast<std::size_t>(threads), 0.0);

    parallel_ranges(n_pairs, threads, [&](long long begin, long long end, int t) {
        double* deri = buffers.data() + static_cast<std::size_t>(t) * width;
        double local = 0.0;
        for (long long p = begin; p < end; ++p) {
            const int64_t i = static_cast<int64_t>(idx_i[p]);
            const int64_t j = static_cast<int64_t>(idx_j[p]);
            const int64_t ti = static_cast<int64_t>(times[i]);
            const int64_t tj = static_cast<int64_t>(times[j]);

            const Coord* ci = coords + 3 * i;
            const Coord* cj = coords + 3 * j;
            const double* mi = mu + 3 * ti;
            const double* mj = mu + 3 * tj;

            const double dx = (double(ci[0]) - mi[0]) - (double(cj[0]) - mj[0]);
            const double dy = (double(ci[1]) - mi[1]) - (double(cj[1]) - mj[1]);
            const double dz = (double(ci[2]) - mi[2]) - (double(cj[2]) - mj[2]);
            const double diff_sq = dx * dx + dy * dy + dz * dz;
            if (use_cutoff && diff_sq > cutoff_sq) continue;

            const double val = std::exp(-diff_sq * inv_sigma_sq) * inv_sigma;
            local += val;

            // the two contributions are exact negatives of each other, so the
            // difference is formed once and applied to both windows
            const double weight = 2.0 * val * inv_sigma_sq;
            double* gi = deri + 3 * ti;
            double* gj = deri + 3 * tj;
            const double cx = -weight * dx;   // == weight * (cj[0]-ci[0]+mi[0]-mj[0])
            const double cy = -weight * dy;
            const double cz = -weight * dz;
            gj[0] += cx; gi[0] -= cx;
            gj[1] += cy; gi[1] -= cy;
            gj[2] += cz; gi[2] -= cz;
        }
        totals[static_cast<std::size_t>(t)] = local;
    });

    DriftCost out;
    out.gradient.assign(width, 0.0);
    for (int t = 0; t < threads; ++t) {
        const double* buf = buffers.data() + static_cast<std::size_t>(t) * width;
        for (std::size_t k = 0; k < width; ++k) out.gradient[k] += buf[k];
        out.total += totals[static_cast<std::size_t>(t)];
    }
    return out;
}

// Per-window overlap for quality control: the overlap achieved with the fitted
// drift, the overlap the same pairs would have with no drift at all, and how
// many pairs each window has.  A window whose observed overlap never rose above
// its own null is one the estimate failed on.
//
// Same-window pairs are skipped: their overlap does not depend on the drift, so
// they say nothing about whether that window's estimate is any good.  Every pair
// counts for both of its windows, as the GPU kernel does.
struct SegmentOverlap {
    std::vector<double> observed;
    std::vector<double> null;
    std::vector<double> counts;
};

template <class Coord, class Index>
SegmentOverlap overlap_per_segment(
    const Coord* coords, const Index* times, const Index* idx_i,
    const Index* idx_j, const double* mu, int64_t n_pairs, int64_t n_segments,
    double sigma, double sigma_factor, int n_threads) {
    const double s_eff = sigma * sigma_factor;
    const double inv_sigma_sq = 1.0 / ((2.0 * s_eff) * (2.0 * s_eff));
    const double inv_sigma = 1.0 / s_eff;

    const int threads = resolve_threads(n_threads, n_pairs);
    const std::size_t width = static_cast<std::size_t>(n_segments);
    std::vector<double> obs(static_cast<std::size_t>(threads) * width, 0.0);
    std::vector<double> nul(static_cast<std::size_t>(threads) * width, 0.0);
    std::vector<double> cnt(static_cast<std::size_t>(threads) * width, 0.0);

    parallel_ranges(n_pairs, threads, [&](long long begin, long long end, int t) {
        const std::size_t base = static_cast<std::size_t>(t) * width;
        double* o = obs.data() + base;
        double* nz = nul.data() + base;
        double* c = cnt.data() + base;
        for (long long p = begin; p < end; ++p) {
            const int64_t i = static_cast<int64_t>(idx_i[p]);
            const int64_t j = static_cast<int64_t>(idx_j[p]);
            const int64_t ti = static_cast<int64_t>(times[i]);
            const int64_t tj = static_cast<int64_t>(times[j]);
            if (ti == tj) continue;

            const Coord* ci = coords + 3 * i;
            const Coord* cj = coords + 3 * j;
            const double* mi = mu + 3 * ti;
            const double* mj = mu + 3 * tj;

            const double ux = double(ci[0]) - double(cj[0]);
            const double uy = double(ci[1]) - double(cj[1]);
            const double uz = double(ci[2]) - double(cj[2]);
            const double dx = ux - (mi[0] - mj[0]);
            const double dy = uy - (mi[1] - mj[1]);
            const double dz = uz - (mi[2] - mj[2]);

            const double val =
                std::exp(-(dx * dx + dy * dy + dz * dz) * inv_sigma_sq) * inv_sigma;
            const double none =
                std::exp(-(ux * ux + uy * uy + uz * uz) * inv_sigma_sq) * inv_sigma;

            o[ti] += val;  o[tj] += val;
            nz[ti] += none; nz[tj] += none;
            c[ti] += 1.0;  c[tj] += 1.0;
        }
    });

    SegmentOverlap out;
    out.observed.assign(width, 0.0);
    out.null.assign(width, 0.0);
    out.counts.assign(width, 0.0);
    for (int t = 0; t < threads; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * width;
        for (std::size_t k = 0; k < width; ++k) {
            out.observed[k] += obs[base + k];
            out.null[k] += nul[base + k];
            out.counts[k] += cnt[base + k];
        }
    }
    return out;
}

}  // namespace smappy
