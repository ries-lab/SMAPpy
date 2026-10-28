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

// The cutoff of `cutoff_sigmas` sigma as a bound on the exponent,
// |d|^2 / (4 s^2) > cutoff^2 / 4.
inline double cutoff_exponent(double cutoff_sigmas) {
    return cutoff_sigmas > 0.0 ? cutoff_sigmas * cutoff_sigmas / 4.0 : 0.0;
}

// `cutoff_sigmas <= 0` disables the cutoff, giving the exact kernel; a positive
// value skips pairs further apart than that many sigma, whose contribution is
// below exp(-cutoff^2/4) and which are the large majority once sigma is small.
// Templated on the caller's dtypes on purpose.  The optimizer hands over
// float32 coordinates and int32 indices and calls this a few hundred times with
// the same arrays; converting them here would copy gigabytes per evaluation,
// which is precisely what COMET's `_fast` path exists to avoid.
//
// smappy: the width is per axis.  `z_factor` is sigma_z / sigma_xy, so a pair's
// distance is measured as dx^2 + dy^2 + (dz/k)^2 -- the Mahalanobis distance of
// a Gaussian k times as wide in z -- and the cutoff is on that distance.  k = 1
// is COMET.  A fixed k > 1 was measured worse than the precision weights below
// at the default sigma (docs/drift.md, "Precision weights and the width in z").
//
// `variances` (n, 3), or null, is each localization's own variance per axis
// (its precision squared): a pair's Gaussian then has the variance
// V_k = 2 s_k^2 + var_i,k + var_j,k per axis and is normalised as a density,
// sqrt(prod V0_k / prod V_k) / s with V0_k = 2 s_k^2, so that a pair of exact
// localizations weighs what it did without and imprecise ones count less --
// DME's weighting (Cnossen et al. 2021) on COMET's cost.  Null is COMET's.
namespace detail {
// What a pair needs from the widths: 1/(2 V_k) per axis and the norm.
struct PairWidth {
    double half_inv[3];   // 1 / (2 V_k)
    double norm;          // the density's normalisation, relative to COMET's
};

inline PairWidth constant_width(double s_eff, double z_factor) {
    const double v_xy = 2.0 * s_eff * s_eff;
    const double v_z = v_xy * z_factor * z_factor;
    return {{0.5 / v_xy, 0.5 / v_xy, 0.5 / v_z}, 1.0 / s_eff};
}

inline PairWidth pair_width(const PairWidth& base, const double* vi,
                            const double* vj) {
    PairWidth w;
    double ratio = 1.0;
    for (int k = 0; k < 3; ++k) {
        const double v0 = 0.5 / base.half_inv[k];
        const double v = v0 + vi[k] + vj[k];
        w.half_inv[k] = 0.5 / v;
        ratio *= v0 / v;
    }
    w.norm = base.norm * std::sqrt(ratio);
    return w;
}
}  // namespace detail

template <class Coord, class Index>
DriftCost cost_and_gradient(const Coord* coords, const Index* times,
                            const Index* idx_i, const Index* idx_j,
                            const double* mu, int64_t n_pairs,
                            int64_t n_segments, double sigma,
                            double sigma_factor, double cutoff_sigmas,
                            int n_threads, double z_factor = 1.0,
                            const double* variances = nullptr) {
    const double s_eff = sigma * sigma_factor;
    const detail::PairWidth base = detail::constant_width(s_eff, z_factor);
    // the cutoff on the Mahalanobis distance: sum d_k^2 / (2 V_k) against
    // cutoff^2 / 4, which for one sigma is |d| > cutoff * s
    const double cutoff_m = cutoff_exponent(cutoff_sigmas);
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

            const double d[3] = {(double(ci[0]) - mi[0]) - (double(cj[0]) - mj[0]),
                                 (double(ci[1]) - mi[1]) - (double(cj[1]) - mj[1]),
                                 (double(ci[2]) - mi[2]) - (double(cj[2]) - mj[2])};
            const detail::PairWidth w =
                variances ? detail::pair_width(base, variances + 3 * i,
                                               variances + 3 * j)
                          : base;
            const double m = d[0] * d[0] * w.half_inv[0] + d[1] * d[1] * w.half_inv[1] +
                             d[2] * d[2] * w.half_inv[2];
            if (use_cutoff && m > cutoff_m) continue;

            const double val = std::exp(-m) * w.norm;
            local += val;

            // the two contributions are exact negatives of each other, so the
            // difference is formed once and applied to both windows:
            // d val / d d_k = -val d_k / V_k
            double* gi = deri + 3 * ti;
            double* gj = deri + 3 * tj;
            for (int k = 0; k < 3; ++k) {
                const double c = -2.0 * val * w.half_inv[k] * d[k];
                gj[k] += c;
                gi[k] -= c;
            }
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
    double sigma, double sigma_factor, int n_threads, double z_factor = 1.0) {
    const double s_eff = sigma * sigma_factor;
    const double inv_sigma_sq = 1.0 / ((2.0 * s_eff) * (2.0 * s_eff));
    const double inv_z_sq = 1.0 / (z_factor * z_factor);   // per axis, as above
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
                std::exp(-(dx * dx + dy * dy + dz * dz * inv_z_sq) * inv_sigma_sq) *
                inv_sigma;
            const double none =
                std::exp(-(ux * ux + uy * uy + uz * uz * inv_z_sq) * inv_sigma_sq) *
                inv_sigma;

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

// The same cost when each frame also has a linear transformation: the
// corrected position of a localization in frame f is A_f x - d_f, with x
// measured from the centre of the data (the caller subtracts it).  This is
// what fits a sample that expands, or optics whose magnification drifts, in
// `smappy.drift` (the *transformation* setting).
//
// `params` is (n_frames, 14) per frame: A row major (9), d (3), and two
// scale factors, g_xy for x and y and g_z for z: the inverse linear scale
// of the frame's transformation in those axes (the caller's choice, e.g.
// det(A_xy)^(-1/2), and 1 for a z that is only shifted).  A pair's
// difference in x and y is multiplied by sqrt(g_xy,i g_xy,j), in z by
// sqrt(g_z,i g_z,j): the geometric mean of the two frames' scales.  Without
// it the overlap grows when everything shrinks and the fit collapses the
// image; with it a transformation common to both frames of a pair cancels,
// so only the change of the transformation is measured -- as only the
// change of the drift is.  The g are inputs, not computed here, so that
// their derivatives with respect to A are the caller's: the gradient
// returns d(total)/dg alongside d/dA and d/dd.
struct TransformedCost {
    double total = 0.0;
    std::vector<double> gradient;   // (n_frames, 14), row major, as `params`
};

template <class Coord, class Index>
TransformedCost transformed_cost_and_gradient(
        const Coord* coords, const Index* times, const Index* idx_i,
        const Index* idx_j, const double* params, int64_t n_pairs,
        int64_t n_frames, double sigma, double sigma_factor,
        double cutoff_sigmas, int n_threads, double z_factor = 1.0) {
    const double s_eff = sigma * sigma_factor;
    const double inv_sigma_sq = 1.0 / ((2.0 * s_eff) * (2.0 * s_eff));
    const double inv_sigma = 1.0 / s_eff;
    // per axis, as in `cost_and_gradient`: z measured in units k times wider
    const double axis_w[3] = {1.0, 1.0, 1.0 / (z_factor * z_factor)};
    const double cutoff_sq =
        cutoff_sigmas > 0.0 ? (cutoff_sigmas * s_eff) * (cutoff_sigmas * s_eff) : 0.0;
    const bool use_cutoff = cutoff_sigmas > 0.0;
    constexpr int W = 14;

    const int threads = resolve_threads(n_threads, n_pairs);
    const std::size_t width = static_cast<std::size_t>(n_frames) * W;
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
            const double* pi = params + W * ti;
            const double* pj = params + W * tj;
            const double xi[3] = {double(coords[3 * i]), double(coords[3 * i + 1]),
                                  double(coords[3 * i + 2])};
            const double xj[3] = {double(coords[3 * j]), double(coords[3 * j + 1]),
                                  double(coords[3 * j + 2])};

            // the scale of x and y (h[0], h[1]) and of z (h[2])
            const double hxy = std::sqrt(pi[12] * pj[12]);
            const double hz = std::sqrt(pi[13] * pj[13]);
            const double h[3] = {hxy, hxy, hz};
            double D[3], Dn[3];
            double diff_sq = 0.0;
            for (int r = 0; r < 3; ++r) {
                const double ci = pi[3 * r] * xi[0] + pi[3 * r + 1] * xi[1] +
                                  pi[3 * r + 2] * xi[2] - pi[9 + r];
                const double cj = pj[3 * r] * xj[0] + pj[3 * r + 1] * xj[1] +
                                  pj[3 * r + 2] * xj[2] - pj[9 + r];
                D[r] = ci - cj;
                Dn[r] = h[r] * D[r];
                diff_sq += axis_w[r] * Dn[r] * Dn[r];
            }
            if (use_cutoff && diff_sq > cutoff_sq) continue;

            const double val = std::exp(-diff_sq * inv_sigma_sq) * inv_sigma;
            local += val;

            const double weight = 2.0 * val * inv_sigma_sq;
            double G[3];          // d val / d D
            for (int r = 0; r < 3; ++r) G[r] = -weight * axis_w[r] * Dn[r] * h[r];
            // d val / d h, for the xy and the z scale
            const double dvdhxy = -weight * (Dn[0] * D[0] + Dn[1] * D[1]);
            const double dvdhz = -weight * axis_w[2] * Dn[2] * D[2];
            double* gi = deri + W * ti;
            double* gj = deri + W * tj;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    gi[3 * r + c] += G[r] * xi[c];
                    gj[3 * r + c] -= G[r] * xj[c];
                }
                gi[9 + r] -= G[r];
                gj[9 + r] += G[r];
            }
            // h = sqrt(g_i g_j): dh/dg_i = h / (2 g_i)
            gi[12] += dvdhxy * hxy / (2.0 * pi[12]);
            gj[12] += dvdhxy * hxy / (2.0 * pj[12]);
            gi[13] += dvdhz * hz / (2.0 * pi[13]);
            gj[13] += dvdhz * hz / (2.0 * pj[13]);
        }
        totals[static_cast<std::size_t>(t)] = local;
    });

    TransformedCost out;
    out.gradient.assign(width, 0.0);
    for (int t = 0; t < threads; ++t) {
        const double* buf = buffers.data() + static_cast<std::size_t>(t) * width;
        for (std::size_t k = 0; k < width; ++k) out.gradient[k] += buf[k];
        out.total += totals[static_cast<std::size_t>(t)];
    }
    return out;
}

}  // namespace smappy
