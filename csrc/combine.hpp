// Reducing one column to one value per group.
//
// The other half of the grouper: `connect` decides which localizations belong
// together, this turns each of those runs into a single number by the column's
// own rule (see `smappy/group.py` for the rules and why they are what they
// are).  Both halves of SMAP's grouper are compiled for the same reason -- the
// work is one pass over every localization, per column.
//
// The input is in *group order*: `order` is the permutation that sorts by
// group id and `starts[g]` is where group g begins in it, with
// `starts[n_groups]` one past the end.  Python builds both once and every
// column reuses them.
//
// Written against the rules rather than ported: the numpy version this
// replaces does the same arithmetic and is the reference the tests compare to.
//
// Why this is compiled and the numpy version is not enough:
//
//   * numpy needs the column as float64 and the weighted product as an array,
//     so a float32 column costs two 457 MB temporaries on a 57 M localization
//     table before `np.bincount` allocates its own output.  Here there are no
//     temporaries at all: the cast, the weight and the sum are one pass.
//   * `np.bincount` cannot be threaded usefully -- every thread wants its own
//     accumulator over all the groups, and at 40 M groups that is 320 MB each.
//     Partitioning by *group* instead gives every thread a disjoint slice of
//     one output and nothing to coordinate, which is what happens below.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include "parallel.hpp"

namespace smappy {

// How a column is reduced.  The names are the accumulation, not the combine
// mode: `mean` and `sum` both accumulate a sum, and Python divides afterwards.
enum class Combine {
    Sum = 0,            // sum(v)                -- "sum"
    Weighted = 1,       // sum(v * w)            -- "mean", before / sum(w)
    InverseSquare = 2,  // sum(1 / v^2)          -- "precision", before 1/sqrt
    Square = 3,         // sum(v^2)              -- "quad", before sqrt
    Min = 4,
    Max = 5,
};

template <class T, class Body>
inline void reduce_groups(const int64_t* order, const int64_t* starts,
                          int64_t n_groups, double* out, int n_threads,
                          double initial, Body step) {
    parallel_ranges(n_groups, n_threads, [&](long long g0, long long g1, int) {
        for (long long g = g0; g < g1; ++g) {
            double acc = initial;
            const int64_t end = starts[g + 1];
            for (int64_t i = starts[g]; i < end; ++i) step(acc, order[i]);
            out[g] = acc;
        }
    });
}

// One column, one value per group.  `weights` is read only for `Weighted` and
// may be null otherwise.  Threads take contiguous ranges of groups, so their
// writes to `out` never meet.
template <class T>
void combine_column(const T* values, const double* weights, const int64_t* order,
                    const int64_t* starts, int64_t n_groups, Combine mode,
                    double* out, int n_threads) {
    const double inf = std::numeric_limits<double>::infinity();
    switch (mode) {
        case Combine::Sum:
            reduce_groups<T>(order, starts, n_groups, out, n_threads, 0.0,
                             [&](double& a, int64_t j) { a += double(values[j]); });
            break;
        case Combine::Weighted:
            reduce_groups<T>(order, starts, n_groups, out, n_threads, 0.0,
                             [&](double& a, int64_t j) { a += double(values[j]) * weights[j]; });
            break;
        case Combine::InverseSquare:
            reduce_groups<T>(order, starts, n_groups, out, n_threads, 0.0,
                             [&](double& a, int64_t j) {
                                 const double v = double(values[j]);
                                 a += 1.0 / (v * v);
                             });
            break;
        case Combine::Square:
            reduce_groups<T>(order, starts, n_groups, out, n_threads, 0.0,
                             [&](double& a, int64_t j) {
                                 const double v = double(values[j]);
                                 a += v * v;
                             });
            break;
        case Combine::Min:
            reduce_groups<T>(order, starts, n_groups, out, n_threads, inf,
                             [&](double& a, int64_t j) {
                                 const double v = double(values[j]);
                                 if (v < a) a = v;
                             });
            break;
        case Combine::Max:
            reduce_groups<T>(order, starts, n_groups, out, n_threads, -inf,
                             [&](double& a, int64_t j) {
                                 const double v = double(values[j]);
                                 if (v > a) a = v;
                             });
            break;
    }
}

}  // namespace smappy
