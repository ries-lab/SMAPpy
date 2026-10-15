// What is around each localization: the neighbours within a radius (a circle,
// a cylinder, or Gaussian-weighted) and a statistic of a column over them.
// The Python side is `smappy.plugins.neighbourhood`.
//
// SMAP's countneighbours*.c and SpatialFilterLocs sort the points by x and,
// for each point, test every point of the strip [x - r, x + r]: no pairs are
// ever stored, but a strip is as tall as the field.  Measured, parallel over
// the points as here: 1 M localizations over 20 um put 5000 points in a strip
// for 20 in the circle, 2.4 s; 5 M over 40 um, 12500 for 25, 23 s.  Here the
// neighbours are sorted by a square cell one search radius wide, row after
// row, so a point's candidates are three runs of the sorted list -- the rows
// above, at and below it, three cells each -- found by binary search: about
// 9 / pi times the neighbours there are, whatever the size of the field, and
// no grid is allocated however sparse the points: 0.5 s and 3 s for the two
// above, the sort included.
//
// The sort was then most of the time -- 2 s of 4.7 for 5 M points, an index
// sorted through a comparison that looks its key up.  It is now the
// grouping's sort (`group.sorted_order`): the high part of the key, the
// cell's row, is a partition worked out by counting, with no comparisons, and
// each row -- a few thousand points -- is then sorted by its column on its
// own thread, small enough to stay in cache.
//
// `Index` is built once from the neighbours, and asked about the points
// around in pieces, so the caller can report progress and stop between them;
// `cell_order` puts those points in the same cell order, so that neighbouring
// queries read the same memory.
//
// `pair_histogram` is the same search with nothing written per point: the
// distances of every pair closer than the cell size, binned, one histogram
// per thread summed at the end -- the pair counts of Spatial statistics'
// g(r) and K(r) (`smappy.plugins.spatial_statistics`).  No pair is ever
// stored, so 10^9 pairs cost a second, not 16 GB.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "parallel.hpp"

namespace py = pybind11;

namespace {

using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Ints = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using Pair = std::pair<int64_t, int64_t>;          // (cell key, index)

enum Mode { Count = 0, Gauss = 1, Mean = 2, Median = 3, Min = 4, Max = 5, Std = 6,
            Quantile = 7, Ball = 8 };

struct Grid {
    double x0, y0, cell;
    int64_t width;
    int64_t cx(double x) const { return static_cast<int64_t>(std::floor((x - x0) / cell)); }
    int64_t cy(double y) const { return static_cast<int64_t>(std::floor((y - y0) / cell)); }
    int64_t key(int64_t ix, int64_t iy) const { return iy * width + ix; }
    int64_t key_of(double x, double y) const { return key(cx(x), cy(y)); }
};

// The points in cell order -- by row, then column -- with their keys.  Rows
// by a stable counting pass, then each row sorted by column on a thread.
std::vector<Pair> sorted_keys(const double* x, const double* y, int64_t n,
                              const Grid& grid, int threads) {
    std::vector<int64_t> row(n), col(n);
    smappy::parallel_ranges(n, threads, [&](long long b, long long e, int) {
        for (long long i = b; i < e; ++i) { row[i] = grid.cy(y[i]); col[i] = grid.cx(x[i]); }
    });
    int64_t rows = 0;
    for (int64_t i = 0; i < n; ++i) rows = std::max(rows, row[i] + 1);
    std::vector<Pair> keys(n);                    // (column, index), then (key, index)
    if (rows > 4 * n + 1000000) {
        // far more rows than points -- a tiny radius over a sparse field: the
        // table of rows would be the cost, so a plain sort of the keys
        for (int64_t i = 0; i < n; ++i) keys[i] = {grid.key(col[i], row[i]), i};
        std::sort(keys.begin(), keys.end());
        return keys;
    }
    std::vector<int64_t> start(rows + 1, 0);
    for (int64_t i = 0; i < n; ++i) ++start[row[i] + 1];
    for (int64_t r = 0; r < rows; ++r) start[r + 1] += start[r];
    {
        std::vector<int64_t> at(start.begin(), start.end() - 1);
        for (int64_t i = 0; i < n; ++i) keys[at[row[i]]++] = {col[i], i};
    }
    smappy::parallel_ranges(rows, threads, [&](long long b, long long e, int) {
        for (long long r = b; r < e; ++r) {
            std::sort(keys.begin() + start[r], keys.begin() + start[r + 1]);
            for (int64_t k = start[r]; k < start[r + 1]; ++k)
                keys[k].first = grid.key(keys[k].first, r);
        }
    });
    return keys;
}

// The value at quantile q of v (sorted in place, partly), linear between the
// order statistics as numpy's default.
double quantile_of(std::vector<double>& v, double q) {
    const size_t n = v.size();
    const double at = q * static_cast<double>(n - 1);
    const size_t lo = static_cast<size_t>(std::floor(at));
    const double frac = at - static_cast<double>(lo);
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    const double low = v[lo];
    if (frac <= 0.0 || lo + 1 >= n) return low;
    const double high = *std::min_element(v.begin() + lo + 1, v.end());
    return low + (high - low) * frac;
}

Grid make_grid(double x0, double y0, double x1, double reach) {
    if (!(reach > 0)) throw std::invalid_argument("the radius must be positive");
    return Grid{x0 - reach, y0 - reach, reach,
                static_cast<int64_t>(std::floor((x1 - x0) / reach)) + 4};
}

// The order that visits points cell by cell, as `Index` stores its own.
Ints cell_order(const Doubles& x, const Doubles& y, double x0, double y0, double x1,
                double reach, int threads) {
    const int64_t n = x.size();
    if (y.size() != n) throw std::invalid_argument("x and y must match");
    const Grid grid = make_grid(x0, y0, x1, reach);
    Ints out(n);
    int64_t* o = out.mutable_data();
    const double* px = x.data();
    const double* py = y.data();
    py::gil_scoped_release release;
    const auto keys = sorted_keys(px, py, n, grid, threads);
    for (int64_t i = 0; i < n; ++i) o[i] = keys[i].second;
    return out;
}

class Index {
public:
    // The neighbours, sorted by cell.  The grid -- (x0, y0) the smallest x
    // and y of both sets, x1 the largest x -- must be the one `cell_order` is
    // given for the points asked about.
    Index(const Doubles& x, const Doubles& y, py::object z_obj, py::object values_obj,
          double x0, double y0, double x1, double reach, int threads)
        : grid_(make_grid(x0, y0, x1, reach)) {
        const int64_t n = x.size();
        if (y.size() != n) throw std::invalid_argument("x and y must match");
        Doubles z, values;
        has_z_ = !z_obj.is_none();
        has_values_ = !values_obj.is_none();
        if (has_z_) {
            z = z_obj.cast<Doubles>();
            if (z.size() != n) throw std::invalid_argument("z must match");
        }
        if (has_values_) {
            values = values_obj.cast<Doubles>();
            if (values.size() != n) throw std::invalid_argument("values must match");
        }
        const double* px = x.data();
        const double* py = y.data();
        const double* pz = has_z_ ? z.data() : nullptr;
        const double* pv = has_values_ ? values.data() : nullptr;
        py::gil_scoped_release release;
        const auto keys = sorted_keys(px, py, n, grid_, threads);
        key_.resize(n); index_.resize(n); x_.resize(n); y_.resize(n);
        if (has_z_) z_.resize(n);
        if (has_values_) v_.resize(n);
        smappy::parallel_ranges(n, threads, [&](long long b, long long e, int) {
            for (long long k = b; k < e; ++k) {
                const int64_t j = keys[k].second;
                key_[k] = keys[k].first; index_[k] = j; x_[k] = px[j]; y_[k] = py[j];
                if (pz) z_[k] = pz[j];
                if (pv) v_[k] = pv[j];
            }
        });
    }

    // Per point asked about: mode, sigma (Gaussian), z_reach / z_sigma (<= 0:
    // no z), q (quantile).  `same[i]` is the neighbour that is point i itself,
    // or -1; it is skipped when counting and kept for a statistic.  A point
    // is a neighbour within the grid's cell laterally.
    py::array_t<double> query(const Doubles& ax, const Doubles& ay, py::object az_obj,
                              py::object same_obj, int mode, double sigma,
                              double z_reach, double z_sigma, double q, int threads) const {
        const int64_t na = ax.size();
        if (ay.size() != na) throw std::invalid_argument("x and y must match");
        const bool use_z = z_reach > 0 && has_z_ && !az_obj.is_none();
        const bool stat = mode >= Mean && mode != Ball;
        if (stat && !has_values_) throw std::invalid_argument("a statistic needs values");
        Doubles az;
        Ints same;
        if (use_z) {
            az = az_obj.cast<Doubles>();
            if (az.size() != na) throw std::invalid_argument("z must match");
        }
        const bool have_same = !same_obj.is_none();
        if (have_same) {
            same = same_obj.cast<Ints>();
            if (same.size() != na) throw std::invalid_argument("same must match");
        }
        py::array_t<double> out(na);
        double* res = out.mutable_data();
        const double fill = stat ? std::numeric_limits<double>::quiet_NaN() : 0.0;
        std::fill_n(res, na, fill);
        if (na == 0 || key_.empty()) return out;
        const double* pax = ax.data();
        const double* pay = ay.data();
        const double* paz = use_z ? az.data() : nullptr;
        const int64_t* ps = have_same ? same.data() : nullptr;
        const double r2 = grid_.cell * grid_.cell, z2 = z_reach * z_reach;
        const double gxy = sigma > 0 ? 1.0 / (2 * sigma * sigma) : 0.0;
        const double gz = z_sigma > 0 ? 1.0 / (2 * z_sigma * z_sigma) : 0.0;

        py::gil_scoped_release release;
        smappy::parallel_ranges(na, threads, [&](long long begin, long long end, int) {
            std::vector<double> found;
            // the three runs of the sorted list of the last cell asked about:
            // in cell order the next point is mostly in the same cell
            int64_t last_x = std::numeric_limits<int64_t>::min(), last_y = 0;
            int64_t run_lo[3] = {0, 0, 0}, run_hi[3] = {0, 0, 0};
            for (long long i = begin; i < end; ++i) {
                const double x = pax[i], y = pay[i], z = use_z ? paz[i] : 0.0;
                const int64_t self = ps ? ps[i] : -1;
                const int64_t ix = grid_.cx(x), iy = grid_.cy(y);
                if (ix != last_x || iy != last_y) {
                    for (int64_t dy = -1; dy <= 1; ++dy) {
                        auto lo = std::lower_bound(key_.begin(), key_.end(),
                                                   grid_.key(ix - 1, iy + dy));
                        auto hi = std::upper_bound(lo, key_.end(), grid_.key(ix + 1, iy + dy));
                        run_lo[dy + 1] = lo - key_.begin();
                        run_hi[dy + 1] = hi - key_.begin();
                    }
                    last_x = ix;
                    last_y = iy;
                }
                double total = 0.0;
                found.clear();
                for (int run = 0; run < 3; ++run) {
                    for (int64_t s = run_lo[run], e = run_hi[run]; s < e; ++s) {
                        const double ddx = x_[s] - x, ddy = y_[s] - y;
                        const double d2 = ddx * ddx + ddy * ddy;
                        if (d2 > r2) continue;
                        double dz2 = 0.0;
                        if (use_z) {
                            const double ddz = z_[s] - z;
                            dz2 = ddz * ddz;
                            if (dz2 > z2) continue;
                        }
                        if (mode == Ball && d2 + dz2 > r2) continue;  // a sphere
                        if (stat) {
                            const double v = v_[s];
                            if (std::isfinite(v)) found.push_back(v);
                            continue;
                        }
                        if (index_[s] == self) continue;       // not its own neighbour
                        total += mode == Gauss ? std::exp(-d2 * gxy - dz2 * gz) : 1.0;
                    }
                }
                if (!stat) { res[i] = total; continue; }
                const size_t n = found.size();
                if (!n) continue;
                double r = 0.0;
                switch (mode) {
                case Mean: case Std: {
                    double sum = 0.0, sq = 0.0;
                    for (double v : found) { sum += v; sq += v * v; }
                    const double mean = sum / n;
                    r = mode == Mean ? mean : std::sqrt(std::max(sq / n - mean * mean, 0.0));
                    break;
                }
                case Min: r = *std::min_element(found.begin(), found.end()); break;
                case Max: r = *std::max_element(found.begin(), found.end()); break;
                case Median: r = quantile_of(found, 0.5); break;
                default: r = quantile_of(found, q); break;
                }
                res[i] = r;
            }
        });
        return out;
    }

    // DBSCAN's clusters, given which of the indexed points are core points
    // (`core`, in the order the points were given): per point the cluster it
    // is in, -1 for none.  Core points within the cell size of each other --
    // in 3D a sphere, with `use_z` -- are one cluster, found by union-find
    // while scanning the cells, so the pairs are never stored; a point that
    // is not core joins the cluster of its nearest core point within reach.
    // Clusters are numbered in the order of their lowest index, as
    // scipy.sparse.csgraph.connected_components numbers them.
    Ints dbscan_join(const py::array_t<bool, py::array::c_style | py::array::forcecast>& core,
                     bool use_z, int threads) const {
        const int64_t n = static_cast<int64_t>(key_.size());
        if (core.size() != n) throw std::invalid_argument("core must match the points");
        if (use_z && !has_z_) throw std::invalid_argument("no z to cluster in");
        Ints out(n);
        int64_t* res = out.mutable_data();
        std::fill_n(res, n, int64_t(-1));
        if (n == 0) return out;
        const bool* pc = core.data();
        const double r2 = grid_.cell * grid_.cell;

        py::gil_scoped_release release;
        std::vector<char> is_core(n);              // by sorted position
        for (int64_t s = 0; s < n; ++s) is_core[s] = pc[index_[s]];
        std::vector<int64_t> parent(n);
        for (int64_t s = 0; s < n; ++s) parent[s] = s;
        auto find = [&](int64_t a) {
            while (parent[a] != a) {
                parent[a] = parent[parent[a]];        // path halving
                a = parent[a];
            }
            return a;
        };
        auto near = [&](int64_t a, int64_t b) {
            const double dx = x_[a] - x_[b], dy = y_[a] - y_[b];
            double d2 = dx * dx + dy * dy;
            if (use_z) { const double dz = z_[a] - z_[b]; d2 += dz * dz; }
            return d2;
        };
        // the runs of the sorted list around position s: the cells of the
        // rows above, at and below it, three each
        auto runs = [&](int64_t s, int64_t* lo, int64_t* hi) {
            const int64_t ix = key_[s] % grid_.width, iy = key_[s] / grid_.width;
            for (int64_t dy = -1; dy <= 1; ++dy) {
                auto a = std::lower_bound(key_.begin(), key_.end(), grid_.key(ix - 1, iy + dy));
                auto b = std::upper_bound(a, key_.end(), grid_.key(ix + 1, iy + dy));
                lo[dy + 1] = a - key_.begin();
                hi[dy + 1] = b - key_.begin();
            }
        };
        // core pairs: union-find, each pair once (the later of the two)
        int64_t lo[3], hi[3], last = -1;
        for (int64_t s = 0; s < n; ++s) {
            if (!is_core[s]) continue;
            if (last < 0 || key_[s] != key_[last]) runs(s, lo, hi);
            last = s;
            for (int run = 0; run < 3; ++run)
                for (int64_t t = lo[run]; t < hi[run]; ++t) {
                    if (t >= s || !is_core[t] || near(s, t) > r2) continue;
                    int64_t a = find(s), b = find(t);
                    if (a != b) parent[std::max(a, b)] = std::min(a, b);
                }
        }
        // each cluster numbered by its lowest index
        std::vector<int64_t> lowest(n, std::numeric_limits<int64_t>::max());
        for (int64_t s = 0; s < n; ++s)
            if (is_core[s]) {
                const int64_t r = find(s);
                lowest[r] = std::min(lowest[r], index_[s]);
            }
        std::vector<int64_t> roots;
        for (int64_t s = 0; s < n; ++s)
            if (is_core[s] && parent[s] == s) roots.push_back(s);
        std::sort(roots.begin(), roots.end(),
                  [&](int64_t a, int64_t b) { return lowest[a] < lowest[b]; });
        std::vector<int64_t> label(n, -1);
        for (size_t c = 0; c < roots.size(); ++c) label[roots[c]] = static_cast<int64_t>(c);
        std::vector<int64_t> cluster(n, -1);    // by sorted position
        for (int64_t s = 0; s < n; ++s)
            if (is_core[s]) cluster[s] = label[find(s)];
        // border points: the nearest core point within reach, on every core
        smappy::parallel_ranges(n, threads, [&](long long b, long long e, int) {
            int64_t lo[3], hi[3], last = -1;
            for (long long s = b; s < e; ++s) {
                if (is_core[s]) { res[index_[s]] = cluster[s]; continue; }
                if (last < 0 || key_[s] != key_[last]) runs(s, lo, hi);
                last = s;
                double best = r2;
                int64_t found = -1;
                for (int run = 0; run < 3; ++run)
                    for (int64_t t = lo[run]; t < hi[run]; ++t) {
                        if (!is_core[t]) continue;
                        const double d2 = near(s, t);
                        if (d2 <= best && (found < 0 || d2 < best ||
                                           index_[t] < index_[found])) {
                            best = d2;
                            found = t;
                        }
                    }
                res[index_[s]] = found < 0 ? -1 : cluster[found];
            }
        });
        return out;
    }

    // The histogram of the distances from each point asked about to the
    // indexed points within the cell size, in ``bins`` bins of ``dr`` from 0:
    // a pair at distance d is in bin floor(d / dr).  ``same[i]`` is the
    // indexed point that is point i itself, or -1, and is left out: a point
    // is not its own pair.  Each ordered pair counts once, so a set against
    // itself counts every pair twice.  Lateral only.
    py::array_t<double> pair_histogram(const Doubles& ax, const Doubles& ay,
                                       py::object same_obj, double dr, int64_t bins,
                                       int threads) const {
        const int64_t na = ax.size();
        if (ay.size() != na) throw std::invalid_argument("x and y must match");
        if (!(dr > 0) || bins <= 0) throw std::invalid_argument("dr and bins must be positive");
        Ints same;
        const bool have_same = !same_obj.is_none();
        if (have_same) {
            same = same_obj.cast<Ints>();
            if (same.size() != na) throw std::invalid_argument("same must match");
        }
        py::array_t<double> out(bins);
        double* res = out.mutable_data();
        std::fill_n(res, bins, 0.0);
        if (na == 0 || key_.empty()) return out;
        const double* pax = ax.data();
        const double* pay = ay.data();
        const int64_t* ps = have_same ? same.data() : nullptr;
        const double r2 = grid_.cell * grid_.cell, inv = 1.0 / dr;
        const int n_threads = smappy::resolve_threads(threads, na);
        // integer counts per thread: a double would lose a count past 2^53,
        // and they are added once at the end
        std::vector<std::vector<int64_t>> local(n_threads, std::vector<int64_t>(bins, 0));

        {
            py::gil_scoped_release release;
            smappy::parallel_ranges(na, n_threads, [&](long long begin, long long end, int t) {
                std::vector<int64_t>& h = local[t];
                int64_t last_x = std::numeric_limits<int64_t>::min(), last_y = 0;
                int64_t run_lo[3] = {0, 0, 0}, run_hi[3] = {0, 0, 0};
                for (long long i = begin; i < end; ++i) {
                    const double x = pax[i], y = pay[i];
                    const int64_t self = ps ? ps[i] : -1;
                    const int64_t ix = grid_.cx(x), iy = grid_.cy(y);
                    if (ix != last_x || iy != last_y) {
                        for (int64_t dy = -1; dy <= 1; ++dy) {
                            auto lo = std::lower_bound(key_.begin(), key_.end(),
                                                       grid_.key(ix - 1, iy + dy));
                            auto hi = std::upper_bound(lo, key_.end(),
                                                       grid_.key(ix + 1, iy + dy));
                            run_lo[dy + 1] = lo - key_.begin();
                            run_hi[dy + 1] = hi - key_.begin();
                        }
                        last_x = ix;
                        last_y = iy;
                    }
                    for (int run = 0; run < 3; ++run) {
                        for (int64_t s = run_lo[run], e = run_hi[run]; s < e; ++s) {
                            const double ddx = x_[s] - x, ddy = y_[s] - y;
                            const double d2 = ddx * ddx + ddy * ddy;
                            if (d2 >= r2 || index_[s] == self) continue;
                            const int64_t b = static_cast<int64_t>(std::sqrt(d2) * inv);
                            if (b < bins) ++h[b];
                        }
                    }
                }
            });
        }
        for (const auto& h : local)
            for (int64_t b = 0; b < bins; ++b) res[b] += static_cast<double>(h[b]);
        return out;
    }

private:
    Grid grid_;
    bool has_z_ = false, has_values_ = false;
    std::vector<int64_t> key_, index_;
    std::vector<double> x_, y_, z_, v_;
};

}  // namespace

PYBIND11_MODULE(_neighbours, m) {
    m.doc() = "Neighbours within a radius and statistics over them (neighbourhood.py)";
    m.def("cell_order", &cell_order, py::arg("x"), py::arg("y"), py::arg("x0"),
          py::arg("y0"), py::arg("x1"), py::arg("reach"), py::arg("threads") = 0);
    py::class_<Index>(m, "Index")
        .def(py::init<const Doubles&, const Doubles&, py::object, py::object, double,
                      double, double, double, int>(),
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("values"), py::arg("x0"),
             py::arg("y0"), py::arg("x1"), py::arg("reach"), py::arg("threads") = 0)
        .def("query", &Index::query, py::arg("x"), py::arg("y"), py::arg("z"),
             py::arg("same"), py::arg("mode"), py::arg("sigma") = 0.0,
             py::arg("z_reach") = 0.0, py::arg("z_sigma") = 0.0, py::arg("q") = 0.5,
             py::arg("threads") = 0)
        .def("pair_histogram", &Index::pair_histogram, py::arg("x"), py::arg("y"),
             py::arg("same"), py::arg("dr"), py::arg("bins"), py::arg("threads") = 0)
        .def("dbscan_join", &Index::dbscan_join, py::arg("core"), py::arg("use_z"),
             py::arg("threads") = 0);
}
