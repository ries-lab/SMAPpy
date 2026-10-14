// What is around each localization: the neighbours within a radius (a circle,
// a cylinder, or Gaussian-weighted) and a statistic of a column over them.
// The Python side is `smappy.plugins.neighbourhood`.
//
// SMAP's countneighbours*.c and SpatialFilterLocs sort the points by x and,
// for each point, test every point of the strip [x - r, x + r]: no pairs are
// ever stored, but a strip is as tall as the field, so a point of a 40 um
// field at 5 M localizations tests ten thousand others to find 25.  Here the
// neighbours are sorted by a square cell one search radius wide, row after
// row, so a point's candidates are three runs of the sorted list -- the rows
// above, at and below it, three cells each -- found by binary search: about
// 9 / pi times the neighbours there are, whatever the size of the field, and
// no grid is allocated however sparse the points.  The points asked about are
// visited in the same cell order, so neighbouring queries read the same
// memory, and split over threads.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"

namespace py = pybind11;

namespace {

using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Ints = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

enum Mode { Count = 0, Gauss = 1, Mean = 2, Median = 3, Min = 4, Max = 5, Std = 6,
            Quantile = 7 };

struct Grid {
    double x0, y0, cell;
    int64_t width;
    int64_t cx(double x) const { return static_cast<int64_t>(std::floor((x - x0) / cell)); }
    int64_t cy(double y) const { return static_cast<int64_t>(std::floor((y - y0) / cell)); }
    int64_t key(int64_t ix, int64_t iy) const { return iy * width + ix; }
};

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

// mode, reach (lateral search radius), sigma (Gaussian), z_reach / z_sigma
// (<= 0: no z), q (quantile).  `same[i]` is the neighbour that is around point
// i itself, or -1; it is skipped when counting and kept for a statistic.
py::array_t<double> around(const Doubles& ax, const Doubles& ay, py::object az_obj,
                           const Doubles& bx, const Doubles& by, py::object bz_obj,
                           py::object values_obj, py::object same_obj, int mode,
                           double reach, double sigma, double z_reach, double z_sigma,
                           double q, int threads) {
    const int64_t na = ax.size(), nb = bx.size();
    if (ay.size() != na || by.size() != nb)
        throw std::invalid_argument("x and y must match");
    if (!(reach > 0)) throw std::invalid_argument("the radius must be positive");
    const bool use_z = z_reach > 0 && !az_obj.is_none() && !bz_obj.is_none();
    Doubles az, bz, values;
    Ints same;
    if (use_z) {
        az = az_obj.cast<Doubles>();
        bz = bz_obj.cast<Doubles>();
        if (az.size() != na || bz.size() != nb) throw std::invalid_argument("z must match");
    }
    const bool stat = mode >= Mean;
    if (stat) {
        if (values_obj.is_none()) throw std::invalid_argument("a statistic needs values");
        values = values_obj.cast<Doubles>();
        if (values.size() != nb) throw std::invalid_argument("values must match");
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
    if (na == 0 || nb == 0) return out;

    const double* pax = ax.data(); const double* pay = ay.data();
    const double* pbx = bx.data(); const double* pby = by.data();
    const double* paz = use_z ? az.data() : nullptr;
    const double* pbz = use_z ? bz.data() : nullptr;
    const double* pv = stat ? values.data() : nullptr;
    const int64_t* ps = have_same ? same.data() : nullptr;

    py::gil_scoped_release release;
    double x0 = pbx[0], x1 = pbx[0], y0 = pby[0];
    for (int64_t j = 0; j < nb; ++j) {
        x0 = std::min(x0, pbx[j]); x1 = std::max(x1, pbx[j]);
        y0 = std::min(y0, pby[j]);
    }
    for (int64_t i = 0; i < na; ++i) {
        x0 = std::min(x0, pax[i]); x1 = std::max(x1, pax[i]);
        y0 = std::min(y0, pay[i]);
    }
    Grid grid{x0 - reach, y0 - reach, reach,
              static_cast<int64_t>(std::floor((x1 - x0) / reach)) + 4};

    // the neighbours in cell order
    std::vector<int64_t> bkey(nb), border(nb);
    for (int64_t j = 0; j < nb; ++j) bkey[j] = grid.key(grid.cx(pbx[j]), grid.cy(pby[j]));
    std::iota(border.begin(), border.end(), 0);
    std::sort(border.begin(), border.end(),
              [&](int64_t a, int64_t b) { return bkey[a] < bkey[b]; });
    std::vector<int64_t> skey(nb);
    std::vector<double> sx(nb), sy(nb), sz(use_z ? nb : 0), sv(stat ? nb : 0);
    for (int64_t k = 0; k < nb; ++k) {
        const int64_t j = border[k];
        skey[k] = bkey[j]; sx[k] = pbx[j]; sy[k] = pby[j];
        if (use_z) sz[k] = pbz[j];
        if (stat) sv[k] = pv[j];
    }
    // the points asked about, in the same order
    std::vector<int64_t> akey(na), aorder(na);
    for (int64_t i = 0; i < na; ++i) akey[i] = grid.key(grid.cx(pax[i]), grid.cy(pay[i]));
    std::iota(aorder.begin(), aorder.end(), 0);
    std::sort(aorder.begin(), aorder.end(),
              [&](int64_t a, int64_t b) { return akey[a] < akey[b]; });

    const double r2 = reach * reach, z2 = z_reach * z_reach;
    const double gxy = sigma > 0 ? 1.0 / (2 * sigma * sigma) : 0.0;
    const double gz = z_sigma > 0 ? 1.0 / (2 * z_sigma * z_sigma) : 0.0;

    smappy::parallel_ranges(na, threads, [&](long long begin, long long end, int) {
        std::vector<double> found;
        for (long long k = begin; k < end; ++k) {
            const int64_t i = aorder[k];
            const double x = pax[i], y = pay[i], z = use_z ? paz[i] : 0.0;
            const int64_t self = ps ? ps[i] : -1;
            const int64_t ix = grid.cx(x), iy = grid.cy(y);
            double total = 0.0;
            found.clear();
            for (int64_t dy = -1; dy <= 1; ++dy) {
                const int64_t lo_key = grid.key(ix - 1, iy + dy);
                const int64_t hi_key = grid.key(ix + 1, iy + dy);
                auto lo = std::lower_bound(skey.begin(), skey.end(), lo_key);
                auto hi = std::upper_bound(lo, skey.end(), hi_key);
                for (int64_t s = lo - skey.begin(), e = hi - skey.begin(); s < e; ++s) {
                    const double ddx = sx[s] - x, ddy = sy[s] - y;
                    const double d2 = ddx * ddx + ddy * ddy;
                    if (d2 > r2) continue;
                    double dz2 = 0.0;
                    if (use_z) {
                        const double ddz = sz[s] - z;
                        dz2 = ddz * ddz;
                        if (dz2 > z2) continue;
                    }
                    if (stat) {
                        const double v = sv[s];
                        if (std::isfinite(v)) found.push_back(v);
                        continue;
                    }
                    if (border[s] == self) continue;           // not its own neighbour
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

}  // namespace

PYBIND11_MODULE(_neighbours, m) {
    m.doc() = "Neighbours within a radius and statistics over them (neighbourhood.py)";
    m.def("around", &around, py::arg("ax"), py::arg("ay"), py::arg("az"), py::arg("bx"),
          py::arg("by"), py::arg("bz"), py::arg("values"), py::arg("same"),
          py::arg("mode"), py::arg("reach"), py::arg("sigma") = 0.0,
          py::arg("z_reach") = 0.0, py::arg("z_sigma") = 0.0, py::arg("q") = 0.5,
          py::arg("threads") = 0);
}
