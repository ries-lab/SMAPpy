// Python bindings for the superresolution renderer.
//
// Separate from _fit3d on purpose: rendering shares no code with fitting, and a
// viewer should not have to carry the fitter.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"
#include "render.hpp"

namespace py = pybind11;

namespace {

using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

// A per-localization array, or a single value broadcast over all of them.
// Returns the stride to use (1 or 0).
long long broadcast_stride(const Array& a, py::ssize_t n, const char* name) {
    if (a.size() == 1) return 0;
    if (a.size() != n)
        throw std::invalid_argument(std::string(name) +
                                    " must have one value per localization, or one in total");
    return 1;
}

py::tuple render(const Array& x, const Array& y, const Array& sigma_x,
                 const Array& sigma_y, const Array& weight, const Array& color,
                 float x0, float y0, float pixelsize, int nx, int ny,
                 bool gaussian, float roiks, int max_halfwidth, int n_threads) {
    if (x.ndim() != 1 || y.ndim() != 1 || x.size() != y.size())
        throw std::invalid_argument("x and y must be matching 1-D arrays");
    if (nx <= 0 || ny <= 0)
        throw std::invalid_argument("the image must have a positive size");
    if (!(pixelsize > 0))
        throw std::invalid_argument("pixelsize must be positive");

    const py::ssize_t n = x.size();
    const bool colored = color.size() != 0;
    if (colored && (color.ndim() != 2 || color.shape(0) != n || color.shape(1) != 3))
        throw std::invalid_argument("color must have shape (n, 3)");

    Array weight_image({py::ssize_t(ny), py::ssize_t(nx)});
    Array color_image(colored ? std::vector<py::ssize_t>{ny, nx, 3}
                              : std::vector<py::ssize_t>{0, 0, 0});
    std::fill_n(weight_image.mutable_data(), static_cast<size_t>(nx) * ny, 0.0f);
    if (colored)
        std::fill_n(color_image.mutable_data(), static_cast<size_t>(nx) * ny * 3, 0.0f);

    smappy::RenderInput in{x.data(), y.data(), sigma_x.data(), sigma_y.data(),
                            weight.data(), colored ? color.data() : nullptr, n,
                            gaussian ? broadcast_stride(sigma_x, n, "sigma") : 0,
                            broadcast_stride(weight, n, "weight")};
    if (gaussian && broadcast_stride(sigma_y, n, "sigma") != in.sigma_stride)
        throw std::invalid_argument("sigma_x and sigma_y must broadcast alike");

    smappy::RenderTarget out{weight_image.mutable_data(),
                              colored ? color_image.mutable_data() : nullptr,
                              nx, ny, x0, y0, 1.0f / pixelsize};

    // One contiguous band of rows per thread; a localization contributes to
    // whichever bands its kernel reaches, and each band writes only its own
    // rows.  With no shared pixels there is nothing to synchronise.
    const int threads = smappy::resolve_threads(n_threads, ny);
    std::vector<long long> counted(threads, 0);
    {
        py::gil_scoped_release release;
        const int band = (ny + threads - 1) / threads;
        smappy::parallel_ranges(threads, threads, [&](long long b0, long long b1, int) {
            for (long long b = b0; b < b1; ++b) {
                const int begin = static_cast<int>(b) * band;
                const int end = std::min(begin + band, ny);
                if (begin >= end) continue;
                counted[b] = gaussian
                    ? smappy::render_gauss_rows(in, out, roiks, max_halfwidth,
                                                 begin, end)
                    : smappy::render_hist_rows(in, out, begin, end);
            }
        });
    }

    long long total = 0;
    for (long long c : counted) total += c;
    return py::make_tuple(weight_image, colored ? py::object(color_image) : py::none(),
                          total);
}

// Line segments, antialiased, for the tracks render mode (`render.py`,
// `render_tracks`).  A pixel takes from a segment its coverage -- how much of
// a line `width` pixels wide passes over it, approximated by the distance of
// its centre from the segment: 1 within width / 2 - 0.5, falling linearly to
// 0 at width / 2 + 0.5 -- times the segment's weight (and colour).  Ends are
// round.  Consecutive segments of a track overlap at their shared point, so
// a joint is up to twice as bright over a pixel or so; at a width of a pixel
// or two that is not seen, and taking the larger rather than the sum would
// need to know which segments are one track's.
//
// The work per segment is its length times its width, not its bounding box:
// each row the segment can reach is visited, and in it only the columns
// between where the segment enters and leaves that row's band.  Rows are
// split into bands per thread as `render` does, so nothing is shared.
py::tuple render_segments(const Array& xa, const Array& ya, const Array& xb,
                          const Array& yb, const Array& weight, const Array& color,
                          float x0, float y0, float pixelsize, int nx, int ny,
                          float width, int n_threads) {
    const py::ssize_t n = xa.size();
    if (ya.size() != n || xb.size() != n || yb.size() != n)
        throw std::invalid_argument("the segments' ends must match");
    if (nx <= 0 || ny <= 0)
        throw std::invalid_argument("the image must have a positive size");
    if (!(pixelsize > 0)) throw std::invalid_argument("pixelsize must be positive");
    const bool colored = color.size() != 0;
    if (colored && (color.ndim() != 2 || color.shape(0) != n || color.shape(1) != 3))
        throw std::invalid_argument("color must have shape (n, 3)");
    const long long wstride = broadcast_stride(weight, n, "weight");

    Array weight_image({py::ssize_t(ny), py::ssize_t(nx)});
    Array color_image(colored ? std::vector<py::ssize_t>{ny, nx, 3}
                              : std::vector<py::ssize_t>{0, 0, 0});
    float* W = weight_image.mutable_data();
    float* C = colored ? color_image.mutable_data() : nullptr;
    std::fill_n(W, static_cast<size_t>(nx) * ny, 0.0f);
    if (colored) std::fill_n(C, static_cast<size_t>(nx) * ny * 3, 0.0f);
    const float* pxa = xa.data();
    const float* pya = ya.data();
    const float* pxb = xb.data();
    const float* pyb = yb.data();
    const float* pw = weight.data();
    const float* pc = colored ? color.data() : nullptr;
    const float inv = 1.0f / pixelsize;
    const double half = std::max(0.5 * width, 0.0);
    const double reach = half + 0.5;          // beyond this a pixel gets nothing

    const int threads = smappy::resolve_threads(n_threads, ny);
    {
        py::gil_scoped_release release;
        const int band = (ny + threads - 1) / threads;
        smappy::parallel_ranges(threads, threads, [&](long long b0, long long b1, int) {
            for (long long b = b0; b < b1; ++b) {
                const int rbegin = static_cast<int>(b) * band;
                const int rend = std::min(rbegin + band, ny);
                if (rbegin >= rend) continue;
                for (py::ssize_t i = 0; i < n; ++i) {
                    // pixel coordinates: pixel (c, r) spans [c, c + 1)
                    const double ax = (pxa[i] - x0) * inv, ay = (pya[i] - y0) * inv;
                    const double bx = (pxb[i] - x0) * inv, by = (pyb[i] - y0) * inv;
                    if (!std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(bx) ||
                        !std::isfinite(by)) continue;
                    const int r0 = std::max(rbegin, static_cast<int>(
                        std::floor(std::min(ay, by) - reach)));
                    const int r1 = std::min(rend - 1, static_cast<int>(
                        std::floor(std::max(ay, by) + reach)));
                    if (r0 > r1) continue;
                    const double dx = bx - ax, dy = by - ay;
                    const double len2 = dx * dx + dy * dy;
                    const float w = pw[wstride * i];
                    for (int r = r0; r <= r1; ++r) {
                        const double yc = r + 0.5;
                        // the part of the segment within reach of this row
                        double t0 = 0.0, t1 = 1.0;
                        if (std::abs(dy) > 1e-12) {
                            double ta = (yc - reach - 0.5 - ay) / dy;
                            double tb = (yc + reach + 0.5 - ay) / dy;
                            if (ta > tb) std::swap(ta, tb);
                            t0 = std::max(0.0, ta);
                            t1 = std::min(1.0, tb);
                            if (t0 > t1) continue;
                        } else if (std::abs(ay - yc) > reach + 0.5) {
                            continue;
                        }
                        const double xa_ = ax + t0 * dx, xb_ = ax + t1 * dx;
                        const int c0 = std::max(0, static_cast<int>(
                            std::floor(std::min(xa_, xb_) - reach)));
                        const int c1 = std::min(nx - 1, static_cast<int>(
                            std::floor(std::max(xa_, xb_) + reach)));
                        for (int c = c0; c <= c1; ++c) {
                            const double xc = c + 0.5;
                            double t = len2 > 0 ? ((xc - ax) * dx + (yc - ay) * dy) / len2 : 0.0;
                            t = std::min(1.0, std::max(0.0, t));
                            const double ex = ax + t * dx - xc, ey = ay + t * dy - yc;
                            const double d = std::sqrt(ex * ex + ey * ey);
                            const double cover = std::min(1.0, std::max(0.0, reach - d));
                            if (cover <= 0) continue;
                            const float v = static_cast<float>(cover) * w;
                            const size_t k = static_cast<size_t>(r) * nx + c;
                            W[k] += v;
                            if (C) {
                                C[3 * k] += v * pc[3 * i];
                                C[3 * k + 1] += v * pc[3 * i + 1];
                                C[3 * k + 2] += v * pc[3 * i + 2];
                            }
                        }
                    }
                }
            }
        });
    }
    return py::make_tuple(weight_image, colored ? py::object(color_image) : py::none());
}

}  // namespace

PYBIND11_MODULE(_render, m) {
    m.doc() = "Accumulate localizations into a superresolution image.";
    m.def("render", &render, py::arg("x"), py::arg("y"), py::arg("sigma_x"),
          py::arg("sigma_y"), py::arg("weight"), py::arg("color"), py::arg("x0"),
          py::arg("y0"), py::arg("pixelsize"), py::arg("nx"), py::arg("ny"),
          py::arg("gaussian"), py::arg("roiks") = 2.7f,
          py::arg("max_halfwidth") = 512, py::arg("n_threads") = 0,
          "Render localizations; returns (weight, color or None, n_localizations).");
    m.def("render_segments", &render_segments, py::arg("xa"), py::arg("ya"), py::arg("xb"),
          py::arg("yb"), py::arg("weight"), py::arg("color"), py::arg("x0"), py::arg("y0"),
          py::arg("pixelsize"), py::arg("nx"), py::arg("ny"), py::arg("width") = 1.0f,
          py::arg("n_threads") = 0,
          "Draw line segments, antialiased; returns (weight, color or None).");
}
