// Python bindings for the maximum-likelihood single-molecule fitter.
//
// One entry point per PSF model; each takes a stack of ROIs (n, sz, sz)
// float32, C-contiguous, and returns (theta, crlb, logl, iterations).
// ROIs are fitted independently, so the stack is simply split over threads.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "filters.hpp"
#include "lm.hpp"
#include "maxima.hpp"
#include "parallel.hpp"
#include "global.hpp"
#include "models.hpp"

namespace py = pybind11;
using smappy::lm_fit;

namespace {

using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

struct Output {
    Array theta, crlb, logl;
    py::array_t<int> iterations;
};

// Check the ROI stack and return (n, sz).
std::pair<py::ssize_t, int> check_rois(const Array& rois) {
    if (rois.ndim() != 3)
        throw std::invalid_argument("rois must have shape (n, roisize, roisize)");
    if (rois.shape(1) != rois.shape(2))
        throw std::invalid_argument("ROIs must be square");
    return {rois.shape(0), static_cast<int>(rois.shape(1))};
}

template <class Model>
Output run(const Model& model, const Array& rois, int iterations, int n_threads) {
    constexpr int NV = Model::NV;
    // (structured bindings cannot be captured by a lambda in C++17)
    const auto dims = check_rois(rois);
    const py::ssize_t n = dims.first;
    const int sz = dims.second;

    Output out{Array({n, py::ssize_t(NV)}), Array({n, py::ssize_t(NV)}),
               Array(n), py::array_t<int>(n)};

    const float* data = rois.data();
    float* theta = out.theta.mutable_data();
    float* crlb = out.crlb.mutable_data();
    float* logl = out.logl.mutable_data();
    int* iters = out.iterations.mutable_data();

    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
    }
    n_threads = std::min<py::ssize_t>(n_threads, std::max<py::ssize_t>(n, 1));

    auto worker = [&](py::ssize_t begin, py::ssize_t end) {
        // each fit is independent: no shared state, no synchronisation
        Model local = model;  // per-thread copy (the spline keeps scratch space)
        for (py::ssize_t i = begin; i < end; ++i)
            lm_fit(local, data + i * sz * sz, sz, iterations, theta + i * NV,
                   crlb + i * NV, logl + i, iters + i);
    };

    {
        py::gil_scoped_release release;
        if (n_threads == 1) {
            worker(0, n);
        } else {
            std::vector<std::thread> pool;
            const py::ssize_t step = (n + n_threads - 1) / n_threads;
            for (int t = 0; t < n_threads; ++t) {
                const py::ssize_t begin = t * step;
                const py::ssize_t end = std::min(begin + step, n);
                if (begin < end) pool.emplace_back(worker, begin, end);
            }
            for (auto& th : pool) th.join();
        }
    }
    return out;
}

py::tuple as_tuple(Output&& o) {
    return py::make_tuple(std::move(o.theta), std::move(o.crlb),
                          std::move(o.logl), std::move(o.iterations));
}

py::tuple fit_gauss_free(const Array& rois, float sigma, int iterations,
                         int n_threads) {
    return as_tuple(run(smappy::GaussFree(sigma), rois, iterations, n_threads));
}

py::tuple fit_gauss_xy(const Array& rois, float sigma, int iterations,
                       int n_threads) {
    return as_tuple(run(smappy::GaussXY(sigma), rois, iterations, n_threads));
}

py::tuple fit_cspline(const Array& rois, const Array& coeff, float z_start,
                      int iterations, int n_threads) {
    if (coeff.ndim() != 4 || coeff.shape(0) != 64)
        throw std::invalid_argument(
            "spline coefficients must have shape (64, nz, ny, nx)");
    const int nz = static_cast<int>(coeff.shape(1));
    const int ny = static_cast<int>(coeff.shape(2));
    const int nx = static_cast<int>(coeff.shape(3));
    smappy::CSpline model(coeff.data(), nx, ny, nz, z_start);
    return as_tuple(run(model, rois, iterations, n_threads));
}

// Global (multi-channel) fits.  One emitter, one ROI per channel, with the
// parameters the caller marks shared fitted once for all channels.
//
// rois   (n, C, sz, sz)  the same emitter in every channel
// link   (n, 2, C, P)    per fit: the offset, then the factor, of each of the
//                        model's P parameters in each channel
// shared (P,)            which of them are fitted once for all
//
// Returns theta and crlb with NV = sum over the P parameters of (1 if shared
// else C) columns, laid out in parameter order with a free parameter's C
// values contiguous.
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

struct GlobalDims {
    py::ssize_t n;
    int channels, sz;
};

// Check the paired ROI stack.  Cheap and idempotent, so a caller that needs
// the channel count before it can build its models may ask twice.
GlobalDims check_global_rois(const Array& rois) {
    if (rois.ndim() != 4 || rois.shape(2) != rois.shape(3))
        throw std::invalid_argument("rois must have shape (n, channels, sz, sz), square");
    const int channels = static_cast<int>(rois.shape(1));
    if (channels < 1 || channels > smappy::MAX_CHANNELS)
        throw std::invalid_argument("between one and four channels");
    return {rois.shape(0), channels, static_cast<int>(rois.shape(2))};
}

// The body every global fit shares: validate the link, plan the layout of the
// global vector and drive the per-ROI loop.  `models` holds one PSF per
// channel and is copied per thread, as in `run` -- the spline keeps scratch.
template <class Model>
Output run_global(const std::vector<Model>& models, const Array& rois,
                  const Array& link, const IntArray& shared, int iterations,
                  int n_threads) {
    constexpr int P = Model::NV;
    const GlobalDims dims = check_global_rois(rois);
    const py::ssize_t n = dims.n;
    const int channels = dims.channels, sz = dims.sz;

    if (static_cast<int>(models.size()) != channels)
        throw std::invalid_argument("one PSF model per channel");
    if (link.ndim() != 4 || link.shape(0) != n || link.shape(1) != 2 ||
        link.shape(2) != channels || link.shape(3) != P)
        throw std::invalid_argument("link must have shape (n, 2, channels, " +
                                    std::to_string(P) + ")");
    if (shared.ndim() != 1 || shared.shape(0) != P)
        throw std::invalid_argument("shared must have " + std::to_string(P) +
                                    " flags");

    smappy::Link plan{};
    plan.shared = shared.data();
    plan.n_channels = channels;
    plan.plan(P);
    const int nv = plan.nv;

    Output out{Array({n, py::ssize_t(nv)}), Array({n, py::ssize_t(nv)}),
               Array(n), py::array_t<int>(n)};

    const float* data = rois.data();
    const float* links = link.data();
    float* theta = out.theta.mutable_data();
    float* crlb = out.crlb.mutable_data();
    float* logl = out.logl.mutable_data();
    int* iters = out.iterations.mutable_data();

    {
        py::gil_scoped_release release;
        smappy::parallel_ranges(n, n_threads, [&](long long begin, long long end, int) {
            std::vector<Model> local(models);   // the spline keeps scratch
            std::vector<const float*> planes(channels);
            smappy::Link mine = plan;
            for (long long i = begin; i < end; ++i) {
                for (int c = 0; c < channels; ++c)
                    planes[c] = data + ((i * channels + c) * sz) * sz;
                mine.offset = links + i * 2 * channels * P;
                mine.factor = mine.offset + channels * P;
                smappy::global_fit(local.data(), planes.data(), sz, mine, iterations,
                                   theta + i * nv, crlb + i * nv, logl + i, iters + i);
            }
        });
    }
    return out;
}

// A cubic-spline PSF per channel; `coeff` is (C, 64, nz, ny, nx), one plane
// per channel, and every channel's spline must be on the same z grid -- which
// is why z needs no offset in the link.
py::tuple fit_cspline_global(const Array& rois, const Array& coeff, const Array& link,
                             const IntArray& shared, float z_start, int iterations,
                             int n_threads) {
    const int channels = check_global_rois(rois).channels;
    if (coeff.ndim() != 5 || coeff.shape(0) != channels || coeff.shape(1) != 64)
        throw std::invalid_argument(
            "spline coefficients must have shape (channels, 64, nz, ny, nx)");

    const int nz = static_cast<int>(coeff.shape(2));
    const int ny = static_cast<int>(coeff.shape(3));
    const int nx = static_cast<int>(coeff.shape(4));
    const py::ssize_t plane = py::ssize_t(64) * nz * ny * nx;

    std::vector<smappy::CSpline> models;
    models.reserve(channels);
    for (int c = 0; c < channels; ++c)
        models.emplace_back(coeff.data() + c * plane, nx, ny, nz, z_start);
    return as_tuple(run_global(models, rois, link, shared, iterations, n_threads));
}

// The same, with a free-width Gaussian in place of the spline: the 2D
// two-colour fit.  `GaussFree`'s fifth parameter is sigma rather than z, so
// the link keeps its shape and `smappy.dualfit` builds it unchanged -- x and y
// carry the registration, the photon number carries the splitter's ratio, and
// sigma needs no link, being free per channel in a ratiometric experiment
// where the two halves see different wavelengths.  `sigma` is each channel's
// starting width, which for the same reason is per channel too.
py::tuple fit_gauss_global(const Array& rois, const Array& sigma, const Array& link,
                           const IntArray& shared, int iterations, int n_threads) {
    const int channels = check_global_rois(rois).channels;
    if (sigma.ndim() != 1 || sigma.shape(0) != channels)
        throw std::invalid_argument("sigma must have one starting width per channel");

    std::vector<smappy::GaussFree> models;
    models.reserve(channels);
    for (int c = 0; c < channels; ++c)
        models.emplace_back(sigma.data()[c]);
    return as_tuple(run_global(models, rois, link, shared, iterations, n_threads));
}

// Difference-of-Gaussians / Gaussian filtering of a block of images.
Array filter_images(const Array& images, float sigma, float sigma_wide,
                    int radius, bool separable, int n_threads) {
    if (images.ndim() != 3)
        throw std::invalid_argument("images must have shape (n, ny, nx)");
    const py::ssize_t n = images.shape(0);
    const int ny = static_cast<int>(images.shape(1));
    const int nx = static_cast<int>(images.shape(2));

    Array out({n, py::ssize_t(ny), py::ssize_t(nx)});
    const float* in = images.data();
    float* dst = out.mutable_data();

    const bool dog = sigma_wide > 0.0f;
    const std::vector<float> kn = smappy::gauss_kernel(sigma, radius);
    const std::vector<float> kw =
        dog ? smappy::gauss_kernel(sigma_wide, radius) : kn;

    std::vector<float> k2d;
    if (!separable) k2d = smappy::dog_kernel_2d(sigma, sigma_wide, radius);

    {
        py::gil_scoped_release release;
        smappy::parallel_ranges(n, n_threads, [&](long long begin, long long end, int) {
            // scratch space is per thread, so the passes never share memory
            std::vector<float> tmp(static_cast<size_t>(ny) * nx * (dog ? 2 : 1));
            for (long long f = begin; f < end; ++f) {
                const float* src = in + f * ny * nx;
                float* d = dst + f * ny * nx;
                if (!separable)
                    smappy::dog2d_frame(src, d, ny, nx, k2d.data(), radius);
                else if (dog)
                    smappy::dog_frame(src, d, ny, nx, kn.data(), kw.data(),
                                       radius, tmp.data());
                else
                    smappy::gauss_frame(src, d, ny, nx, kn.data(), radius,
                                         tmp.data());
            }
        });
    }
    return out;
}

// Strict local maxima of a block of images (n, ny, nx).
// Returns (frame, y, x, value) arrays.  `threshold` pre-filters pixels; pass
// -inf to get every maximum (needed by the dynamic cutoff, which derives its
// threshold from the distribution of all maxima).
py::tuple find_maxima(const Array& images, float threshold, int n_threads) {
    if (images.ndim() != 3)
        throw std::invalid_argument("images must have shape (n, ny, nx)");
    const py::ssize_t n = images.shape(0);
    const int ny = static_cast<int>(images.shape(1));
    const int nx = static_cast<int>(images.shape(2));
    const float* data = images.data();

    const int threads = smappy::resolve_threads(n_threads, n);
    std::vector<std::vector<smappy::Maximum>> per_thread(threads);
    {
        py::gil_scoped_release release;
        smappy::parallel_ranges(n, threads, [&](long long begin, long long end, int t) {
            auto& found = per_thread[t];
            found.reserve(static_cast<size_t>(end - begin) * 64);
            for (long long f = begin; f < end; ++f)
                smappy::find_maxima_frame(data + f * ny * nx, ny, nx,
                                           static_cast<int>(f), threshold, found);
        });
    }

    // concatenating in thread order keeps the maxima sorted by frame, which the
    // per-frame cutoff relies on
    size_t total = 0;
    for (const auto& v : per_thread) total += v.size();
    std::vector<smappy::Maximum> found;
    found.reserve(total);
    for (auto& v : per_thread)
        found.insert(found.end(), v.begin(), v.end());

    const py::ssize_t count = static_cast<py::ssize_t>(found.size());
    py::array_t<int32_t> frame(count), y(count), x(count);
    py::array_t<float> value(count);
    int32_t* pf = frame.mutable_data();
    int32_t* py_ = y.mutable_data();
    int32_t* px = x.mutable_data();
    float* pv = value.mutable_data();
    for (py::ssize_t i = 0; i < count; ++i) {
        pf[i] = found[i].frame;
        py_[i] = found[i].y;
        px[i] = found[i].x;
        pv[i] = found[i].value;
    }
    return py::make_tuple(std::move(frame), std::move(y), std::move(x),
                          std::move(value));
}

}  // namespace

PYBIND11_MODULE(_fit3d, m) {
    m.doc() = "Maximum-likelihood single-molecule fitting (Poisson noise)";

    m.def("fit_gauss_free", &fit_gauss_free, py::arg("rois"),
          py::arg("sigma") = 1.0f, py::arg("iterations") = 50,
          py::arg("n_threads") = 0,
          "Fit (x, y, photons, background, sigma).");

    m.def("fit_gauss_xy", &fit_gauss_xy, py::arg("rois"), py::arg("sigma") = 1.0f,
          py::arg("iterations") = 50, py::arg("n_threads") = 0,
          "Fit (x, y, photons, background, sigma_x, sigma_y).");

    m.def("filter_images", &filter_images, py::arg("images"), py::arg("sigma"),
          py::arg("sigma_wide") = 0.0f, py::arg("radius") = 4,
          py::arg("separable") = true, py::arg("n_threads") = 0,
          "Gaussian (sigma_wide=0) or difference-of-Gaussians filtering.");

    m.def("find_maxima", &find_maxima, py::arg("images"),
          py::arg("threshold") = -std::numeric_limits<float>::infinity(),
          py::arg("n_threads") = 0,
          "Strict 3x3 local maxima of an image block -> (frame, y, x, value).");

    m.def("fit_cspline", &fit_cspline, py::arg("rois"), py::arg("coeff"),
          py::arg("z_start"), py::arg("iterations") = 50, py::arg("n_threads") = 0,
          "Fit (x, y, photons, background, z) with a cubic-spline PSF.");

    m.def("fit_cspline_global", &fit_cspline_global, py::arg("rois"),
          py::arg("coeff"), py::arg("link"), py::arg("shared"), py::arg("z_start"),
          py::arg("iterations") = 50, py::arg("n_threads") = 0,
          "Fit one emitter across several channels with a spline PSF each, "
          "sharing the parameters marked in `shared`.");

    m.def("fit_gauss_global", &fit_gauss_global, py::arg("rois"), py::arg("sigma"),
          py::arg("link"), py::arg("shared"), py::arg("iterations") = 50,
          py::arg("n_threads") = 0,
          "Fit one emitter across several channels with a free-width Gaussian "
          "each -- (x, y, photons, background, sigma) -- sharing the parameters "
          "marked in `shared`.");
}
