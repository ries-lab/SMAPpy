// Python binding for the drift cost function.  See drift.hpp.
//
// Each kernel is registered four times, one per (coordinate, index) dtype the
// optimizer might hand over.  That is not gold-plating: the optimizer builds
// float32 coordinates and int32 indices once and then calls this a few hundred
// times, so a binding that accepted only float64/int64 would convert -- and at
// three hundred million pairs that copy is gigabytes per evaluation.  pybind11
// tries overloads without conversion first, so the exact match wins and nothing
// is copied; anything else falls through to the last overload, which converts.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "drift.hpp"

namespace py = pybind11;

namespace {

template <class T>
using Array = py::array_t<T, py::array::c_style>;
using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Indices = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

template <class Coord, class Index>
void check(const Array<Coord>& coords, const Array<Index>& times,
           const Array<Index>& idx_i, const Array<Index>& idx_j,
           const Doubles& mu) {
    if (coords.ndim() != 2 || coords.shape(1) != 3)
        throw std::invalid_argument("coords must be (n, 3)");
    if (mu.ndim() != 2 || mu.shape(1) != 3)
        throw std::invalid_argument("mu must be (n_segments, 3)");
    if (times.ndim() != 1 || times.shape(0) != coords.shape(0))
        throw std::invalid_argument("times must be (n,), one per localization");
    if (idx_i.ndim() != 1 || idx_j.ndim() != 1 || idx_i.size() != idx_j.size())
        throw std::invalid_argument("idx_i and idx_j must be matching 1-D arrays");
}

template <class Coord, class Index>
py::tuple cost_and_gradient(const Array<Coord>& coords, const Array<Index>& times,
                            const Array<Index>& idx_i, const Array<Index>& idx_j,
                            const Doubles& mu, double sigma, double sigma_factor,
                            double cutoff_sigmas, int n_threads, double z_factor,
                            const py::object& variances) {
    check<Coord, Index>(coords, times, idx_i, idx_j, mu);
    const int64_t n_segments = mu.shape(0);
    const int64_t n_pairs = idx_i.size();
    if (!(z_factor > 0.0)) throw std::invalid_argument("z_factor must be positive");
    // per-localization variances, (n, 3), or none (`drift.hpp`)
    Doubles var;
    const double* var_data = nullptr;
    if (!variances.is_none()) {
        var = Doubles::ensure(variances);
        if (!var || var.ndim() != 2 || var.shape(0) != coords.shape(0) ||
            var.shape(1) != 3)
            throw std::invalid_argument("variances must be (n, 3), one row per localization");
        var_data = var.data();
    }

    smappy::DriftCost result;
    {
        py::gil_scoped_release release;
        result = smappy::cost_and_gradient<Coord, Index>(
            coords.data(), times.data(), idx_i.data(), idx_j.data(), mu.data(),
            n_pairs, n_segments, sigma, sigma_factor, cutoff_sigmas, n_threads,
            z_factor, var_data);
    }

    py::array_t<double> gradient(std::vector<py::ssize_t>{n_segments, 3});
    std::copy(result.gradient.begin(), result.gradient.end(), gradient.mutable_data());
    return py::make_tuple(result.total, gradient);
}

template <class Coord, class Index>
py::tuple overlap_per_segment(const Array<Coord>& coords, const Array<Index>& times,
                              const Array<Index>& idx_i, const Array<Index>& idx_j,
                              const Doubles& mu, double sigma, double sigma_factor,
                              int n_threads, double z_factor) {
    check<Coord, Index>(coords, times, idx_i, idx_j, mu);
    const int64_t n_segments = mu.shape(0);
    const int64_t n_pairs = idx_i.size();

    smappy::SegmentOverlap out;
    {
        py::gil_scoped_release release;
        out = smappy::overlap_per_segment<Coord, Index>(
            coords.data(), times.data(), idx_i.data(), idx_j.data(), mu.data(),
            n_pairs, n_segments, sigma, sigma_factor, n_threads, z_factor);
    }

    auto to_array = [n_segments](const std::vector<double>& v) {
        py::array_t<double> a(n_segments);
        std::copy(v.begin(), v.end(), a.mutable_data());
        return a;
    };
    return py::make_tuple(to_array(out.observed), to_array(out.null),
                          to_array(out.counts));
}

template <class Coord, class Index>
py::tuple transformed_cost_and_gradient(
        const Array<Coord>& coords, const Array<Index>& times,
        const Array<Index>& idx_i, const Array<Index>& idx_j,
        const Doubles& params, double sigma, double sigma_factor,
        double cutoff_sigmas, int n_threads, double z_factor) {
    if (coords.ndim() != 2 || coords.shape(1) != 3)
        throw std::invalid_argument("coords must be (n, 3)");
    if (params.ndim() != 2 || params.shape(1) != 14)
        throw std::invalid_argument("params must be (n_frames, 14)");
    if (times.ndim() != 1 || times.shape(0) != coords.shape(0))
        throw std::invalid_argument("times must be (n,), one per localization");
    if (idx_i.ndim() != 1 || idx_j.ndim() != 1 || idx_i.size() != idx_j.size())
        throw std::invalid_argument("idx_i and idx_j must be matching 1-D arrays");
    const int64_t n_frames = params.shape(0);
    const int64_t n_pairs = idx_i.size();

    smappy::TransformedCost result;
    {
        py::gil_scoped_release release;
        result = smappy::transformed_cost_and_gradient<Coord, Index>(
            coords.data(), times.data(), idx_i.data(), idx_j.data(), params.data(),
            n_pairs, n_frames, sigma, sigma_factor, cutoff_sigmas,
            n_threads, z_factor);
    }
    py::array_t<double> gradient(std::vector<py::ssize_t>{n_frames, 14});
    std::copy(result.gradient.begin(), result.gradient.end(), gradient.mutable_data());
    return py::make_tuple(result.total, gradient);
}

template <class Coord, class Index>
void bind(py::module_& m) {
    m.def("cost_and_gradient", &cost_and_gradient<Coord, Index>, py::arg("coords"),
          py::arg("times"), py::arg("idx_i"), py::arg("idx_j"), py::arg("mu"),
          py::arg("sigma"), py::arg("sigma_factor"),
          py::arg("cutoff_sigmas") = 0.0, py::arg("n_threads") = 0,
          py::arg("z_factor") = 1.0, py::arg("variances") = py::none(),
          "Pair overlap and its gradient; cutoff_sigmas <= 0 is the exact kernel, "
          "z_factor is sigma_z / sigma_xy, variances (n, 3) each localization's.");
    m.def("overlap_per_segment", &overlap_per_segment<Coord, Index>,
          py::arg("coords"), py::arg("times"), py::arg("idx_i"), py::arg("idx_j"),
          py::arg("mu"), py::arg("sigma"), py::arg("sigma_factor"),
          py::arg("n_threads") = 0, py::arg("z_factor") = 1.0,
          "(observed, null, counts) per time window, for quality control.");
    m.def("transformed_cost_and_gradient",
          &transformed_cost_and_gradient<Coord, Index>, py::arg("coords"),
          py::arg("times"), py::arg("idx_i"), py::arg("idx_j"), py::arg("params"),
          py::arg("sigma"), py::arg("sigma_factor"),
          py::arg("cutoff_sigmas") = 0.0, py::arg("n_threads") = 0,
          py::arg("z_factor") = 1.0,
          "Pair overlap and its gradient with a linear transformation per frame.");
}

}  // namespace

PYBIND11_MODULE(_drift, m) {
    m.doc() = "COMET's drift cost function, compiled.";
    // the dtypes the optimizer actually produces come first
    bind<float, int32_t>(m);
    bind<float, int64_t>(m);
    bind<double, int32_t>(m);
    bind<double, int64_t>(m);
}
