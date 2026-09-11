// Python binding for the grouper's linking core.  See group.hpp.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "combine.hpp"
#include "group.hpp"

namespace py = pybind11;

namespace {

using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Frames = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

py::tuple connect(const Doubles& x, const Doubles& y, const Frames& frame,
                  double dx, int64_t dt, py::object z_obj, double dz) {
    if (x.ndim() != 1 || y.size() != x.size() || frame.size() != x.size())
        throw std::invalid_argument("x, y and frame must be matching 1-D arrays");

    const py::ssize_t n = x.size();
    py::array_t<int64_t> list(n);
    std::fill_n(list.mutable_data(), n, int64_t(0));

    Doubles z;
    const double* zp = nullptr;
    if (!z_obj.is_none()) {
        z = z_obj.cast<Doubles>();
        if (z.size() != n) throw std::invalid_argument("z must match x");
        zp = z.data();
    }
    int64_t groups = 0;
    {
        py::gil_scoped_release release;   // sequential, but long-running
        groups = smappy::connect_single(x.data(), y.data(), frame.data(), n, dx,
                                         dt, list.mutable_data(), zp, dz);
    }
    return py::make_tuple(list, groups);
}

// One column reduced to one value per group.  The column keeps whatever dtype
// it is stored as -- forcing it to float64 first is the 457 MB temporary this
// exists to avoid -- so the dtype is dispatched here rather than by pybind.
py::array_t<double> combine(const py::array& values, py::object weights_obj,
                            const Frames& order, const Frames& starts, int mode,
                            int threads) {
    if (order.ndim() != 1 || starts.ndim() != 1 || starts.size() < 1)
        throw std::invalid_argument("order and starts must be 1-D, starts non-empty");
    const int64_t n_groups = starts.size() - 1;
    if (mode < 0 || mode > 5) throw std::invalid_argument("unknown combine mode");
    const auto kind = static_cast<smappy::Combine>(mode);

    Doubles weights;
    const double* wp = nullptr;
    if (!weights_obj.is_none()) {
        weights = weights_obj.cast<Doubles>();
        if (weights.size() != values.size())
            throw std::invalid_argument("weights must match the column");
        wp = weights.data();
    } else if (kind == smappy::Combine::Weighted) {
        throw std::invalid_argument("a weighted combine needs weights");
    }

    py::array_t<double> out(n_groups);
    const auto* op = order.data();
    const auto* sp = starts.data();
    auto* dst = out.mutable_data();

    py::gil_scoped_release release;
    const auto dt = values.dtype();
    if (dt.is(py::dtype::of<float>()))
        smappy::combine_column(static_cast<const float*>(values.data()), wp, op, sp,
                               n_groups, kind, dst, threads);
    else if (dt.is(py::dtype::of<double>()))
        smappy::combine_column(static_cast<const double*>(values.data()), wp, op, sp,
                               n_groups, kind, dst, threads);
    else if (dt.is(py::dtype::of<int64_t>()))
        smappy::combine_column(static_cast<const int64_t*>(values.data()), wp, op, sp,
                               n_groups, kind, dst, threads);
    else if (dt.is(py::dtype::of<int32_t>()))
        smappy::combine_column(static_cast<const int32_t*>(values.data()), wp, op, sp,
                               n_groups, kind, dst, threads);
    else {
        py::gil_scoped_acquire hold;      // an unusual dtype: let numpy convert
        const Doubles cast = values.cast<Doubles>();
        py::gil_scoped_release again;
        smappy::combine_column(cast.data(), wp, op, sp, n_groups, kind, dst, threads);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_group, m) {
    m.doc() = "Frame-to-frame linking of localizations.";
    m.def("connect", &connect, py::arg("x"), py::arg("y"), py::arg("frame"),
          py::arg("dx"), py::arg("dt"), py::arg("z") = py::none(), py::arg("dz") = 0.0,
          "Link one block sorted by (frame, x), optionally within dz in z; "
          "returns (group_ids, n_groups).");
    m.def("combine", &combine, py::arg("values"), py::arg("weights"), py::arg("order"),
          py::arg("starts"), py::arg("mode"), py::arg("threads") = 0,
          "Reduce one column to one value per group, over data in group order. "
          "mode: 0 sum, 1 weighted sum, 2 sum of 1/v^2, 3 sum of v^2, 4 min, 5 max.");
}
