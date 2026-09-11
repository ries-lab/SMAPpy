// Global (multi-channel) maximum-likelihood fit: one emitter seen in several
// channels at once, some of its parameters shared between them.
//
// Ported from GlobLoc (Li et al., Nat. Commun. 13, 3133 (2022); Ries lab /
// Li lab, `CPUmleFit_LM_MultiChannel`).  The algorithm is the same
// Levenberg-Marquardt loop as `lm.hpp` -- same damping, same step limiting,
// same convergence test -- over a *linked* parameter vector, so the two stay
// side by side rather than one being a special case of the other.
//
// The link.  Each channel c has the model's own parameters `theta_c[p]`;
// they are written in terms of a global vector as
//
//     shared p:  theta_c[p] = global[slot] * factor[c][p] + offset[c][p]
//     free   p:  theta_c[p] = global[slot + c]            + offset[c][p]
//
// so a shared parameter costs one entry and a free one costs C.  GlobLoc uses
// exactly this form: the offsets carry the registration between the channels
// (the sub-pixel shift left over after the ROIs were cut, and any z offset)
// and the factors carry a scale (the photon ratio of a ratiometric splitter,
// or the local magnification between two views).  It is a *diagonal* link:
// there is no cross term, so a rotation between channels is not represented
// exactly and is left to the fit as a small residual, as in GlobLoc.
//
// The global vector is laid out in parameter order, a free parameter's C
// values contiguous -- GlobLoc's `newThetaAll`.
#pragma once

#include <cmath>
#include <cstring>

#include "linalg.hpp"
#include "lm.hpp"

namespace smappy {

constexpr int MAX_CHANNELS = 4;
// Nothing shared costs one entry per parameter per channel; `linalg.hpp`'s
// solvers are sized to match.
constexpr int MAX_GLOBAL_NV = MAX_NV;
static_assert(MAX_GLOBAL_NV >= 8 * MAX_CHANNELS, "the solvers must cover the widest link");

// Where parameter `p` of channel `c` lives in the global vector.
struct Link {
    const int* shared;      // P flags: 1 = one value for every channel
    const float* offset;    // (C, P)
    const float* factor;    // (C, P), used by shared parameters only
    int n_channels;
    int slots[MAX_GLOBAL_NV];   // slot of parameter p (index p), see below
    int nv;

    // `slots[p]` is where parameter p starts; a free parameter occupies
    // slots[p] .. slots[p] + C - 1.
    void plan(int n_params) {
        nv = 0;
        for (int p = 0; p < n_params; ++p) {
            slots[p] = nv;
            nv += shared[p] ? 1 : n_channels;
        }
    }

    int slot(int p, int c) const { return shared[p] ? slots[p] : slots[p] + c; }
};

// global -> per-channel parameters.
template <int P>
inline void expand(const Link& link, const float* global, float* theta) {
    for (int c = 0; c < link.n_channels; ++c)
        for (int p = 0; p < P; ++p)
            theta[c * P + p] = link.shared[p]
                ? global[link.slots[p]] * link.factor[c * P + p] + link.offset[c * P + p]
                : global[link.slots[p] + c] + link.offset[c * P + p];
}

// The derivative of the global vector's entry `l` on channel `c`:
// d theta_c[p] / d global[l], folded into the chain rule below.
template <class Model>
inline void chain(const Link& link, const float* dudt, int channel, int nv,
                  float* dudt_global) {
    constexpr int P = Model::NV;
    std::memset(dudt_global, 0, nv * sizeof(float));
    for (int p = 0; p < P; ++p) {
        if (link.shared[p])
            dudt_global[link.slots[p]] += dudt[p] * link.factor[channel * P + p];
        else
            dudt_global[link.slots[p] + channel] += dudt[p];
    }
}

// Poisson log-likelihood error, gradient and Hessian summed over every pixel
// of every channel -- `lm.hpp`'s `accumulate` with the chain rule of the link.
//
// The pixel loop is outside and the channel loop inside, as in GlobLoc: every
// channel contributes to the same pixel's term before the next pixel, so the
// sums are accumulated in the reference's order and not merely its value.
// `dudt_global` is GlobLoc's `newDudtAll`, (C, nv).
template <class Model>
inline void accumulate_global(const Model* models, const float* const* data, int sz,
                              const Link& link, const float* theta, float* err,
                              float* jacobian, float* hessian) {
    constexpr int P = Model::NV;
    const int nv = link.nv, C = link.n_channels;
    float dudt[P], dudt_global[MAX_CHANNELS * MAX_GLOBAL_NV];
    float mu[MAX_CHANNELS], d[MAX_CHANNELS];

    *err = 0.0f;
    std::memset(jacobian, 0, nv * sizeof(float));
    std::memset(hessian, 0, nv * nv * sizeof(float));

    for (int c = 0; c < C; ++c) models[c].prepare(theta + c * P, sz);

    for (int ix = 0; ix < sz; ++ix)
        for (int iy = 0; iy < sz; ++iy) {
            for (int c = 0; c < C; ++c) {
                models[c].value(ix, iy, theta + c * P, dudt, &mu[c]);
                chain<Model>(link, dudt, c, nv, dudt_global + c * nv);
                d[c] = data[c][iy * sz + ix];

                if (d[c] > 0.0f)
                    *err += 2.0f * ((mu[c] - d[c]) - d[c] * std::log(mu[c] / d[c]));
                else {
                    *err += 2.0f * mu[c];
                    d[c] = 0.0f;
                }
            }

            for (int l = 0; l < nv; ++l)
                for (int c = 0; c < C; ++c)
                    jacobian[l] += (1.0f - d[c] / mu[c]) * dudt_global[c * nv + l];

            for (int l = 0; l < nv; ++l)
                for (int m = l; m < nv; ++m) {
                    for (int c = 0; c < C; ++c)
                        hessian[l * nv + m] += d[c] / (mu[c] * mu[c]) *
                            dudt_global[c * nv + l] * dudt_global[c * nv + m];
                    hessian[m * nv + l] = hessian[l * nv + m];
                }
        }
}

template <class Model>
inline void crlb_and_logl_global(const Model* models, const float* const* data, int sz,
                                 const Link& link, const float* theta, float* crlb,
                                 float* logl) {
    constexpr int P = Model::NV;
    const int nv = link.nv, C = link.n_channels;
    float M[MAX_GLOBAL_NV * MAX_GLOBAL_NV] = {0}, Minv[MAX_GLOBAL_NV * MAX_GLOBAL_NV] = {0};
    float dudt[P], dudt_global[MAX_CHANNELS * MAX_GLOBAL_NV];
    float mu[MAX_CHANNELS], d[MAX_CHANNELS];
    float div = 0.0f;

    for (int c = 0; c < C; ++c) models[c].prepare(theta + c * P, sz);

    for (int ix = 0; ix < sz; ++ix)
        for (int iy = 0; iy < sz; ++iy) {
            for (int c = 0; c < C; ++c) {
                models[c].value(ix, iy, theta + c * P, dudt, &mu[c]);
                chain<Model>(link, dudt, c, nv, dudt_global + c * nv);
                d[c] = data[c][iy * sz + ix];
            }

            for (int k = 0; k < nv; ++k)
                for (int l = k; l < nv; ++l) {
                    for (int c = 0; c < C; ++c)
                        M[k * nv + l] += dudt_global[c * nv + l] *
                                         dudt_global[c * nv + k] / mu[c];
                    M[l * nv + k] = M[k * nv + l];
                }

            for (int c = 0; c < C; ++c)
                if (mu[c] > 0.0f) {
                    if (d[c] > 0.0f)
                        div += d[c] * std::log(mu[c]) - mu[c] - d[c] * std::log(d[c]) + d[c];
                    else
                        div += -mu[c];
                }
        }

    mat_inv_n(M, Minv, crlb, nv);
    *logl = div;
}

// The model's own limits, applied in the global vector's units.  A shared
// parameter is clamped once, through channel 0, so the channels cannot pull
// it in different directions.
template <class Model>
inline void clamp_global(const Model* models, const Link& link, int sz, float* global) {
    constexpr int P = Model::NV;
    float scratch[P];
    for (int c = 0; c < link.n_channels; ++c) {
        if (c > 0) {
            bool any_free = false;
            for (int p = 0; p < P; ++p) any_free |= !link.shared[p];
            if (!any_free) break;
        }
        for (int p = 0; p < P; ++p) scratch[p] = global[link.slot(p, c)];
        models[c].clamp(scratch, sz);
        for (int p = 0; p < P; ++p)
            if (!link.shared[p] || c == 0) global[link.slot(p, c)] = scratch[p];
    }
}

// Fit one emitter across `C` channels.  `global` and `crlb` need `link.nv`
// entries; `models[c]` and `data[c]` are that channel's PSF and ROI.
template <class Model>
inline void global_fit(const Model* models, const float* const* data, int sz,
                       const Link& link, int iterations, float* global, float* crlb,
                       float* logl, int* used_iterations) {
    constexpr int P = Model::NV;
    const int nv = link.nv, C = link.n_channels;

    float theta[P * MAX_CHANNELS];
    float old_global[MAX_GLOBAL_NV], maxjump[MAX_GLOBAL_NV];
    float new_update[MAX_GLOBAL_NV], old_update[MAX_GLOBAL_NV];
    float jacobian[MAX_GLOBAL_NV] = {0};
    float hessian[MAX_GLOBAL_NV * MAX_GLOBAL_NV] = {0};
    float L[MAX_GLOBAL_NV * MAX_GLOBAL_NV] = {0}, U[MAX_GLOBAL_NV * MAX_GLOBAL_NV] = {0};

    for (int i = 0; i < nv; ++i) {
        new_update[i] = 1e13f;
        old_update[i] = 1e13f;
        maxjump[i] = 1.0f;
    }

    // Start each channel from its own ROI, exactly as the single-channel
    // fitter would, and take a shared parameter's start from channel 0.  A
    // free one keeps every channel's own guess, which is the point of leaving
    // it free -- the photon split of a ratiometric splitter, say.
    float start[P * MAX_CHANNELS], jump[P * MAX_CHANNELS];
    for (int c = 0; c < C; ++c) {
        for (int p = 0; p < P; ++p) jump[c * P + p] = 1.0f;
        models[c].init(data[c], sz, start + c * P, jump + c * P);
    }
    for (int p = 0; p < P; ++p)
        for (int c = 0; c < C; ++c) {
            const int s = link.slot(p, c);
            if (link.shared[p] && c > 0) continue;
            global[s] = start[c * P + p];
            maxjump[s] = jump[c * P + p];
        }

    expand<P>(link, global, theta);
    for (int i = 0; i < nv; ++i) old_global[i] = global[i];

    float new_lambda = INIT_LAMBDA, old_lambda = INIT_LAMBDA, mu = 1.0f;
    float new_err = 0.0f, old_err = 1e13f;
    int err_flag = 0;

    accumulate_global(models, data, sz, link, theta, &new_err, jacobian, hessian);

    int k = 0;
    for (; k < iterations; ++k) {
        if (std::fabs((new_err - old_err) / new_err) < TOLERANCE) break;  // converged

        if (new_err > ACCEPTANCE * old_err) {
            for (int i = 0; i < nv; ++i) {
                global[i] = old_global[i];
                new_update[i] = old_update[i];
            }
            new_lambda = old_lambda;
            new_err = old_err;
            mu = std::max((1 + new_lambda * SCALE_UP) / (1 + new_lambda), 1.3f);
            new_lambda *= SCALE_UP;
        } else if (new_err < old_err && err_flag == 0) {
            new_lambda *= SCALE_DOWN;
            mu = 1 + new_lambda;
        }

        for (int i = 0; i < nv; ++i) hessian[i * nv + i] *= mu;

        std::memset(L, 0, nv * nv * sizeof(float));
        std::memset(U, 0, nv * nv * sizeof(float));
        err_flag = cholesky(hessian, nv, L, U);

        if (err_flag != 0) {
            mu = std::max((1 + new_lambda * SCALE_UP) / (1 + new_lambda), 1.3f);
            new_lambda *= SCALE_UP;
            continue;
        }

        for (int i = 0; i < nv; ++i) {
            old_global[i] = global[i];
            old_update[i] = new_update[i];
        }
        old_lambda = new_lambda;
        old_err = new_err;

        lu_evaluate(L, U, jacobian, nv, new_update);

        for (int i = 0; i < nv; ++i) {
            if (new_update[i] / old_update[i] < -0.5f) maxjump[i] *= 0.5f;
            new_update[i] = new_update[i] / (1 + std::fabs(new_update[i] / maxjump[i]));
            global[i] -= new_update[i];
        }
        clamp_global<Model>(models, link, sz, global);

        expand<P>(link, global, theta);
        accumulate_global(models, data, sz, link, theta, &new_err, jacobian, hessian);
    }

    *used_iterations = k;
    expand<P>(link, global, theta);
    crlb_and_logl_global(models, data, sz, link, theta, crlb, logl);
}

}  // namespace smappy
