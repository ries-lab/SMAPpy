// Greedy frame-to-frame linking: the core of SMAP's grouper.
//
// A single emitter is localized in many consecutive frames, so a raw table
// counts one blink many times.  Linking walks each particle forward: from its
// running position, look in the next frame for the first localization inside a
// `dx`-half-width **box** (a box, not a radius -- SMAP tests |dx| and |dy|
// separately), and allow up to `dt` dark frames before closing the group.  The
// first match wins, not the nearest one.
//
// The running position is a recursive two-point mean, `xh = (xh + x_new)/2`,
// which weights recent frames far more than early ones.  That is fine for
// following a particle, which is all it is for -- the group's actual position
// is the weighted mean computed afterwards.
//
// With `var` given -- each localization's lateral precision squared -- the
// test is SMAP's other mode (`connectsinglesigma.c`): a localization joins if
// its distance from the running position is within what the two precisions
// allow,
//
//     r^2 < k2 * (var_h + var),   clamped to [r2min, dx^2],
//
// so `dx` becomes the upper bound and the box only the search window.  `var_h`
// is the variance of the running position, carried through the same mean:
// `xh = (xh + x)/2` has variance `(var_h + var)/4`.  SMAP took the seed's
// precision for both terms instead, so a bright seed held a dim trace to its
// own precision and a dim seed let a bright one wander.
//
// Input must be sorted by (frame, x) ascending within one block, and `list`
// zeroed.  On exit `list` holds 1-based group ids.
//
// Ported from SMAP's `connectsingle2c.c` with one bug fixed: the original
// tested `list[thisentry] > 0` *before* the bounds check, so with the last
// localization already assigned it read -- and could then write -- one past the
// end of the array.  `Grouper.m` carries two workarounds for the resulting
// unassigned first/last entries ("FIX connectsingle doesnt assign last loc.").
// With the tests ordered correctly every localization gets a group and the
// workarounds are unnecessary.
#pragma once

#include <cstdint>

namespace smappy {

inline int64_t connect_single(const double* x, const double* y,
                              const int64_t* frame, int64_t n, double dx,
                              int64_t dt, int64_t* list,
                              const double* var = nullptr, double k2 = 0.0,
                              double r2min = 0.0) {
    const double r2max = dx * dx;
    int64_t entry = 0, particle = 0;

    while (entry < n) {
        while (entry < n && list[entry] > 0) ++entry;   // bounds test first
        if (entry >= n) break;

        list[entry] = ++particle;
        double xh = x[entry], yh = y[entry];
        double vh = var ? var[entry] : 0.0;
        int64_t fh = frame[entry], dark = 0, test = entry;

        while (test < n && dark <= dt) {
            bool found = false;

            // one emitter cannot be localized twice in the same frame
            while (test < n && frame[test] == fh) ++test;
            const int64_t next = fh + 1;

            // sorted by x within the frame, so walk up to the window
            while (test < n && frame[test] == next && x[test] < xh - dx) ++test;

            while (test < n && frame[test] == next && x[test] < xh + dx) {
                bool near;
                if (var) {
                    const double ex = x[test] - xh, ey = y[test] - yh;
                    double limit = k2 * (vh + var[test]);   // inf for no precision
                    if (limit < r2min) limit = r2min;
                    if (limit > r2max) limit = r2max;
                    near = ex * ex + ey * ey < limit;
                } else {
                    near = y[test] > yh - dx && y[test] < yh + dx;
                }
                if (list[test] == 0 && near) {
                    found = true;
                    list[test] = particle;
                    xh = (x[test] + xh) / 2;
                    yh = (y[test] + yh) / 2;
                    if (var) vh = (vh + var[test]) / 4;
                    fh = next;
                    dark = 0;
                    break;
                }
                ++test;
            }
            if (!found) {
                fh = next;
                ++dark;
            }
        }
    }
    return particle;
}

}  // namespace smappy
