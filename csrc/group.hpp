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

// `z` may be null: then z is not looked at; otherwise a link also needs
// |z - zh| < dz, so two emitters above each other stay two.
inline int64_t connect_single(const double* x, const double* y,
                              const int64_t* frame, int64_t n, double dx,
                              int64_t dt, int64_t* list,
                              const double* z = nullptr, double dz = 0.0) {
    int64_t entry = 0, particle = 0;

    while (entry < n) {
        while (entry < n && list[entry] > 0) ++entry;   // bounds test first
        if (entry >= n) break;

        list[entry] = ++particle;
        double xh = x[entry], yh = y[entry];
        double zh = z ? z[entry] : 0.0;
        int64_t fh = frame[entry], dark = 0, test = entry;

        while (test < n && dark <= dt) {
            bool found = false;

            // one emitter cannot be localized twice in the same frame
            while (test < n && frame[test] == fh) ++test;
            const int64_t next = fh + 1;

            // sorted by x within the frame, so walk up to the window
            while (test < n && frame[test] == next && x[test] < xh - dx) ++test;

            while (test < n && frame[test] == next && x[test] < xh + dx) {
                if (list[test] == 0 && y[test] > yh - dx && y[test] < yh + dx
                    && (!z || (z[test] > zh - dz && z[test] < zh + dz))) {
                    found = true;
                    list[test] = particle;
                    xh = (x[test] + xh) / 2;
                    yh = (y[test] + yh) / 2;
                    if (z) zh = (z[test] + zh) / 2;
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
