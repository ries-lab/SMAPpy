"""The GPU engine (wgpu-py: Metal on macOS, Vulkan / DX12 elsewhere, headless).

The localizations of a table live on the GPU once (x, y, z, precision,
weight, colour value); a render uploads only the projection and the filter's
index list.  A compute shader projects every selected point, clips it to the
slab, works out its sigma the way `SigmaSettings` does, and splats the same
pixel-integral Gaussian as the C++ kernel (``erf(right) - erf(left)``,
normalised by the ROI's own sum) into 64-bit fixed-point accumulators with
atomics -- exact, whatever the density, where float blending would round.
The planes come back to the CPU, so `DisplaySettings.apply` and everything
after it are unchanged.  A render pipeline draws alpha sprites for the
point-cloud mode.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from . import lut as luts
from .render import FieldOfView, RenderedImage, ROI_SIGMA

SCALE = 2.0 ** 20            # fixed point: 1.0 -> 2^20; 64 bits hold 1.7e13 per pixel
WORKGROUP = 256

COMMON = """
struct Params {
    m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>,   // rotation, rows
    pivot: vec4<f32>,       // xyz; w = focal length (0 = orthographic)
    slab_c: vec4<f32>,      // slab centre xyz; w = angle (rad)
    slab_h: vec4<f32>,      // slab half sizes; w = 1 if the slab applies
    fov: vec4<f32>,         // x0, y0, pixelsize, roi_sigma
    size: vec4<f32>,        // nx, ny, sigma mode (0 hist, 1 constant, 2 precision), constant sigma
    sig: vec4<f32>,         // factor, floor, cap, use weight column
    depth: vec4<f32>,       // attenuation length (0 off), front, colour mode (0 none, 1 field, 2 depth), n
    crange: vec4<f32>,      // colour lo, hi, point radius (px), point alpha
    dclip: vec4<f32>,       // depth lo, hi, 1 if the clip applies, pad
    sph: vec4<f32>,         // spheres: depth min, depth max (front), SSAO radius (px), SSAO strength
    light: vec4<f32>,       // light direction in view space (x right, y down, z to the viewer), ambient
};
struct Pt { x: f32, y: f32, z: f32, prec: f32, w: f32, cval: f32, p0: f32, p1: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pts: array<Pt>;
@group(0) @binding(2) var<storage, read> sel: array<u32>;
@group(0) @binding(3) var<storage, read> lut: array<vec4<f32>, 256>;

struct Proj { px: f32, py: f32, d: f32, inside: bool };

fn project(p: Pt) -> Proj {
    var out: Proj;
    out.inside = true;
    if (P.slab_h.w > 0.5) {
        let a = P.slab_c.w;
        let dx = p.x - P.slab_c.x;
        let dy = p.y - P.slab_c.y;
        let u = cos(a) * dx + sin(a) * dy;
        let v = -sin(a) * dx + cos(a) * dy;
        let w = p.z - P.slab_c.z;
        if (abs(u) > P.slab_h.x || abs(v) > P.slab_h.y || abs(w) > P.slab_h.z) {
            out.inside = false;
        }
    }
    let q = vec3<f32>(p.x - P.pivot.x, p.y - P.pivot.y, p.z - P.pivot.z);
    var vx = dot(P.m0.xyz, q);
    var vy = dot(P.m1.xyz, q);
    let d = dot(P.m2.xyz, q);
    if (P.dclip.z > 0.5 && (d < P.dclip.x || d > P.dclip.y)) { out.inside = false; }
    if (P.pivot.w > 0.0) {
        let s = 1.0 / max(1.0 - d / P.pivot.w, 0.05);
        vx = vx * s;
        vy = vy * s;
    }
    out.px = (vx - P.fov.x) / P.fov.z;
    out.py = (vy - P.fov.y) / P.fov.z;
    out.d = d;
    return out;
}

fn sigma_px(p: Pt) -> f32 {
    if (P.size.z < 0.5) { return 0.0; }
    if (P.size.z < 1.5) { return P.size.w / P.fov.z; }
    var s = p.prec * P.sig.x;
    if (s != s) { s = P.sig.y; }                  // NaN precision -> the floor
    s = clamp(s, P.sig.y, P.sig.z);
    return s / P.fov.z;
}

fn weight(p: Pt, d: f32) -> f32 {
    var w = 1.0;
    if (P.sig.w > 0.5) { w = p.w; }
    if (P.depth.x > 0.0) { w = w * exp(-(P.depth.y - d) / P.depth.x); }
    return w;
}

fn colour(p: Pt, d: f32) -> vec3<f32> {
    if (P.depth.z < 0.5) { return vec3<f32>(0.0); }
    var v = p.cval;
    if (P.depth.z > 1.5) { v = d; }
    var t = 0.0;
    if (P.crange.y > P.crange.x) { t = (v - P.crange.x) * 255.0 / (P.crange.y - P.crange.x); }
    let i = u32(clamp(t, 0.0, 255.0));
    return lut[i].xyz;
}

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
fn erf(x: f32) -> f32 {
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return sign(x) * y;
}
"""

SPLAT = COMMON + """
@group(0) @binding(4) var<storage, read_write> acc: array<atomic<u32>>;

fn add64(base: u32, v: f32) {
    let t = v * 1048576.0;                          // SCALE
    if (t <= 0.0) { return; }
    let hi = floor(t / 4294967296.0);
    let lo = u32(t - hi * 4294967296.0);
    let old = atomicAdd(&acc[base], lo);
    var carry = u32(hi);
    if (old > 4294967295u - lo) { carry = carry + 1u; }
    if (carry > 0u) { atomicAdd(&acc[base + 1u], carry); }
}

@compute @workgroup_size(256)
fn splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (f32(i) >= P.depth.w) { return; }
    let p = pts[sel[i]];
    let pr = project(p);
    if (!pr.inside) { return; }
    let nx = i32(P.size.x);
    let ny = i32(P.size.y);
    let w = weight(p, pr.d);
    let c = colour(p, pr.d);
    let s = sigma_px(p);
    let cx = i32(floor(pr.px));
    let cy = i32(floor(pr.py));
    if (s <= 0.0) {                                 // histogram: one pixel
        if (cx < 0 || cy < 0 || cx >= nx || cy >= ny) { return; }
        let base = u32(cy * nx + cx) * 8u;
        add64(base, w);
        add64(base + 2u, w * c.x);
        add64(base + 4u, w * c.y);
        add64(base + 6u, w * c.z);
        return;
    }
    let d = i32(floor(P.fov.w * s + 1.0));
    if (cx + d < 0 || cy + d < 0 || cx - d >= nx || cy - d >= ny) { return; }
    let scale = 1.0 / (s * 1.4142135623730951);
    let ox = pr.px - f32(cx);
    let oy = pr.py - f32(cy);
    let fd = f32(d);
    // pixel k spans [k, k + 1) relative to the centre pixel, as the C++ kernel has it
    let normx = erf((fd + 1.0 - ox) * scale) - erf((-fd - ox) * scale);
    let normy = erf((fd + 1.0 - oy) * scale) - erf((-fd - oy) * scale);
    let norm = w / (normx * normy);
    for (var j = -d; j <= d; j = j + 1) {
        let row = cy + j;
        if (row < 0 || row >= ny) { continue; }
        let fj = f32(j);
        let ky = erf((fj + 1.0 - oy) * scale) - erf((fj - oy) * scale);
        for (var k = -d; k <= d; k = k + 1) {
            let col = cx + k;
            if (col < 0 || col >= nx) { continue; }
            let fk = f32(k);
            let kx = erf((fk + 1.0 - ox) * scale) - erf((fk - ox) * scale);
            let v = norm * kx * ky;
            let base = u32(row * nx + col) * 8u;
            add64(base, v);
            if (P.depth.z > 0.5) {
                add64(base + 2u, v * c.x);
                add64(base + 4u, v * c.y);
                add64(base + 6u, v * c.z);
            }
        }
    }
}
"""

POINTS = COMMON + """
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) @interpolate(flat) idx: u32,
              @location(1) @interpolate(flat) centre: vec2<f32>,
              @location(2) @interpolate(flat) rgb: vec3<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
                                      vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let p = pts[sel[ii]];
    let pr = project(p);
    var out: VOut;
    out.idx = ii;
    out.centre = vec2<f32>(pr.px, pr.py);
    out.rgb = colour(p, pr.d);
    if (P.depth.z < 0.5) { out.rgb = lut[255].xyz; }
    let r = P.crange.z + 1.0;
    if (!pr.inside) {
        out.pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);          // off screen, degenerate
        return out;
    }
    let q = mix(vec2<f32>(pr.px - r, pr.py - r), vec2<f32>(pr.px + r, pr.py + r), corners[vi]);
    out.pos = vec4<f32>(q.x / P.size.x * 2.0 - 1.0, 1.0 - q.y / P.size.y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let r = length(in.pos.xy - in.centre) / max(P.crange.z, 0.5);
    let alpha = P.crange.w * (1.0 - smoothstep(0.7, 1.0, r));
    return vec4<f32>(in.rgb * alpha, alpha);
}
"""


SPHERES = COMMON + """
struct VOut { @builtin(position) pos: vec4<f32>,
              @location(0) @interpolate(flat) centre: vec3<f32>,   // px, py, depth (nm)
              @location(1) @interpolate(flat) radius: f32,
              @location(2) @interpolate(flat) rgb: vec3<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
                                      vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let p = pts[sel[ii]];
    let pr = project(p);
    var out: VOut;
    out.centre = vec3<f32>(pr.px, pr.py, pr.d);
    out.radius = P.crange.z;
    var rgb = colour(p, pr.d);
    if (P.depth.z < 0.5) { rgb = lut[255].xyz; }
    out.rgb = rgb * weight(p, pr.d);               // depth dimming, if asked for
    let r = P.crange.z + 1.0;
    if (!pr.inside) {
        out.pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        return out;
    }
    let q = mix(vec2<f32>(pr.px - r, pr.py - r), vec2<f32>(pr.px + r, pr.py + r), corners[vi]);
    out.pos = vec4<f32>(q.x / P.size.x * 2.0 - 1.0, 1.0 - q.y / P.size.y * 2.0, 0.0, 1.0);
    return out;
}

struct FOut { @location(0) albedo: vec4<f32>, @location(1) normal_depth: vec4<f32>,
              @builtin(frag_depth) depth: f32 };

@fragment
fn fs(in: VOut) -> FOut {
    // a ray along the depth axis through this pixel: where does it meet the sphere?
    let q = in.pos.xy - in.centre.xy;                     // pixels from the centre
    let r2 = in.radius * in.radius;
    let h2 = r2 - dot(q, q);
    if (h2 < 0.0) { discard; }
    let dz = sqrt(h2);                                    // pixels towards the viewer
    let surface = in.centre.z + dz * P.fov.z;             // depth in nm (bigger = nearer)
    let n = normalize(vec3<f32>(q.x, q.y, dz) / in.radius);
    var out: FOut;
    out.albedo = vec4<f32>(in.rgb, 1.0);
    out.normal_depth = vec4<f32>(n, surface - P.sph.x + 1.0);   // 0 stays "nothing here"
    out.depth = clamp((P.sph.y - surface) / max(P.sph.y - P.sph.x, 1e-6), 0.0, 1.0);
    return out;
}
"""

SHADE = """
struct Params {
    m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, pivot: vec4<f32>, slab_c: vec4<f32>,
    slab_h: vec4<f32>, fov: vec4<f32>, size: vec4<f32>, sig: vec4<f32>, depth: vec4<f32>,
    crange: vec4<f32>, dclip: vec4<f32>, sph: vec4<f32>, light: vec4<f32>,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var albedo: texture_2d<f32>;
@group(0) @binding(2) var gbuf: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> ao_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> rgb_out: array<vec4<f32>>;

fn hash(p: vec2<i32>) -> f32 {
    let x = f32(p.x) * 12.9898 + f32(p.y) * 78.233;
    return fract(sin(x) * 43758.5453);
}

// 16 hemisphere samples (z up), denser near the centre
const KERNEL = array<vec3<f32>, 16>(
    vec3<f32>(0.10, 0.02, 0.15), vec3<f32>(-0.05, 0.12, 0.10), vec3<f32>(0.02, -0.09, 0.20),
    vec3<f32>(-0.14, -0.06, 0.12), vec3<f32>(0.22, 0.11, 0.30), vec3<f32>(-0.18, 0.21, 0.25),
    vec3<f32>(0.05, -0.28, 0.35), vec3<f32>(-0.30, -0.15, 0.40), vec3<f32>(0.38, -0.12, 0.45),
    vec3<f32>(-0.10, 0.42, 0.50), vec3<f32>(0.25, 0.35, 0.55), vec3<f32>(-0.45, 0.10, 0.60),
    vec3<f32>(0.15, -0.52, 0.65), vec3<f32>(-0.35, -0.40, 0.70), vec3<f32>(0.55, 0.25, 0.75),
    vec3<f32>(-0.20, 0.60, 0.85));

@compute @workgroup_size(16, 16)
fn ao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = i32(P.size.x);
    let ny = i32(P.size.y);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= nx || y >= ny) { return; }
    let g = textureLoad(gbuf, vec2<i32>(x, y), 0);
    let idx = u32(y * nx + x);
    if (g.w <= 0.0) { ao_out[idx] = 1.0; return; }
    let n = g.xyz;
    let d = g.w;
    let R = P.sph.z;                                   // pixels
    // a tangent frame, turned by a per-pixel angle so banding becomes noise
    let a = hash(vec2<i32>(x, y)) * 6.2831853;
    var up = vec3<f32>(cos(a), sin(a), 0.0);
    let t = normalize(up - n * dot(up, n));
    let b = cross(n, t);
    var occluded = 0.0;
    for (var i = 0; i < 16; i = i + 1) {
        let k = KERNEL[i];
        let s = (t * k.x + b * k.y + n * k.z) * R;    // pixels; z in pixels of depth too
        let sx = x + i32(round(s.x));
        let sy = y + i32(round(s.y));
        if (sx < 0 || sy < 0 || sx >= nx || sy >= ny) { continue; }
        let sample_depth = d + s.z * P.fov.z;         // nm
        let there = textureLoad(gbuf, vec2<i32>(sx, sy), 0).w;
        if (there <= 0.0) { continue; }
        let dz = there - sample_depth;                // > 0: the surface there is in front of the sample
        if (dz > 0.05 * R * P.fov.z && dz < 2.0 * R * P.fov.z) { occluded = occluded + 1.0; }
    }
    ao_out[idx] = 1.0 - P.sph.w * occluded / 16.0;
}

@compute @workgroup_size(16, 16)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = i32(P.size.x);
    let ny = i32(P.size.y);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= nx || y >= ny) { return; }
    let idx = u32(y * nx + x);
    let g = textureLoad(gbuf, vec2<i32>(x, y), 0);
    if (g.w <= 0.0) { rgb_out[idx] = vec4<f32>(0.0, 0.0, 0.0, 1.0); return; }
    // a 4x4 box blur of the noisy occlusion
    var ao = 0.0;
    var cnt = 0.0;
    for (var j = -2; j < 2; j = j + 1) {
        for (var i = -2; i < 2; i = i + 1) {
            let sx = x + i;
            let sy = y + j;
            if (sx < 0 || sy < 0 || sx >= nx || sy >= ny) { continue; }
            if (textureLoad(gbuf, vec2<i32>(sx, sy), 0).w <= 0.0) { continue; }
            ao = ao + ao_out[u32(sy * nx + sx)];
            cnt = cnt + 1.0;
        }
    }
    ao = ao / max(cnt, 1.0);
    let n = normalize(g.xyz);
    let l = normalize(P.light.xyz);
    let v = vec3<f32>(0.0, 0.0, 1.0);
    let diffuse = max(dot(n, l), 0.0);
    let spec = pow(max(dot(reflect(-l, n), v), 0.0), 32.0);
    let base = textureLoad(albedo, vec2<i32>(x, y), 0).xyz;
    let c = base * (P.light.w + (1.0 - P.light.w) * diffuse) * ao + vec3<f32>(0.25 * spec * ao);
    rgb_out[idx] = vec4<f32>(c, 1.0);
}
"""


class GPUEngine:
    """One device; tables cached on it; renders on demand."""

    def __init__(self):
        import wgpu
        self.wgpu = wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync()
        self.info = dict(adapter.info)
        dev = self.device
        entries = [
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.VERTEX
             | wgpu.ShaderStage.FRAGMENT, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}]
        self.splat_layout = dev.create_bind_group_layout(entries=entries + [
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}}])
        self.points_layout = dev.create_bind_group_layout(entries=entries)
        self.splat = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self.splat_layout]),
            compute={"module": dev.create_shader_module(code=SPLAT), "entry_point": "splat"})
        over = {"operation": wgpu.BlendOperation.add, "src_factor": wgpu.BlendFactor.one,
                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha}
        pm = dev.create_shader_module(code=POINTS)
        self.points = dev.create_render_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self.points_layout]),
            vertex={"module": pm, "entry_point": "vs"},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={"module": pm, "entry_point": "fs",
                      "targets": [{"format": "rgba16float", "blend": {"color": over, "alpha": over}}]})
        # spheres: a g-buffer pass with a depth buffer, then SSAO and shading
        sm = dev.create_shader_module(code=SPHERES)
        self.spheres = dev.create_render_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self.points_layout]),
            vertex={"module": sm, "entry_point": "vs"},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={"format": "depth24plus", "depth_write_enabled": True,
                           "depth_compare": wgpu.CompareFunction.less},
            fragment={"module": sm, "entry_point": "fs",
                      "targets": [{"format": "rgba16float"}, {"format": "rgba16float"}]})
        self.shade_layout = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}}])
        shade_module = dev.create_shader_module(code=SHADE)
        shade_pl = dev.create_pipeline_layout(bind_group_layouts=[self.shade_layout])
        self.ao_pipeline = dev.create_compute_pipeline(
            layout=shade_pl, compute={"module": shade_module, "entry_point": "ao"})
        self.shade_pipeline = dev.create_compute_pipeline(
            layout=shade_pl, compute={"module": shade_module, "entry_point": "shade"})
        self._gbuf = None
        self._gbuf_shape = None
        self._tables: Dict[tuple, tuple] = {}        # key -> (buffer, n)
        self._selection: Dict[tuple, tuple] = {}     # key -> (buffer, n)
        self._luts: Dict[tuple, object] = {}
        self._acc = None
        self._acc_shape = None
        self._target = None
        self._target_shape = None

    @property
    def name(self) -> str:
        return f"{self.info.get('device', 'GPU')} ({self.info.get('backend_type', '?')})"

    # ------------------------------------------------------------- caches
    def table(self, key, x, y, z=None, precision=None, weights=None, cvalues=None):
        """Upload a table once; ``key`` says which (the table object and the
        column names).  Returns the number of points."""
        if key in self._tables:
            return self._tables[key][1]
        n = len(x)
        rec = np.zeros((n, 8), np.float32)
        rec[:, 0], rec[:, 1] = x, y
        rec[:, 2] = 0.0 if z is None else z
        rec[:, 3] = np.nan if precision is None else precision
        rec[:, 4] = 1.0 if weights is None else weights
        rec[:, 5] = 0.0 if cvalues is None else cvalues
        buf = self.device.create_buffer_with_data(data=rec.tobytes() if n else bytes(32),
                                                  usage=self.wgpu.BufferUsage.STORAGE)
        self._tables.clear()                         # one table at a time on the GPU
        self._tables[key] = (buf, n)
        self._selection.clear()
        # keys carry Python ids; keep what they refer to alive so an id is not
        # reused by a new object while the cache still names the old one
        self._keepalive = [x, y]
        return n

    def selection(self, key, indices: np.ndarray):
        if key in self._selection:
            return self._selection[key][:2]
        idx = np.ascontiguousarray(indices, dtype=np.uint32)
        buf = self.device.create_buffer_with_data(data=idx.tobytes() if idx.size else bytes(4),
                                                  usage=self.wgpu.BufferUsage.STORAGE)
        if len(self._selection) > 4:
            self._selection.clear()
        self._selection[key] = (buf, int(idx.size), indices)   # the array itself keeps its id
        return self._selection[key][:2]

    def lut_buffer(self, lut, invert: bool):
        key = (lut if isinstance(lut, str) else id(lut), invert)
        if key not in self._luts:
            table = np.zeros((256, 4), np.float32)
            table[:, :3] = luts.get(lut, invert)[:256]
            self._luts[key] = self.device.create_buffer_with_data(
                data=table.tobytes(), usage=self.wgpu.BufferUsage.STORAGE)
        return self._luts[key]

    # ------------------------------------------------------------- params
    @staticmethod
    def params(fov: FieldOfView, matrix=None, pivot=(0, 0, 0), focal=None, slab=None,
               sigma_mode: str = "precision", sigma: float = 0.0, factor: float = 1.0,
               floor: float = 0.0, cap: float = 1e30, use_weight: bool = False,
               depth_lambda: Optional[float] = None, depth_front: float = 0.0,
               color_mode: int = 0, color_range=(0.0, 1.0), n: int = 0,
               radius: float = 2.0, alpha: float = 0.5,
               depth_clip: Optional[Tuple[float, float]] = None,
               depth_range: Tuple[float, float] = (0.0, 1.0),
               ssao_radius: float = 8.0, ssao_strength: float = 0.7,
               light=(-0.4, -0.6, 0.7), ambient: float = 0.25) -> np.ndarray:
        m = np.eye(3) if matrix is None else np.asarray(matrix, float)
        rows = [np.r_[m[i], 0.0] for i in range(3)]
        px, py, pz = pivot
        if slab is not None:
            sc = np.r_[slab.center, math.radians(slab.angle)]
            sh = np.r_[slab.size / 2, 1.0]
        else:
            sc, sh = np.zeros(4), np.zeros(4)
        mode = {"hist": 0.0, "gauss": 1.0, "precision": 2.0}[sigma_mode]
        return np.concatenate([
            rows[0], rows[1], rows[2],
            [px, py, pz, focal or 0.0], sc, sh,
            [fov.x0, fov.y0, fov.pixelsize, ROI_SIGMA],
            [fov.nx, fov.ny, mode, sigma],
            [factor, floor, cap, 1.0 if use_weight else 0.0],
            [depth_lambda or 0.0, depth_front, float(color_mode), float(n)],
            [color_range[0], color_range[1], radius, alpha],
            [depth_clip[0], depth_clip[1], 1.0, 0.0] if depth_clip else [0.0, 0.0, 0.0, 0.0],
            [depth_range[0], depth_range[1], ssao_radius, ssao_strength],
            [light[0], light[1], light[2], ambient],
        ]).astype(np.float32)

    # ------------------------------------------------------------- render
    def render_planes(self, table_key, selection, fov: FieldOfView, params: np.ndarray,
                      lut="hot", invert: bool = False, colored: bool = False) -> RenderedImage:
        wgpu, dev = self.wgpu, self.device
        tbuf, _ = self._tables[table_key]
        sbuf, n = selection
        nx, ny = fov.nx, fov.ny
        size = nx * ny * 8 * 4
        if self._acc_shape != (nx, ny):
            self._acc = dev.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE
                                          | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
            self._acc_shape = (nx, ny)
        ubuf = dev.create_buffer_with_data(data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        bind = dev.create_bind_group(layout=self.splat_layout, entries=[
            {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}},
            {"binding": 1, "resource": {"buffer": tbuf, "offset": 0, "size": tbuf.size}},
            {"binding": 2, "resource": {"buffer": sbuf, "offset": 0, "size": sbuf.size}},
            {"binding": 3, "resource": {"buffer": self.lut_buffer(lut, invert), "offset": 0,
                                        "size": 256 * 16}},
            {"binding": 4, "resource": {"buffer": self._acc, "offset": 0, "size": size}}])
        out = dev.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        enc = dev.create_command_encoder()
        enc.clear_buffer(self._acc, 0, size)
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.splat)
        cp.set_bind_group(0, bind)
        if n:
            cp.dispatch_workgroups((n + WORKGROUP - 1) // WORKGROUP)
        cp.end()
        enc.copy_buffer_to_buffer(self._acc, 0, out, 0, size)
        dev.queue.submit([enc.finish()])
        out.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(out.read_mapped(), np.uint32).reshape(ny, nx, 4, 2)
        out.unmap()
        planes = (raw[..., 1].astype(np.float64) * 4294967296.0 + raw[..., 0]) / SCALE
        planes = planes.astype(np.float32)
        weight = np.ascontiguousarray(planes[..., 0])
        color = np.ascontiguousarray(planes[..., 1:4]) if colored else None
        return RenderedImage(fov, weight, color, n_locs=int(n))

    def render_points(self, table_key, selection, fov: FieldOfView, params: np.ndarray,
                      lut="hot", invert: bool = False) -> np.ndarray:
        """Alpha sprites over black, in the selection's order; RGB in [0, 1]."""
        wgpu, dev = self.wgpu, self.device
        tbuf, _ = self._tables[table_key]
        sbuf, n = selection
        nx, ny = fov.nx, fov.ny
        if self._target_shape != (nx, ny):
            self._target = dev.create_texture(size=(nx, ny, 1), format="rgba16float",
                                              usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                                              | wgpu.TextureUsage.COPY_SRC)
            self._target_shape = (nx, ny)
        ubuf = dev.create_buffer_with_data(data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        bind = dev.create_bind_group(layout=self.points_layout, entries=[
            {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}},
            {"binding": 1, "resource": {"buffer": tbuf, "offset": 0, "size": tbuf.size}},
            {"binding": 2, "resource": {"buffer": sbuf, "offset": 0, "size": sbuf.size}},
            {"binding": 3, "resource": {"buffer": self.lut_buffer(lut, invert), "offset": 0,
                                        "size": 256 * 16}}])
        enc = dev.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{
            "view": self._target.create_view(), "clear_value": (0, 0, 0, 0),
            "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}])
        rp.set_pipeline(self.points)
        rp.set_bind_group(0, bind)
        if n:
            rp.draw(6, n, 0, 0)
        rp.end()
        row = (nx * 8 + 255) // 256 * 256
        out = dev.create_buffer(size=row * ny, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        enc.copy_texture_to_buffer({"texture": self._target},
                                   {"buffer": out, "bytes_per_row": row, "rows_per_image": ny},
                                   (nx, ny, 1))
        dev.queue.submit([enc.finish()])
        out.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(out.read_mapped(), np.uint8).reshape(ny, row)
        out.unmap()
        rgba = raw[:, :nx * 8].view(np.float16).reshape(ny, nx, 4).astype(np.float32)
        return np.clip(rgba[..., :3], 0, 1)

    def render_spheres(self, draws, fov: FieldOfView, shade_params: np.ndarray) -> np.ndarray:
        """Opaque shaded spheres.  ``draws`` are (table_key, selection, params,
        lut, invert) per layer, all into one depth buffer, so spheres of one
        layer hide those of another.  Returns RGB in [0, 1]."""
        wgpu, dev = self.wgpu, self.device
        nx, ny = fov.nx, fov.ny
        if self._gbuf_shape != (nx, ny):
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
            self._gbuf = (dev.create_texture(size=(nx, ny, 1), format="rgba16float", usage=usage),
                          dev.create_texture(size=(nx, ny, 1), format="rgba16float", usage=usage),
                          dev.create_texture(size=(nx, ny, 1), format="depth24plus",
                                             usage=wgpu.TextureUsage.RENDER_ATTACHMENT))
            self._gbuf_shape = (nx, ny)
        albedo, gbuf, depth = self._gbuf
        enc = dev.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[{"view": albedo.create_view(), "clear_value": (0, 0, 0, 0),
                                "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store},
                               {"view": gbuf.create_view(), "clear_value": (0, 0, 0, 0),
                                "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}],
            depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 1.0,
                                      "depth_load_op": wgpu.LoadOp.clear,
                                      "depth_store_op": wgpu.StoreOp.store})
        rp.set_pipeline(self.spheres)
        keep = []
        for table_key, (sbuf, n), params, lut, invert in draws:
            tbuf, _ = self._tables[table_key]
            ubuf = dev.create_buffer_with_data(data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
            bind = dev.create_bind_group(layout=self.points_layout, entries=[
                {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}},
                {"binding": 1, "resource": {"buffer": tbuf, "offset": 0, "size": tbuf.size}},
                {"binding": 2, "resource": {"buffer": sbuf, "offset": 0, "size": sbuf.size}},
                {"binding": 3, "resource": {"buffer": self.lut_buffer(lut, invert), "offset": 0,
                                            "size": 256 * 16}}])
            keep.append((ubuf, bind))
            rp.set_bind_group(0, bind)
            if n:
                rp.draw(6, n, 0, 0)
        rp.end()
        # SSAO, then blur + shade
        ubuf = dev.create_buffer_with_data(data=shade_params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        ao_buf = dev.create_buffer(size=nx * ny * 4, usage=wgpu.BufferUsage.STORAGE)
        out_buf = dev.create_buffer(size=nx * ny * 16, usage=wgpu.BufferUsage.STORAGE
                                    | wgpu.BufferUsage.COPY_SRC)
        bind = dev.create_bind_group(layout=self.shade_layout, entries=[
            {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}},
            {"binding": 1, "resource": albedo.create_view()},
            {"binding": 2, "resource": gbuf.create_view()},
            {"binding": 3, "resource": {"buffer": ao_buf, "offset": 0, "size": ao_buf.size}},
            {"binding": 4, "resource": {"buffer": out_buf, "offset": 0, "size": out_buf.size}}])
        groups = ((nx + 15) // 16, (ny + 15) // 16)
        for pipeline in (self.ao_pipeline, self.shade_pipeline):
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipeline)
            cp.set_bind_group(0, bind)
            cp.dispatch_workgroups(*groups)
            cp.end()
        read = dev.create_buffer(size=nx * ny * 16, usage=wgpu.BufferUsage.COPY_DST
                                 | wgpu.BufferUsage.MAP_READ)
        enc.copy_buffer_to_buffer(out_buf, 0, read, 0, nx * ny * 16)
        dev.queue.submit([enc.finish()])
        read.map_sync(wgpu.MapMode.READ)
        rgba = np.frombuffer(read.read_mapped(), np.float32).reshape(ny, nx, 4).copy()
        read.unmap()
        return np.clip(rgba[..., :3], 0, 1)

    # ------------------------------------------------- the 2D convenience
    def render(self, x, y, fov: FieldOfView, sigma=None, weights=None, colors=None
               ) -> RenderedImage:
        """The GPU twin of :func:`smappy.render.render`, for tests and scripts.

        ``colors`` are (n, 3) RGB, applied through a 256-entry identity trick:
        the colour-weighted planes need per-point colours, so they are passed
        as the colour value with a grey LUT... no: `render` takes explicit
        colours, so here each point's colour is uploaded as its own LUT index
        is not possible -- instead the three planes are rendered as three
        weighted passes.  Fine for tests; the engine proper colours by field.
        """
        x = np.asarray(x, np.float32)
        n = x.size
        key = ("adhoc", id(x))
        use_sigma = sigma is not None
        prec = None if sigma is None else np.broadcast_to(np.asarray(sigma, np.float32), (n,))
        self.table(key, x, np.asarray(y, np.float32), None, prec, weights, None)
        sel = self.selection((key, "all"), np.arange(n, dtype=np.uint32))
        base = dict(fov=fov, sigma_mode="precision" if use_sigma else "hist", factor=1.0,
                    floor=0.0, cap=1e30, use_weight=weights is not None, n=n)
        rendered = self.render_planes(key, sel, fov, self.params(**base))
        if colors is not None:
            colors = np.asarray(colors, np.float32)
            planes = []
            for c in range(3):
                w = colors[:, c] * (1.0 if weights is None else np.asarray(weights, np.float32))
                key_c = ("adhoc", id(x), c)
                self.table(key_c, x, np.asarray(y, np.float32), None, prec, w, None)
                sel_c = self.selection((key_c, "all"), np.arange(n, dtype=np.uint32))
                planes.append(self.render_planes(key_c, sel_c, fov,
                                                 self.params(**{**base, "use_weight": True})).weight)
            rendered.color = np.stack(planes, axis=-1)
        return rendered


_shared: Optional[GPUEngine] = None


def gpu_engine() -> GPUEngine:
    """One engine per process; raises if there is no usable GPU."""
    global _shared
    if _shared is None:
        _shared = GPUEngine()
    return _shared


def available() -> bool:
    try:
        return gpu_engine() is not None
    except Exception:
        return False
