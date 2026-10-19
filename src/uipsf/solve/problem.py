"""A fitting problem: the model, the noise, the data and the parameters, as the solver sees them.

The solver needs three things from it: the objective, its gradient and the
Gauss–Newton matrix, all in the block layout of plan §5.6 -- one dense
block for the shared parameters (pupil, blur), one small block per emitter
(position, photons, background) and the coupling between them.  It knows
nothing about optics.

**The objective** is ½·Σ deviance + priors: the negative log-likelihood up
to a constant, *summed* over every pixel of every emitter.  v1 averaged it
per batch, so its gradients depended on how the beads fell into batches
(review F1).  Here a chunk of emitters is only a unit of memory: every sum
is formed per emitter on the device, then added up in float64 on the CPU,
so the result does not depend on the chunk size (tests/uipsf_tests/test_solve.py).

**Jacobians are never stored whole.**  Per chunk:
- the emitter block takes one forward-mode JVP per emitter parameter, with
  a unit tangent on that parameter of *every* emitter at once: emitters are
  independent (plan §5.4), so one pass gives that column of every
  emitter's Jacobian;
- the shared block takes one JVP per shared parameter, batched with
  `torch.func.vmap` in groups of ``tangents`` (memory against speed).
Each JVP differentiates only its own block; the other block is held
constant rather than given a zero tangent, which would cost as much as a
real one.  Each chunk adds JᵀWJ and Jᵀg into the blocks, W being the noise
model's Fisher weight.

**Priors** are least-squares terms, so they add exactly to the gradient
and the Gauss–Newton matrix (`Params.prior_residuals`).
"""
from __future__ import annotations

import platform
from dataclasses import dataclass

import torch

from ..device import to_cpu64


@dataclass
class Normal:
    """The normal equations at one point, float64 on the CPU: H = [[H_ss, H_se], [H_esᵀ, H_ee]]."""
    objective: float
    H_ss: torch.Tensor          # [P_s, P_s]
    H_se: torch.Tensor          # [E, P_s, P_e]
    H_ee: torch.Tensor          # [E, P_e, P_e]
    g_s: torch.Tensor           # [P_s]
    g_e: torch.Tensor           # [E, P_e]


# Output values (emitters × tangents × pixels) per batch of Jacobian columns, by device.  2²¹ is
# 5 tangents of 20 beads of 41 × 21² pixels, ~3 GB of intermediate images; twice that swapped on a
# 32 GB M1 Pro (Problem.__init__).
TANGENT_BUDGET = {"cpu": 2 ** 21, "mps": 2 ** 21}

# At most this many tangents per batch.  x86 CPUs (Linux, Windows) are fastest with few: on 4 cloud
# cores (6 beads, one normal_equations, best of 3) 2 tangents 5.0 s, 4: 5.6 s, 8: 5.9 s, 16: 6.5 s; at
# 20 beads flat within the noise (2: 20.9 s, 16: 21.8 s).  Apple's CPU and GPU are fastest with 8–16.
TANGENT_MAX = 2 if platform.machine().lower() in ("x86_64", "amd64", "i386", "i686") else 16


class Problem:
    def __init__(self, model, noise, batch, chunk_size: int | None = None, tangents: int | None = None,
                 device=None, linearise: bool = True):
        """``chunk_size`` emitters are computed at once, ``tangents`` Jacobian columns per batch; both
        are memory only, the sums do not depend on them (tests/uipsf_tests/test_solve.py).  The batch may stay on
        the CPU; each chunk is moved to ``device`` (default: the batch's).

        Defaults: as many tangents as keep a chunk's tangent images under `TANGENT_BUDGET` values for
        the device, at most 16 (`TANGENT_MAX` on the CPU); all emitters in one chunk unless even one
        tangent would exceed it.
        Measured on an M1 Pro (dim beads, 41 slices of 21², n_max 6, one `normal_equations`):

        ======  =========  ==================  ==================  =============================
        device  beads      1 tangent           default             notes
        ======  =========  ==================  ==================  =============================
        mps     6          0.94 s              0.63 s (16)
        mps     20         2.08 s              1.65 s (5)          16 tangents: 4.5 s (swapping)
        cpu     6          1.59 s              1.23 s (16)         8 tangents: 1.21 s
        cpu     20         3.96 s              3.7 s (5)           flat from 2 to 16
        ======  =========  ==================  ==================  =============================

        The old rule (2¹⁹ values) gave one tangent at 20 beads.  Chunks smaller than all the beads
        were never faster.  The same budget is best on both devices; CUDA is unmeasured.
        """
        self.model, self.noise, self.batch = model, noise, batch
        self.device = batch.data.device if device is None else torch.device(device)
        per_emitter = batch.data[0].numel()
        budget = TANGENT_BUDGET.get(self.device.type, 2 ** 21)
        self.chunk_size = chunk_size or min(batch.n_emitters, max(1, budget // per_emitter))
        if tangents is None:
            cap = TANGENT_MAX if self.device.type == "cpu" else 16
            tangents = max(1, min(cap, budget // (min(self.chunk_size, batch.n_emitters) * per_emitter)))
        self.tangents = int(tangents)
        self.linearise = linearise          # pupil columns from the model's linearisation (False: all JVPs)

    def chunks(self):
        E = self.batch.n_emitters
        for a in range(0, E, self.chunk_size):
            sl = slice(a, min(a + self.chunk_size, E))
            yield sl, self.batch.subset(sl).to(self.device)

    def expected(self, params) -> torch.Tensor:
        """μ for every emitter, [E, C, K, Y, X], on the CPU."""
        return torch.cat([self.model(params.subset(sl).physical(), b).detach().cpu()
                          for sl, b in self.chunks()])

    def deviance(self, params) -> torch.Tensor:
        """Per-pixel deviance, [E, C, K, Y, X] on the CPU (0 where masked): the residual atlas' input."""
        out = []
        with torch.no_grad():
            for sl, b in self.chunks():
                mu = self.model(params.subset(sl).physical(), b)
                out.append(self.noise.deviance(mu, b.data, b.s, b.mask).cpu())
        return torch.cat(out)

    def objective(self, params) -> float:
        """½·Σ deviance + priors, summed per emitter and then in float64."""
        total = 0.0
        with torch.no_grad():
            for sl, b in self.chunks():
                mu = self.model(params.subset(sl).physical(), b)
                D = self.noise.deviance(mu, b.data, b.s, b.mask)
                total += 0.5 * float(to_cpu64(D.flatten(1).sum(1)).sum())
            total += float(to_cpu64(params.prior_penalty()))
        return total

    def objectives(self, params) -> torch.Tensor:
        """[E] float64 on the CPU: each emitter's share of `objective`, ½ its deviance plus its own priors
        (the shared parameters' priors left out), for solvers that treat emitters one by one."""
        out = []
        with torch.no_grad():
            for sl, b in self.chunks():
                mu = self.model(params.subset(sl).physical(), b)
                D = self.noise.deviance(mu, b.data, b.s, b.mask)
                out.append(0.5 * to_cpu64(D.flatten(1).sum(1)))
            r, _ = params.prior_residuals("emitter")
        return torch.cat(out) + 0.5 * (to_cpu64(r) ** 2).sum(1)

    def value_and_grad(self, params) -> tuple[float, torch.Tensor, torch.Tensor]:
        """(objective, g_s [P_s], g_e [E, P_e]), float64 on the CPU: the gradient of `normal_equations`
        from one backward pass instead of a Jacobian, about one forward + backward per chunk.  It is
        what a solver that does not rebuild the Jacobian every step needs (plan §9, M7)."""
        Ps, Pe, E = params.shared_vector().numel(), params.emitter_matrix().shape[1], params.n_emitters
        g_s, g_e = torch.zeros(Ps, dtype=torch.float64), torch.zeros(E, Pe, dtype=torch.float64)
        objective = 0.0
        for sl, b in self.chunks():
            pc = params.subset(sl)
            us = pc.shared_vector().detach().requires_grad_(Ps > 0)
            ue = pc.emitter_matrix().detach().requires_grad_(Pe > 0)
            mu = self.model(pc.replace(shared=us, emitter=ue).physical(), b)
            with torch.no_grad():
                D = self.noise.deviance(mu, b.data, b.s, b.mask)
                objective += 0.5 * float(to_cpu64(D.flatten(1).sum(1)).sum())
                grad, _ = self.noise.derivatives(mu, b.data, b.s, b.mask)
            inputs = [t for t in (us, ue) if t.requires_grad]
            if inputs:
                grads = iter(torch.autograd.grad(mu, inputs, grad))
                if Ps:
                    g_s += to_cpu64(next(grads))
                if Pe:
                    g_e[sl] = to_cpu64(next(grads))
        r, dr = (to_cpu64(t) for t in params.prior_residuals("shared"))
        g_s += dr * r
        r, dr = (to_cpu64(t) for t in params.prior_residuals("emitter"))
        g_e += dr * r
        objective += float(to_cpu64(params.prior_penalty()))
        return objective, g_s, g_e

    def gauss_newton_operator(self, params, trace: bool = True):
        """``product(v_s, v_e) -> (Hv_s, Hv_e)``: the Gauss–Newton matrix of `normal_equations` times a
        vector, without building it (float64 on the CPU, priors included).

        Each product is one JVP and one VJP through the model at ``params``.  `torch.func.linearize`
        traces the model once (1.8 s for six beads at n_max 8 on 4 CPU cores); a product then costs
        0.17 s, 0.8 of a gradient, against 8.4 s for the whole matrix.  Matrix-free solvers need it
        (plan §9 M7, candidate C), and so will models whose shared block is too large to build.
        The traces of every chunk are kept until the operator is dropped.  ``trace=False`` skips the
        trace: each product then recomputes the model (2.3 gradients' worth), cheaper for few products.
        """
        parts = []
        for sl, b in self.chunks():
            pc = params.subset(sl)
            us, ue = pc.shared_vector().detach(), pc.emitter_matrix().detach()

            def f(a, e, pc=pc, b=b):
                return self.model(pc.replace(shared=a, emitter=e).physical(), b).flatten(1)

            mu, vjp_fn = torch.func.vjp(f, us, ue)
            try:
                if not trace:
                    raise NotImplementedError
                _, jvp_fn = torch.func.linearize(f, us, ue)
            except Exception:                           # untraced, or a device or op linearize cannot trace
                def jvp_fn(a, e, f=f, us=us, ue=ue):
                    return torch.func.jvp(f, (us, ue), (a, e))[1]
            with torch.no_grad():
                _, w = self.noise.derivatives(mu.reshape(b.data.shape), b.data, b.s, b.mask)
            parts.append((sl, us, ue, vjp_fn, jvp_fn, w.flatten(1)))
        _, dr_s = (to_cpu64(t) for t in params.prior_residuals("shared"))
        _, dr_e = (to_cpu64(t) for t in params.prior_residuals("emitter"))

        def product(v_s: torch.Tensor, v_e: torch.Tensor):
            out_s, out_e = torch.zeros_like(v_s), torch.zeros_like(v_e)
            for sl, us, ue, vjp_fn, jvp_fn, w in parts:
                a, e = v_s.to(us.device, us.dtype), v_e[sl].to(ue.device, ue.dtype)
                with torch.no_grad():
                    dmu = jvp_fn(a, e)
                gs, ge = vjp_fn(w * dmu)
                out_s += to_cpu64(gs)
                out_e[sl] = to_cpu64(ge)
            return out_s + dr_s ** 2 * v_s, out_e + dr_e ** 2 * v_e

        return product

    def normal_equations(self, params) -> Normal:
        Ps, Pe, E = params.shared_vector().numel(), params.emitter_matrix().shape[1], params.n_emitters
        f64 = dict(dtype=torch.float64)
        H_ss, g_s = torch.zeros(Ps, Ps, **f64), torch.zeros(Ps, **f64)
        H_se, H_ee = torch.zeros(E, Ps, Pe, **f64), torch.zeros(E, Pe, Pe, **f64)
        g_e = torch.zeros(E, Pe, **f64)
        objective = 0.0
        for sl, b in self.chunks():
            pc = params.subset(sl)
            mu, J_s, J_e = self._jacobians(pc, b)                       # J: [P, E_c, M]
            D = self.noise.deviance(mu, b.data, b.s, b.mask)
            objective += 0.5 * float(to_cpu64(D.flatten(1).sum(1)).sum())
            grad, w = self.noise.derivatives(mu, b.data, b.s, b.mask)
            grad, w = grad.flatten(1), w.flatten(1)                     # [E_c, M]
            wJ_s, wJ_e = J_s * w, J_e * w
            # Per emitter on the device, then summed in float64: exact for any chunking.
            H_ss += to_cpu64(torch.einsum("iem,jem->eij", wJ_s, J_s)).sum(0)
            g_s += to_cpu64(torch.einsum("iem,em->ei", J_s, grad)).sum(0)
            H_se[sl] = to_cpu64(torch.einsum("iem,jem->eij", wJ_s, J_e))
            H_ee[sl] = to_cpu64(torch.einsum("iem,jem->eij", wJ_e, J_e))
            g_e[sl] = to_cpu64(torch.einsum("iem,em->ei", J_e, grad))
        r, dr = (to_cpu64(t) for t in params.prior_residuals("shared"))
        H_ss += torch.diag(dr ** 2)
        g_s += dr * r
        r, dr = (to_cpu64(t) for t in params.prior_residuals("emitter"))
        H_ee += torch.diag_embed(dr ** 2)
        g_e += dr * r
        objective += float(to_cpu64(params.prior_penalty()))
        return Normal(objective, H_ss, H_se, H_ee, g_s, g_e)

    def _jacobians(self, pc, b):
        """μ and the Jacobians of the chunk's μ, flattened per emitter: [P_s, E, M] and [P_e, E, M]."""
        us, ue = pc.shared_vector(), pc.emitter_matrix()
        E, Pe = ue.shape[0], ue.shape[1]

        def mu_of(us_, ue_):
            return self.model(pc.replace(shared=us_, emitter=ue_).physical(), b).flatten(1)

        with torch.no_grad():
            mu = mu_of(us, ue)

        def emitter_column(t):
            return torch.func.jvp(lambda v: mu_of(us, v), (ue,), (t,))[1]

        eye_e = torch.eye(Pe, dtype=ue.dtype, device=ue.device)[:, None, :].expand(Pe, E, Pe)
        with torch.no_grad():
            linear = getattr(self.model, "emitter_linearisation", None) if self.linearise else None
            lin = linear(pc.physical(), b) if linear is not None else None
            J_s = self._shared_columns(pc, b, us, ue, mu_of, mu, lin)
            if lin is not None and Pe:
                J_e = self._parts_columns(lin, lambda v: pc.replace(shared=us, emitter=v).physical(), ue,
                                          eye_e).reshape(Pe, E, -1)
            else:
                J_e = self._columns(emitter_column, eye_e, (0, E, mu.shape[1]), mu)
        return mu.reshape(b.data.shape), J_s, J_e

    @staticmethod
    def _parts_jvps(parts, physical, u, tangents):
        """The JVPs of the model's cheap components (positions, photons, background) for ``tangents`` on
        ``u``: ((xyz, N, b), (dxyz, dN, db)), the derivatives with a leading tangent axis."""
        def column(t):
            return torch.func.jvp(lambda v: parts(physical(v)), (u,), (t,))
        return torch.func.vmap(column, out_dims=(None, 0))(tangents)

    @classmethod
    def _parts_columns(cls, lin, physical, u, tangents, jvps=None):
        """Columns [T, E, C, K, Y, X] of parameters that act through the model's cheap components alone,
        from its `emitter_linearisation` (I, dI, parts): N·Σ_d (∂xyz_d/∂u)·dI_d + (∂N/∂u)·I + ∂b/∂u."""
        I, dI, parts = lin
        (xyz, N, bg), (dxyz, dN, dbg) = jvps or cls._parts_jvps(parts, physical, u, tangents)
        dpsf = torch.einsum("teckd,deckyx->teckyx", dxyz.to(dI.dtype), dI)
        return N[None, ..., None, None] * dpsf + dN[..., None, None] * I + dbg.expand(-1, *I.shape)

    def _shared_columns(self, pc, b, us, ue, mu_of, mu, lin=None):
        """[P_s, E, M], in three kinds:
        - parameters that act only through a channel's pupil field, from the model's `pupil_linearisation`:
          a JVP of the small pupil function, then one matrix DFT per column, 2.5× cheaper than a JVP
          through the whole model;
        - parameters that act only through where the emitter is and how bright (the channel transform,
          shares and offsets, a per-slice factor), from the `emitter_linearisation` the emitter block
          needs anyway: JVPs of the small components only.  With a bead kernel a slice's image depends
          on its neighbours' positions, and the linearisation is the derivative for a shift of the whole
          stack, so only columns that move every slice alike qualify there (not the drift or z scale);
        - the rest (the blur, and those) by JVPs through the whole model."""
        Ps, E, M = us.numel(), ue.shape[0], mu.shape[1]
        eye = torch.eye(Ps, dtype=us.dtype, device=us.device)
        J = mu.new_empty(Ps, E, M)
        if Ps == 0:
            return J
        done = torch.zeros(Ps, dtype=torch.bool)
        names = [None] * Ps
        for n, lo, hi in pc.layout("shared"):
            names[lo:hi] = [n] * (hi - lo)
        linear = getattr(self.model, "pupil_linearisation", None) if self.linearise else None
        if linear is not None:
            C = b.data.shape[1]
            Jc = J.view(Ps, E, C, -1)
            for c, group, field, apply in linear(pc.physical(), b):
                idx = [i for i in range(Ps) if names[i] in group]
                if not idx:
                    continue
                def field_column(t, field=field):
                    return torch.func.jvp(lambda v: field(pc.replace(shared=v, emitter=ue).physical()), (us,),
                                          (t,))[1]
                dA = torch.func.vmap(field_column)(eye[idx])
                rows = torch.tensor(idx, device=J.device)
                Jc[rows] = 0
                for i in range(0, len(idx), self.tangents):
                    Jc[rows[i:i + self.tangents], :, c] = apply(dA[i:i + self.tangents]).reshape(
                        len(idx[i:i + self.tangents]), E, -1)
                done[idx] = True
        psf_names = getattr(self.model, "psf_names", None)
        if lin is not None and psf_names is not None:
            inside = psf_names()
            idx = [i for i in range(Ps) if not done[i] and names[i] not in inside]
            if idx:
                physical = lambda v: pc.replace(shared=v, emitter=ue).physical()   # noqa: E731
                jvps = self._parts_jvps(lin[2], physical, us, eye[idx])
                dxyz = jvps[1][0]                                                    # [T, E, C, K, 3]
                if any(ch.psf.bead_radius_nm > 0 for ch in getattr(self.model, "channels", [])):
                    uniform = (dxyz == dxyz[..., :1, :]).flatten(1).all(1).cpu()
                else:
                    uniform = torch.ones(len(idx), dtype=torch.bool)
                keep = uniform.nonzero()[:, 0]
                if len(keep):
                    sel = tuple(t[keep] for t in jvps[1])
                    cols = self._parts_columns(lin, physical, us, None, (jvps[0], sel))
                    J[[idx[k] for k in keep.tolist()]] = cols.reshape(len(keep), E, -1)
                    done[[idx[k] for k in keep.tolist()]] = True
        rest = (~done).nonzero()[:, 0].tolist()
        if rest:
            def shared_column(t):
                return torch.func.jvp(lambda v: mu_of(v, ue), (us,), (t,))[1]
            J[rest] = self._columns(shared_column, eye[rest], None, None)
        return J

    def _columns(self, column, tangents, empty_shape, like):
        if tangents.shape[0] == 0:
            return like.new_zeros(empty_shape)
        parts = [torch.func.vmap(column)(tangents[i:i + self.tangents])
                 for i in range(0, tangents.shape[0], self.tangents)]
        return torch.cat(parts)
