"""Transforms, coefficient selectors and the optics emulation.

Everything a worker does to its momentum before it reaches the wire:

* ``Chunker`` splits a tensor into chunks and applies an orthonormal DCT-II to each one, either
  2-D (c x c chunks of a matrix, as DeMo does) or 1-D (runs of n values of the flattened tensor,
  as the photonic module's 1xN radix-4 transforms would).
* A selector picks which coefficients of each chunk to send:
  - ``topk``: the k largest by magnitude, per chunk (DeMo as published; needs comparisons);
  - a fixed band: the same k positions in every chunk, chosen before training (a gather, no
    comparisons), in one of several shapes (low, zig-zag, high, random);
  - a calibrated band: positions chosen once from the average energy map of the first K steps,
    then frozen (still comparison-free at run time).
* ``OpticsEmulator`` replaces the exact transform with what an analogue optical transform with
  binary (NRZ) input would deliver: a fixed, pre-calibrated input full scale with saturation,
  exact two's-complement bit planes, a noisy readout of each plane at a chosen ENOB, and weighted
  recombination.

Conventions: a DCT here is the orthonormal DCT-II, the same basis DeMo uses (norm="ortho").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


# ───────────────────────────────────────────────────────────── DCT ──
def dct_matrix(n: int, dtype=torch.float64) -> torch.Tensor:
    """Orthonormal DCT-II matrix D (n x n): coefficients = D @ x, and D @ D.T = I.

    Built with NumPy: torch.cos on float64 intermittently raised "Illegal instruction" on the
    AVX-only test machine (an i7-3770), and the matrix is only built once per chunk size."""
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    d = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    d[0] /= np.sqrt(2.0)
    return torch.from_numpy(d).to(dtype)


def largest_divisor_at_most(n: int, target: int) -> int:
    """The largest divisor of n that does not exceed target (DeMo's rule for chunk sides)."""
    best = 1
    for d in range(1, min(n, target) + 1):
        if n % d == 0:
            best = d
    return best


class Chunker:
    """Chunk a tensor and DCT each chunk.

    mode "2d": a matrix (R x C) is cut into h x w tiles, h and w the largest divisors of R and C
    not above ``size``; vectors fall back to 1-D runs. mode "1d": the flattened tensor is cut into
    runs of length m, the largest divisor of numel not above ``size``.

    ``encode`` returns coefficients shaped (chunks, m) with each chunk flattened row-major, so a
    position index means the same frequency in every chunk of the tensor.
    """

    def __init__(self, shape: torch.Size, mode: str, size: int, dtype=torch.float32, device=None):
        self.shape = tuple(shape)
        self.device = torch.device(device or "cpu")
        self.mode = mode
        if mode == "2d" and len(self.shape) == 2:
            r, c = self.shape
            self.h, self.w = largest_divisor_at_most(r, size), largest_divisor_at_most(c, size)
            self.dh, self.dw = dct_matrix(self.h, dtype).to(self.device), dct_matrix(self.w, dtype).to(self.device)
            self.chunk_shape = (self.h, self.w)
        elif mode in ("1d", "2d"):
            n = math.prod(self.shape)
            self.m = largest_divisor_at_most(n, size)
            self.d = dct_matrix(self.m, dtype).to(self.device)
            self.chunk_shape = (self.m,)
        else:
            raise ValueError(mode)
        self.is2d = len(self.chunk_shape) == 2
        self.m = math.prod(self.chunk_shape)

    # Every method accepts either one tensor of self.shape, or a stack of them with one leading
    # batch dimension (the workers); chunk outputs then carry the same leading dimension.
    def _lead(self, x: torch.Tensor, ndim: int) -> tuple:
        return tuple(x.shape[:1]) if x.dim() == ndim + 1 else ()

    def to_chunks(self, x: torch.Tensor) -> torch.Tensor:
        """Values grouped by chunk, (chunks, m), in the same layout as the coefficients."""
        b = self._lead(x, len(self.shape))
        if self.is2d:
            r, c = self.shape
            t = x.reshape(*b, r // self.h, self.h, c // self.w, self.w).transpose(-3, -2)
            return t.reshape(*b, -1, self.m)
        return x.reshape(*b, -1, self.m)

    def from_chunks(self, t: torch.Tensor) -> torch.Tensor:
        b = self._lead(t, 2)
        if self.is2d:
            r, c = self.shape
            t = t.reshape(*b, r // self.h, c // self.w, self.h, self.w).transpose(-3, -2)
            return t.reshape(*b, *self.shape)
        return t.reshape(*b, *self.shape)

    def transform_matrix(self) -> torch.Tensor:
        """The whole per-chunk transform as one (m x m) matrix acting on a flattened chunk."""
        if self.is2d:
            return torch.kron(self.dh, self.dw)
        return self.d

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        chunks = self.to_chunks(x)
        if self.is2d:
            t = chunks.reshape(*chunks.shape[:-1], self.h, self.w)
            return (self.dh @ t @ self.dw.T).reshape(chunks.shape)
        return chunks @ self.d.T

    def decode(self, coef: torch.Tensor) -> torch.Tensor:
        if self.is2d:
            t = coef.reshape(*coef.shape[:-1], self.h, self.w)
            return self.from_chunks((self.dh.T @ t @ self.dw).reshape(coef.shape))
        return self.from_chunks(coef @ self.d)

    @property
    def chunks(self) -> int:
        return math.prod(self.shape) // self.m


# ─────────────────────────────────────────────────────── selectors ──
def band_order(chunk_shape: tuple, shape: str, seed: int = 0) -> torch.Tensor:
    """All positions of a chunk, in the order a band of this shape takes them.

    low:    lowest frequencies first. 2-D: square shells (max(i, j), then i + j), so a band of
            k = s^2 is the s x s top-left square. 1-D: index order.
    zigzag: 2-D: anti-diagonals (i + j, then i), a triangle. 1-D: same as low.
    high:   the reverse of low.
    random: a fixed random permutation (the control).
    """
    if len(chunk_shape) == 2:
        h, w = chunk_shape
        ii, jj = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        ii, jj = ii.flatten(), jj.flatten()
        # scale to the unit square so non-square chunks still give a square-ish low band
        fi, fj = ii.double() / h, jj.double() / w
        if shape in ("low", "high"):
            keys = (torch.maximum(fi, fj), fi + fj, ii.double())
        elif shape == "zigzag":
            keys = (fi + fj, ii.double())
        elif shape == "random":
            keys = None
        else:
            raise ValueError(shape)
    else:
        (m,) = chunk_shape
        if shape not in ("low", "high", "zigzag", "random"):
            raise ValueError(shape)
        keys = None if shape == "random" else (torch.arange(m).double(),)
    n = math.prod(chunk_shape)
    if keys is None:
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(n, generator=g)
    order = torch.arange(n)
    for key in reversed(keys):                      # lexicographic: last key first, stable sorts
        order = order[torch.argsort(key[order], stable=True)]
    return order.flip(0) if shape == "high" else order


def keep_count(m: int, keep: float) -> int:
    """Coefficients kept per chunk for a keep fraction: at least 1."""
    return max(1, round(m * keep))


def topk_select(coef: torch.Tensor, k: int):
    """Per-chunk top-k by magnitude (DeMo as published): (indices, values)."""
    idx = torch.topk(coef.abs(), k=k, dim=-1, largest=True, sorted=False).indices
    return idx, torch.gather(coef, -1, idx)


def scatter_mean(chunks: int, m: int, idx_list, val_list, dtype) -> torch.Tensor:
    """Combine every worker's sparse coefficients: the mean of the contributions at each position
    (positions nobody sent stay zero). This is DeMo's aggregation (scatter_reduce, "mean")."""
    out = torch.zeros(chunks, m, dtype=dtype, device=idx_list[0].device)
    idx = torch.cat(idx_list, dim=-1)
    val = torch.cat(val_list, dim=-1)
    out.scatter_reduce_(-1, idx, val, reduce="mean", include_self=False)
    return out


def captured_energy(coef: torch.Tensor, positions: torch.Tensor | None, k: int) -> float:
    """Fraction of the coefficients' energy kept by a fixed set of positions, or by per-chunk
    top-k when positions is None. Mean over chunks, each chunk weighted by its energy (so the
    result is the share of the tensor's total energy that survives)."""
    e = coef.double() ** 2
    tot = e.sum()
    if tot == 0:
        return float("nan")
    if positions is None:
        kept = torch.topk(e, k=k, dim=-1).values.sum()
    else:
        kept = e[:, positions].sum()
    return float(kept / tot)


# ──────────────────────────────────────────────────────── optics ──
@dataclass
class OpticsConfig:
    """An analogue optical transform with binary input.

    planes:  input bits P. Each value is quantised to a P-bit two's-complement integer against a
             fixed full scale, and each bit plane (a 0/1 vector) passes through the optics
             separately: P passes per chunk. None = no input quantisation (exact input).
    enob:    effective bits of each plane's readout. Readout noise RMS = FS x 2^-ENOB x 2/sqrt(12),
             FS the readout full scale. None = exact readout.
    kappa:   input full scale = kappa x the RMS of the delta measured over the calibration steps
             (a sum of squares: MAC only, no comparisons). Values beyond it saturate.
    native:  largest transform length the module does natively; chunk sides above it are refused.
    """
    planes: int | None = 8
    enob: float | None = 8.0
    kappa: float = 4.0
    native: int = 256


class OpticsEmulator:
    """Emulate an in-transit transform of one tensor's chunks, returning only the band outputs.

    The readout full scale of a plane is fixed by the transform itself: with a 0/1 input, output
    i can reach at most sum_j |T_ij|, so FS = max_i sum_j |T_ij| is a constant known before
    training (no comparisons at run time).
    """

    def __init__(self, chunker: Chunker, positions: torch.Tensor, cfg: OpticsConfig):
        side = max(chunker.chunk_shape)
        if side > cfg.native or side & (side - 1):
            raise ValueError(f"chunk side {side} must be a power of 2 no larger than {cfg.native}")
        self.cfg = cfg
        self.ch, self.positions = chunker, positions.to(chunker.device)
        self.t_band = chunker.transform_matrix()[self.positions]           # (k, m)
        self.fs_out = float(chunker.transform_matrix().double().abs().sum(dim=1).max())
        self.scale_in: float | None = None
        self.clipped = 0
        self.total = 0

    @property
    def is_exact(self) -> bool:
        """No input quantisation and no readout noise: the optics adds nothing to the exact band."""
        return self.cfg.planes is None and self.cfg.enob is None

    def calibrate(self, rms: float) -> None:
        self.scale_in = self.cfg.kappa * rms if rms > 0 else 1.0

    def plane_noise_rms(self) -> float:
        if self.cfg.enob is None:
            return 0.0
        return self.fs_out * 2.0 ** (-self.cfg.enob) * 2.0 / math.sqrt(12.0)

    def __call__(self, chunks: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """chunks: (n, m) input values. Returns (n, k) band coefficients as the optics would."""
        x = chunks.to(self.t_band.dtype)
        p = self.cfg.planes
        if p is None:
            y = self._band(x)
            if self.cfg.enob is not None:
                y = y + torch.randn(y.shape, generator=gen, dtype=y.dtype, device=y.device) * self.plane_noise_rms()
            return y.to(chunks.dtype)
        assert self.scale_in is not None, "calibrate() first"
        qmax = 2 ** (p - 1) - 1
        q = torch.round(x / self.scale_in * qmax)
        clip = q.abs() > qmax
        self.clipped += int(clip.sum())
        self.total += q.numel()
        q = q.clamp(-qmax, qmax).to(torch.int64)
        u = q & ((1 << p) - 1)                                   # two's-complement bit patterns
        bits = torch.arange(p, device=x.device)
        planes = ((u.unsqueeze(0) >> bits.view(-1, 1, 1)) & 1).to(x.dtype)   # (P, n, m): exact binary (NRZ) input
        out = self._band(planes.reshape(-1, planes.shape[-1])).reshape(p, x.shape[0], -1)   # one pass per plane
        sigma = self.plane_noise_rms()
        if sigma:
            out = out + torch.randn(out.shape, generator=gen, dtype=out.dtype, device=out.device) * sigma
        weight = 2.0 ** bits.to(out.dtype)
        weight[-1] = -weight[-1]                                 # the MSB carries the sign
        y = (weight.view(-1, 1, 1) * out).sum(0)
        return (y * (self.scale_in / qmax)).to(chunks.dtype)

    def _band(self, x: torch.Tensor) -> torch.Tensor:
        """Band coefficients of each row of x (rows = flattened chunks). 2-D chunks use the
        separable transform, which is cheaper than the dense band matrix."""
        ch = self.ch
        if ch.is2d:
            t = x.reshape(-1, ch.h, ch.w)
            return (ch.dh @ t @ ch.dw.T).reshape(x.shape[0], -1)[:, self.positions]
        return x @ self.t_band.T

    def clip_rate(self) -> float:
        return self.clipped / self.total if self.total else 0.0


# ─────────────────────────────────────────────────── byte counting ──
VALUE_BYTES = 2          # values on the wire in BF16, for every variant


def index_bytes(m: int) -> int:
    """Bytes for one position index within a chunk of m coefficients, rounded up to whole bytes
    (12 bits for a 64 x 64 chunk -> 2 bytes). DeMo's reference code sends int64 (8 bytes)."""
    return max(1, math.ceil(math.log2(m) / 8)) if m > 1 else 1
