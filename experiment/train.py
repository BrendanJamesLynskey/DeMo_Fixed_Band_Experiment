"""Train one configuration: W simulated data-parallel workers in one process.

The workers share one set of weights. Each has its own data shard and, for the DeMo variants, its
own momentum (DeMo's "delta"). Communication is the aggregation of what each worker transmits;
bytes are counted analytically.

Variants (the same keep fraction for every compressed variant):
  A  adamw    AdamW on all-reduced gradients: the DDP baseline.
  B  topk     DeMo as published: chunked DCT, per-chunk top-k, values and indices sent.
  C  band     DeMo with a fixed band of positions (shape: low, zigzag, high, random); values only.
  D  calib    DeMo with a band chosen once from the mean energy map of the first K steps.
  E  optics   C or D, with the band coefficients computed by an emulated optical transform
              (fixed calibrated input scale, binary bit-plane input, noisy readout at an ENOB).

DeMo update, per tensor (as in Peng et al., arXiv:2411.19870, and their reference code):
  delta_w <- decay * delta_w + lr * grad_w
  q_w      = compress(DCT(delta_w))                  sent by worker w
  delta_w <- delta_w - IDCT(q_w)                      error feedback: keep what was not sent
  u        = IDCT(mean over workers of q_w, per position)
  p       <- p - lr * sign(u)

Usage:  python experiment/train.py --variant band --steps 200 --out results/smoke.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compressors as C  # noqa: E402
import data as D  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunConfig:
    dataset: str = "shakespeare"
    variant: str = "band"            # adamw | topk | band | calib | optics
    band: str = "low"                # band shape: low | zigzag | high | random (calib/optics: low or calib)
    keep: float = 1 / 16
    chunk_mode: str = "2d"           # 2d | 1d
    chunk: int = 64                  # 2d: chunk side; 1d: run length
    workers: int = 8
    batch: int = 8                   # sequences per worker per step
    block: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    steps: int = 1000
    lr: float = 1e-3
    warmup: int = 50
    decay: float = 0.999             # DeMo's compression_decay
    weight_decay: float = 0.1        # AdamW only
    calib_start: int = 50            # D and E: first step of the calibration window (after warm-up)
    calib_steps: int = 50            # D and E: steps used to choose the band / input full scale
    enob: float | None = 8.0         # E
    planes: int | None = 8           # E
    kappa: float = 4.0               # E
    native: int = 256                # E
    seed: int = 0
    eval_every: int = 50
    eval_batches: int = 8
    diag_every: int = 50
    ckpt_every: int = 100
    threads: int = 4
    device: str = "auto"             # auto: CUDA when available, else CPU
    extra: dict = field(default_factory=dict)

    def name(self) -> str:
        parts = [self.dataset, self.variant]
        if self.variant != "adamw":
            if self.variant in ("band", "calib", "optics"):
                parts.append(self.band)
            parts += [f"k{round(1 / self.keep)}", f"{self.chunk_mode}{self.chunk}"]
        if self.variant == "optics":
            parts += [f"enob{self.enob}", f"p{self.planes}"]
        parts += [f"lr{self.lr:g}", f"s{self.seed}"]
        return "-".join(str(p) for p in parts)


def lr_at(cfg: RunConfig, step: int) -> float:
    """Linear warm-up, then cosine decay to 10% (both optimisers)."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


class DeMoState:
    """Per-tensor chunkers, bands, per-worker deltas and the optics emulators."""

    def __init__(self, cfg: RunConfig, params: dict[str, torch.Tensor]):
        self.cfg = cfg
        dev = self.device = next(iter(params.values())).device
        self.ch, self.k, self.pos = {}, {}, {}
        # one stacked tensor per parameter: worker w's delta is delta[n][w]
        self.delta = {n: torch.zeros((cfg.workers, *p.shape), dtype=p.dtype, device=p.device)
                      for n, p in params.items()}
        self.energy = {}
        self.sumsq = {n: 0.0 for n in params}
        self.count = {n: 0 for n in params}
        self.emu: dict[str, C.OpticsEmulator] = {}
        self.diag_coef: dict[str, dict[str, torch.Tensor]] = {}
        # worker 0's momentum without compression or feedback, for the diagnostics only
        self.shadow = {n: torch.zeros_like(p) for n, p in params.items()}
        self.calibrated = cfg.variant not in ("calib", "optics")
        for n, p in params.items():
            # the optics takes power-of-2 sides only: a 384-vector is cut into runs of 128, not 192
            ch = C.Chunker(p.shape, cfg.chunk_mode, cfg.chunk, dtype=p.dtype, device=dev,
                           pow2=cfg.variant == "optics")
            if cfg.variant == "optics":
                C.OpticsEmulator.check(ch, cfg.native)          # fail at start, not after calibration
            self.ch[n] = ch
            self.k[n] = C.keep_count(ch.m, cfg.keep)
            shape = cfg.band if cfg.band in ("low", "zigzag", "high", "random") else "low"
            self.pos[n] = C.band_order(ch.chunk_shape, shape, seed=cfg.seed)[: self.k[n]].to(dev)
            self.energy[n] = torch.zeros(ch.m, dtype=torch.float64, device=dev)

    def payload_bytes(self) -> int:
        """Bytes each worker originates per step."""
        tot = 0
        for n, ch in self.ch.items():
            per = C.VALUE_BYTES + (C.index_bytes(ch.m) if self.cfg.variant == "topk" else 0)
            tot += ch.chunks * self.k[n] * per
        return tot

    def payload_bytes_int64(self) -> int:
        """As the reference code counts top-k: int64 indices."""
        return sum(ch.chunks * self.k[n] * (C.VALUE_BYTES + 8) for n, ch in self.ch.items())

    def finish_calibration(self) -> None:
        cfg = self.cfg
        for n, ch in self.ch.items():
            if cfg.band == "calib":
                # chosen once, offline: the k positions with the most mean energy
                self.pos[n] = torch.topk(self.energy[n], self.k[n]).indices.sort().values
            if cfg.variant == "optics":
                emu = C.OpticsEmulator(ch, self.pos[n], C.OpticsConfig(cfg.planes, cfg.enob, cfg.kappa, cfg.native))
                emu.calibrate(math.sqrt(self.sumsq[n] / max(1, self.count[n])))
                self.emu[n] = emu
        self.calibrated = True

    def state(self):
        return {"delta": self.delta, "shadow": self.shadow, "energy": self.energy, "sumsq": self.sumsq, "count": self.count,
                "pos": self.pos, "calibrated": self.calibrated,
                "emu": {n: (e.scale_in, e.clipped, e.total) for n, e in self.emu.items()}}

    def load(self, s):
        self.delta, self.energy, self.sumsq, self.count = s["delta"], s["energy"], s["sumsq"], s["count"]
        self.shadow = s["shadow"]
        # checkpoints written before the workers were batched hold a list of tensors per parameter
        self.delta = {n: torch.stack(v) if isinstance(v, list) else v for n, v in self.delta.items()}
        self.pos, self.calibrated = s["pos"], s["calibrated"]
        cfg = self.cfg
        for n, (scale, clipped, total) in s["emu"].items():
            e = C.OpticsEmulator(self.ch[n], self.pos[n], C.OpticsConfig(cfg.planes, cfg.enob, cfg.kappa, cfg.native))
            e.scale_in, e.clipped, e.total = scale, clipped, total
            self.emu[n] = e


@torch.no_grad()
def demo_step(st: DeMoState, params: dict, grads: dict, lr: float, step: int, gen: torch.Generator,
              diag: bool = False):
    """One DeMo step for every tensor. grads[n] stacks the workers' gradients, (W, *shape); all
    workers are processed together (the same arithmetic as a loop over workers)."""
    cfg = st.cfg
    W = cfg.workers
    calibrating = not st.calibrated and step >= cfg.calib_start
    for n, p in params.items():
        ch, k = st.ch[n], st.k[n]
        d, g = st.delta[n], grads[n]                                 # (W, *shape)
        d.mul_(cfg.decay).add_(g, alpha=lr)
        coef = ch.encode(d)                                          # (W, chunks, m)
        st.shadow[n].mul_(cfg.decay).add_(g[0], alpha=lr)
        if diag:
            st.diag_coef[n] = {"delta": coef[0].clone(),                 # what DeMo compresses
                               "momentum": ch.encode(st.shadow[n]),      # plain momentum
                               "grad": ch.encode(g[0])}
        if calibrating:
            st.energy[n] += (coef.double() ** 2).sum((0, 1))
            st.sumsq[n] += float((d.double() ** 2).sum())
            st.count[n] += d.numel()
        if cfg.variant == "topk":
            idx, val = C.topk_select(coef, k)                        # (W, chunks, k)
        else:
            idx = st.pos[n].expand(W, ch.chunks, k)
            val = coef[..., st.pos[n]]
        # what reaches the wire: in E, the band as the emulated optics computes it
        if cfg.variant == "optics" and st.calibrated and not st.emu[n].is_exact:
            tx = st.emu[n](ch.to_chunks(d).reshape(W * ch.chunks, ch.m), gen).reshape(W, ch.chunks, k)
        else:
            tx = val
        # error feedback: each sender removes the exact coefficients it sent (in E it cannot see
        # the optics' noise, so the noise is not fed back)
        d.sub_(ch.decode(torch.zeros_like(coef).scatter_(-1, idx, val)))
        # every worker's contributions to a chunk, side by side: worker 0's k, then worker 1's ...
        agg = C.scatter_mean(ch.chunks, ch.m, [idx.transpose(0, 1).reshape(ch.chunks, W * k)],
                             [tx.transpose(0, 1).reshape(ch.chunks, W * k)], p.dtype)
        p.add_(ch.decode(agg).sign(), alpha=-lr)
    if calibrating and step + 1 >= cfg.calib_start + cfg.calib_steps:
        st.finish_calibration()


@torch.no_grad()
def diagnostics(st: DeMoState, params: dict) -> dict:
    """Energy captured by top-k and by each band shape (same k), on three tensors of worker 0:

    delta:    what DeMo compresses this step (momentum plus the residual error feedback keeps);
    momentum: a plain momentum of the same decay, never compressed (the "real momentum" question);
    grad:     this step's gradient.
    Captured at this step, before compression, by demo_step."""
    return {src: _captured(st, params, src) for src in ("delta", "momentum", "grad")}


def _captured(st: DeMoState, params: dict, src: str) -> dict:
    out: dict[str, dict] = {}
    groups = {"all": [], "attn": [], "mlp": [], "embed": []}
    for n in params:
        if st.ch[n].is2d or st.cfg.chunk_mode == "1d":
            g = "embed" if n.startswith(("wte", "wpe")) else "attn" if ".attn." in n else "mlp" if (".fc." in n or ".out." in n) else None
            groups["all"].append(n)
            if g:
                groups[g].append(n)
    for gname, names in groups.items():
        if not names:
            continue
        tot = {s: 0.0 for s in ("topk", "low", "zigzag", "high", "random", "calib")}
        energy = 0.0
        for n in names:
            ch, k = st.ch[n], st.k[n]
            coef = st.diag_coef[n][src]
            e = float((coef.double() ** 2).sum())
            if e == 0:
                continue
            energy += e
            tot["topk"] += e * C.captured_energy(coef, None, k)
            for shape in ("low", "zigzag", "high", "random"):
                pos = C.band_order(ch.chunk_shape, shape, seed=st.cfg.seed)[:k].to(st.device)
                tot[shape] += e * C.captured_energy(coef, pos, k)
            if st.calibrated and st.cfg.band == "calib":
                tot["calib"] += e * C.captured_energy(coef, st.pos[n], k)
        if energy:
            out[gname] = {s: v / energy for s, v in tot.items() if v or s != "calib"}
    return out


def evaluate(model, batches) -> float:
    model.eval()
    with torch.no_grad():
        loss = sum(float(model(x, y)) for x, y in batches) / len(batches)
    model.train()
    return loss


def run(cfg: RunConfig, out: Path, max_seconds: float | None = None) -> Path:
    dev = resolve_device(cfg.device)
    torch.set_num_threads(cfg.threads)
    if dev.type == "cuda":
        # cuBLAS needs this for reproducible matmuls; scatter_reduce has no deterministic CUDA
        # kernel, so on a GPU the setting warns instead of failing (runs are close, not bit-equal)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
        warnings.filterwarnings("ignore", message=".*defaults to a non-deterministic algorithm.*")
    else:
        torch.use_deterministic_algorithms(True)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT / "results" / "ckpt" / (out.stem + ".pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, vocab, meta = D.load(cfg.dataset)
    shards = D.Shards(train_ids, cfg.workers, cfg.block, cfg.batch, cfg.seed)
    vbatches = [(x.to(dev), y.to(dev)) for x, y in D.val_batches(val_ids, cfg.block, cfg.batch, cfg.eval_batches)]
    model = GPT(GPTConfig(vocab, cfg.block, cfg.n_layer, cfg.n_head, cfg.n_embd), seed=cfg.seed).to(dev)
    params = dict(model.named_parameters())
    n_params = model.n_params()
    W = cfg.workers

    opt = None
    st = None
    if cfg.variant == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
        payload = n_params * C.VALUE_BYTES
        wire = 2 * (W - 1) / W * payload                    # ring all-reduce, per worker
        payload64 = payload
    else:
        st = DeMoState(cfg, {n: p.detach() for n, p in params.items()})
        payload = st.payload_bytes()
        wire = (W - 1) * payload                            # ring all-gather, per worker
        payload64 = st.payload_bytes_int64() if cfg.variant == "topk" else payload
    gen = torch.Generator(device=dev).manual_seed(cfg.seed + 777)
    gbuf = {n: torch.zeros((W, *p.shape), dtype=p.dtype, device=dev) for n, p in params.items()}

    start, train_acc, train_n, elapsed = 0, 0.0, 0, 0.0
    if ckpt.exists():
        # load on the CPU: generator states must stay CPU byte tensors; load_state_dict moves the
        # model and optimiser state to the parameters' device, and the DeMo state is moved here
        s = torch.load(ckpt, weights_only=False, map_location="cpu")
        model.load_state_dict(s["model"])
        if opt:
            opt.load_state_dict(s["opt"])
        if st:
            st.load(_to_device(s["demo"], dev))
        shards.set_state(s["shards"])
        gen.set_state(s["gen"])
        start, train_acc, train_n, elapsed = s["step"], s["train_acc"], s["train_n"], s["elapsed"]
        # drop log lines written after the checkpoint
        lines = [ln for ln in out.read_text().splitlines() if json.loads(ln).get("step", -1) <= start
                 or json.loads(ln)["type"] == "config"]
        out.write_text("\n".join(lines) + "\n")
    else:
        header = {"type": "config", "name": cfg.name(), "config": dataclasses.asdict(cfg), "n_params": n_params,
                  "data": meta, "payload_bytes_per_step": payload, "payload_bytes_int64_per_step": payload64,
                  "wire_bytes_per_step": wire, "dense_bytes_per_step": n_params * C.VALUE_BYTES,
                  "hardware": {"machine": platform.machine(), "processor": platform.processor() or _cpu_name(),
                               "threads": cfg.threads, "torch": torch.__version__, "python": platform.python_version(),
                               "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}}
        out.write_text(json.dumps(header) + "\n")

    clip_mark = (0, 0)
    log = open(out, "a")
    t0 = time.time() - elapsed
    session_start = time.time()
    for step in range(start, cfg.steps):
        lr = lr_at(cfg, step)
        loss_sum = 0.0
        for w in range(W):
            x, y = (t.to(dev) for t in shards.batch_for(w))
            model.zero_grad(set_to_none=True)
            loss = model(x, y)
            loss.backward()
            loss_sum += float(loss.detach())
            for n, p in params.items():
                gbuf[n][w].copy_(p.grad)
        train_acc += loss_sum / W
        train_n += 1
        if opt:
            for g in opt.param_groups:
                g["lr"] = lr
            for n, p in params.items():
                p.grad = gbuf[n].mean(0)
            opt.step()
        else:
            demo_step(st, {n: p.detach() for n, p in params.items()}, gbuf, lr, step, gen,
                      diag=(step + 1) % cfg.diag_every == 0)

        done = step + 1
        if st and done % cfg.diag_every == 0:
            log.write(json.dumps({"type": "diag", "step": done, "energy": diagnostics(st, params)}) + "\n")
            st.diag_coef.clear()
        if done % cfg.eval_every == 0 or done == cfg.steps:
            rec = {"type": "eval", "step": done, "val_loss": evaluate(model, vbatches),
                   "train_loss": train_acc / train_n, "lr": lr,
                   "payload_bytes": payload * done, "wire_bytes": wire * done,
                   "seconds": time.time() - t0}
            if st and st.emu:
                clipped, total = sum(e.clipped for e in st.emu.values()), sum(e.total for e in st.emu.values())
                rec["clip_rate"] = clipped / max(1, total)
                rec["clip_rate_interval"] = (clipped - clip_mark[0]) / max(1, total - clip_mark[1])
                clip_mark = (clipped, total)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            train_acc, train_n = 0.0, 0
        if done % cfg.ckpt_every == 0 and done < cfg.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt else None,
                        "demo": st.state() if st else None, "shards": shards.state(), "gen": gen.get_state(),
                        "step": done, "train_acc": train_acc, "train_n": train_n,
                        "elapsed": time.time() - t0}, ckpt)
            if max_seconds is not None and time.time() - session_start > max_seconds:
                log.close()
                return out
    log.write(json.dumps({"type": "done", "step": cfg.steps, "seconds": time.time() - t0}) + "\n")
    log.close()
    ckpt.unlink(missing_ok=True)
    return out


def _to_device(obj, dev: torch.device):
    """Move every tensor in a nested structure of dicts, lists and tuples to a device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: _to_device(v, dev) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, dev) for v in obj)
    return obj


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _cpu_name() -> str:
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(RunConfig):
        if f.name == "extra":
            continue
        typ = {"float | None": float, "int | None": int}.get(str(f.type), None) or type(f.default)
        ap.add_argument("--" + f.name.replace("_", "-"), type=typ, default=f.default)
    ap.add_argument("--out", type=Path, default=None)
    a = vars(ap.parse_args(argv))
    out = a.pop("out")
    cfg = RunConfig(**a)
    out = out or ROOT / "results" / (cfg.name() + ".jsonl")
    run(cfg, out)
    print(out)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
