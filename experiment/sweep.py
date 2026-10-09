"""Run a grid of configurations, one at a time, resumably.

Each run executes in its own process, so a crash (this CPU has thrown "Illegal instruction" from
PyTorch kernels) costs at most the steps since the last checkpoint: the run is retried and resumes
from it. A run whose results/<name>.jsonl ends in a "done" record is skipped. With --commit, each
completed run's results are committed to git as soon as it finishes.

Usage:
  python experiment/sweep.py --grid smoke              # list with --dry-run first
  python experiment/sweep.py --grid main --commit
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import ROOT, RunConfig, run  # noqa: E402

RESULTS = ROOT / "results"

# Model and run length for each dataset, sized to the test machine (see README, "Hardware").
SIZES = {
    "synthetic": dict(dataset="synthetic", n_layer=1, n_head=2, n_embd=64, block=64, batch=2, workers=2,
                      steps=20, warmup=4, calib_start=4, calib_steps=4, eval_every=10, diag_every=10,
                      ckpt_every=10, eval_batches=2),
    "shakespeare": dict(dataset="shakespeare", n_layer=4, n_head=4, n_embd=128, block=128, batch=8,
                        steps=600, warmup=50, calib_start=50, calib_steps=50, eval_every=50, diag_every=50,
                        ckpt_every=100),
    "tinystories": dict(dataset="tinystories", n_layer=6, n_head=6, n_embd=192, block=256, batch=4,
                        steps=1500, warmup=100, calib_start=100, calib_steps=50, eval_every=100,
                        diag_every=100, ckpt_every=100),
}

LR = {"adamw": 3e-3, "demo": 1e-3}


def grid(name: str, dataset: str, seeds=(0, 1, 2)) -> list[RunConfig]:
    base = SIZES[dataset]

    def cfg(variant, seed, **kw):
        lr = LR["adamw" if variant == "adamw" else "demo"]
        return RunConfig(variant=variant, seed=seed, lr=lr, **{**base, **kw})

    keep = 1 / 16
    runs: list[RunConfig] = []
    if name == "ci":
        return [cfg(v, 0, keep=keep, band="calib" if v == "calib" else "low")
                for v in ("adamw", "topk", "band", "calib", "optics")]
    if name in ("main", "core"):
        for s in seeds:
            runs += [cfg("adamw", s),
                     cfg("topk", s, keep=keep),                                  # B
                     cfg("band", s, keep=keep, band="low"),                      # C, DeMo's 2-D chunks
                     cfg("calib", s, keep=keep, band="calib"),                   # D
                     cfg("band", s, keep=keep, band="low", chunk_mode="1d", chunk=256)]   # C, 1-D (module)
    if name in ("main", "optics"):
        for enob in (6.0, 8.0, 10.0, 12.0):                                      # E: ENOB tolerance
            runs.append(cfg("optics", 0, keep=keep, band="low", chunk_mode="1d", chunk=256, enob=enob, planes=8))
        for planes in (4, 12):                                                   # E: planes
            runs.append(cfg("optics", 0, keep=keep, band="low", chunk_mode="1d", chunk=256, enob=8.0, planes=planes))
    if name in ("main", "shapes"):
        for shape in ("zigzag", "high", "random"):                               # F
            runs.append(cfg("band", 0, keep=keep, band=shape))
        runs.append(cfg("band", 0, keep=keep, band="low", chunk_mode="1d", chunk=64))
    if name == "smoke":
        runs = [cfg(v, 0, keep=keep, band="calib" if v == "calib" else "low", steps=30, eval_every=30,
                    diag_every=30, ckpt_every=1000) for v in ("adamw", "topk", "band", "calib", "optics")]
    return runs


def is_done(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text().strip().splitlines()
    return bool(lines) and json.loads(lines[-1]).get("type") == "done"


def commit(paths: list[Path], message: str) -> None:
    subprocess.run(["git", "add", "--", *map(str, paths)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=ROOT, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="smoke")
    ap.add_argument("--dataset", default="shakespeare", choices=sorted(SIZES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--one", help=argparse.SUPPRESS)          # internal: run one JSON config
    a = ap.parse_args(argv)

    if a.one:
        cfg = RunConfig(**json.loads(a.one))
        run(cfg, RESULTS / a.dataset / (cfg.name() + ".jsonl"))
        return

    runs = grid(a.grid, a.dataset)
    todo = [c for c in runs if not is_done(RESULTS / a.dataset / (c.name() + ".jsonl"))]
    print(f"{len(runs)} runs in grid '{a.grid}' on {a.dataset}; {len(todo)} to do")
    if a.dry_run:
        for c in todo:
            print("  ", c.name(), c.steps, "steps")
        return
    for c in todo:
        out = RESULTS / a.dataset / (c.name() + ".jsonl")
        for attempt in range(a.retries + 1):
            t = time.time()
            r = subprocess.run([sys.executable, __file__, "--dataset", a.dataset, "--one",
                                json.dumps(dataclasses.asdict(c))], cwd=ROOT)
            if is_done(out):
                print(f"done {c.name()} in {time.time() - t:.0f} s", flush=True)
                break
            print(f"run {c.name()} exited with {r.returncode}; resuming (attempt {attempt + 1})", flush=True)
        else:
            print(f"giving up on {c.name()}", flush=True)
            continue
        if a.commit:
            commit([out], f"Results: {a.dataset} {c.name()}")


if __name__ == "__main__":
    main()
