"""Run a grid of configurations, one at a time, resumably.

Each run executes in its own process, so a crash (this CPU has thrown "Illegal instruction" from
PyTorch kernels) costs at most the steps since the last checkpoint: the run is retried and resumes
from it. A run whose results/<name>.jsonl ends in a "done" record is skipped. With --commit, each
completed run's results are committed to git as soon as it finishes.

The "all" grid is the whole experiment, unattended: a learning-rate sweep (short runs, kept in
results/<dataset>-lr/), an automatic choice of the rate with the lowest final validation loss for
each optimiser (results/<dataset>/lr_choice.json), then the main grid at those rates.

Usage:
  python experiment/sweep.py --grid smoke              # list with --dry-run first
  python experiment/sweep.py --grid all --dataset tinystories --commit --push
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
    # GPU profile: the directions' size (about 16M parameters) and about 49M tokens per run
    "tinystories-gpu": dict(dataset="tinystories", n_layer=8, n_head=6, n_embd=384, block=256, batch=8,
                            steps=3000, warmup=200, calib_start=200, calib_steps=100, eval_every=200,
                            diag_every=200, ckpt_every=500, threads=2),
}

LR = {"adamw": 3e-3, "demo": 1e-3}                  # defaults, replaced by lr_choice.json when present
LR_GRID = {"adamw": (1e-3, 3e-3), "demo": (3e-4, 1e-3, 3e-3)}
LR_STEPS = 400


def chosen_lr(dataset: str) -> dict:
    f = RESULTS / dataset / "lr_choice.json"
    return {**LR, **json.loads(f.read_text())} if f.exists() else dict(LR)


def grid(name: str, dataset: str, seeds=(0, 1, 2)) -> list[RunConfig]:
    base = SIZES[dataset]
    lrs = chosen_lr(dataset)

    def cfg(variant, seed, **kw):
        lr = lrs["adamw" if variant == "adamw" else "demo"]
        return RunConfig(variant=variant, seed=seed, **{"lr": lr, **base, **kw})

    if name == "lr":
        short = dict(steps=LR_STEPS, warmup=50, calib_start=50, eval_every=100, diag_every=100, ckpt_every=100)
        if dataset.endswith("-gpu"):
            short = dict(steps=800, warmup=100, calib_start=100, eval_every=200, diag_every=200, ckpt_every=200)
        return ([cfg("adamw", 0, lr=lr, **short) for lr in LR_GRID["adamw"]] +
                [cfg("topk", 0, lr=lr, keep=1 / 16, **short) for lr in LR_GRID["demo"]])

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
        # first as designed (full scale fixed at calibration, feedback of the exact band), which
        # diverged; then with the full scale re-set every 100 steps and feedback of the band sent
        for fix in (dict(), dict(fs_every=100, feedback="sent")):
            for enob in (6.0, 8.0, 10.0, 12.0):                                  # E: ENOB tolerance
                runs.append(cfg("optics", 0, keep=keep, band="low", chunk_mode="1d", chunk=256, enob=enob, planes=8, **fix))
            for planes in (4, 12):                                               # E: planes
                runs.append(cfg("optics", 0, keep=keep, band="low", chunk_mode="1d", chunk=256, enob=8.0, planes=planes, **fix))
        # which of the two changes matters: feedback of the band sent with the full scale fixed
        runs.append(cfg("optics", 0, keep=keep, band="low", chunk_mode="1d", chunk=256, enob=8.0, planes=8, feedback="sent"))
    if name in ("main", "shapes"):
        for shape in ("zigzag", "high", "random"):                               # F
            runs.append(cfg("band", 0, keep=keep, band=shape))
        runs.append(cfg("band", 0, keep=keep, band="low", chunk_mode="1d", chunk=64))
    if name == "smoke":
        runs = [cfg(v, 0, keep=keep, band="calib" if v == "calib" else "low", steps=30, eval_every=30,
                    diag_every=30, ckpt_every=1000) for v in ("adamw", "topk", "band", "calib", "optics")]
    return runs


def pick_lr(dataset: str) -> dict:
    """The rate with the lowest final validation loss in the learning-rate sweep, per optimiser."""
    best: dict[str, tuple[float, float]] = {}
    for p in (RESULTS / f"{dataset}-lr").glob("*.jsonl"):
        recs = [json.loads(ln) for ln in p.read_text().splitlines()]
        if recs[-1].get("type") != "done":
            continue
        c = recs[0]["config"]
        key = "adamw" if c["variant"] == "adamw" else "demo"
        loss = [r for r in recs if r["type"] == "eval"][-1]["val_loss"]
        if key not in best or loss < best[key][1]:
            best[key] = (c["lr"], loss)
    choice = {k: v[0] for k, v in best.items()}
    (RESULTS / dataset).mkdir(parents=True, exist_ok=True)
    (RESULTS / dataset / "lr_choice.json").write_text(json.dumps(
        {**choice, "_val_loss": {k: v[1] for k, v in best.items()}}, indent=1) + "\n")
    return choice


def is_done(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text().strip().splitlines()
    return bool(lines) and json.loads(lines[-1]).get("type") == "done"


def commit(paths: list[Path], message: str, push: bool = False) -> None:
    subprocess.run(["git", "add", "--", *map(str, paths)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=ROOT, check=True)
    if push:
        # Another machine may have pushed results meanwhile: rebase onto it and retry once. A failed
        # push (network) must not stop the sweep; the next push carries this commit too.
        if subprocess.run(["git", "push", "-q", "origin", "HEAD"], cwd=ROOT).returncode:
            if subprocess.run(["git", "pull", "-q", "--rebase", "origin", "main"], cwd=ROOT).returncode == 0:
                subprocess.run(["git", "push", "-q", "origin", "HEAD"], cwd=ROOT)


def run_grid(runs: list[RunConfig], outdir: str, a) -> None:
    todo = [c for c in runs if not is_done(RESULTS / outdir / (c.name() + ".jsonl"))]
    print(f"{len(runs)} runs for results/{outdir}; {len(todo)} to do", flush=True)
    if a.dry_run:
        for c in todo:
            print("  ", c.name(), c.steps, "steps")
        return
    for c in todo:
        out = RESULTS / outdir / (c.name() + ".jsonl")
        for attempt in range(a.retries + 1):
            t = time.time()
            r = subprocess.run([sys.executable, __file__, "--outdir", outdir, "--one",
                                json.dumps(dataclasses.asdict(c))], cwd=ROOT)
            if is_done(out):
                print(f"done {c.name()} in {time.time() - t:.0f} s", flush=True)
                break
            print(f"run {c.name()} exited with {r.returncode}; resuming (attempt {attempt + 1})", flush=True)
        else:
            print(f"giving up on {c.name()}", flush=True)
            continue
        if a.commit:
            commit([out], f"Results: {outdir} {c.name()}", a.push)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="smoke")
    ap.add_argument("--dataset", default="shakespeare", choices=sorted(SIZES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--outdir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--one", help=argparse.SUPPRESS)          # internal: run one JSON config
    a = ap.parse_args(argv)

    if a.one:
        cfg = RunConfig(**json.loads(a.one))
        run(cfg, RESULTS / a.outdir / (cfg.name() + ".jsonl"))
        return

    if a.grid == "all":
        run_grid(grid("lr", a.dataset), f"{a.dataset}-lr", a)
        if a.dry_run:
            return
        choice = pick_lr(a.dataset)
        print(f"learning rates chosen: {choice}", flush=True)
        if a.commit:
            commit([RESULTS / a.dataset / "lr_choice.json"], f"Results: {a.dataset} learning-rate choice", a.push)
        for stage in ("core", "optics", "shapes"):
            run_grid(grid(stage, a.dataset), a.dataset, a)
        return
    run_grid(grid(a.grid, a.dataset), a.dataset if a.grid != "lr" else f"{a.dataset}-lr", a)


if __name__ == "__main__":
    main()
