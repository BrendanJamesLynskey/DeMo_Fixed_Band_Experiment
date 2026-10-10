"""The CPU check behind the optics fix (TinyShakespeare, 600 steps, 1-D 256-point chunks, keep 1/16).

Seven runs into results/shakespeare-optics-check/, each skipped if already done:
  band                 C, the exact band (the reference)
  fixed                E as first designed: full scale fixed at calibration, feedback of the exact band
  agc                  ... full scale re-set every 50 steps
  agc_enob12           ... and ENOB 12
  agc_nonoise          ... and no readout noise (8-plane quantisation and clipping only)
  agc_fbsent_enob8     re-set full scale and feedback of the band sent, ENOB 8
  fixed_fbsent_enob8   feedback of the band sent with the full scale fixed, ENOB 8
All optics runs use 8 planes and ENOB 8 unless stated.

Usage: python scripts/optics_feedback_check.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment"))
import sweep  # noqa: E402
import train  # noqa: E402
from sweep import is_done  # noqa: E402

OUT = train.ROOT / "results" / "shakespeare-optics-check"
BASE = {**sweep.SIZES["shakespeare"], "chunk_mode": "1d", "chunk": 256, "keep": 1 / 16, "lr": 1e-3,
        "threads": 3, "diag_every": 1000}
OPT = dict(variant="optics", enob=8.0, planes=8)
RUNS = {
    "band": dict(variant="band"),
    "fixed": OPT,
    "agc": {**OPT, "fs_every": 50},
    "agc_enob12": {**OPT, "fs_every": 50, "enob": 12.0},
    "agc_nonoise": {**OPT, "fs_every": 50, "enob": None},
    "agc_fbsent_enob8": {**OPT, "fs_every": 50, "feedback": "sent"},
    "fixed_fbsent_enob8": {**OPT, "feedback": "sent"},
}

WHAT = {
    "band": "C, exact band (reference)",
    "fixed": "E, full scale fixed, feedback of the exact band (as first designed)",
    "agc": "E, full scale re-set every 50 steps, feedback of the exact band",
    "agc_enob12": "E, as above, ENOB 12",
    "agc_nonoise": "E, as above, no readout noise",
    "agc_fbsent_enob8": "E, full scale re-set every 50 steps, feedback of the band sent",
    "fixed_fbsent_enob8": "E, full scale fixed, feedback of the band sent",
}
SHOW = (100, 200, 300, 400, 500, 600)

evals = {}
for name, kw in RUNS.items():
    f = OUT / f"{name}.jsonl"
    if not is_done(f):
        train.run(train.RunConfig(**{**BASE, **kw}), f)
    evals[name] = {r["step"]: r for r in map(json.loads, f.read_text().splitlines()) if r["type"] == "eval"}

L = ["# Optics feedback check (TinyShakespeare, CPU)\n",
     "Written by `scripts/optics_feedback_check.py`. 4 layers, width 128, 8 workers, 600 steps, 1-D 256-point "
     "chunks, keep 1/16, learning rate 1e-3, calibration over steps 50 to 99. Optics: 8 planes and ENOB 8 unless "
     "stated, kappa 4. Validation loss, with the clip rate over the preceding interval in brackets.\n",
     "| Run | " + " | ".join(f"step {s}" for s in SHOW) + " |",
     "|---|" + "---|" * len(SHOW)]
for name, ev in evals.items():
    cells = []
    for s in SHOW:
        r = ev[s]
        c = r.get("clip_rate_interval")
        cells.append(f"{r['val_loss']:.3f}" + (f" ({100 * c:.1f}%)" if c is not None and s > BASE["calib_start"] + BASE["calib_steps"] else ""))
    L.append(f"| {WHAT[name]} | " + " | ".join(cells) + " |")
(OUT / "results.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
