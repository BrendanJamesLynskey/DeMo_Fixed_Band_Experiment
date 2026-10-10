"""Summarise the runs: results.md, results.json and plots. Every number in the write-up comes from here.

Reads results/<dataset>/*.jsonl, groups runs that differ only in seed, and reports the mean and
spread (sample standard deviation) over seeds.

Usage: python experiment/analyse.py --dataset tinystories
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LABEL = {"adamw": "A  AdamW, all-reduce", "topk": "B  DeMo, top-k", "band": "C  DeMo, fixed band",
         "calib": "D  DeMo, calibrated band", "optics": "E  DeMo, band via emulated optics"}


def load(dataset: str):
    runs = []
    for p in sorted((ROOT / "results" / dataset).glob("*.jsonl")):
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        if not recs or recs[-1].get("type") != "done":
            continue
        cfg = recs[0]
        runs.append({"name": cfg["name"], "group": re.sub(r"-s\d+$", "", cfg["name"]), "cfg": cfg["config"],
                     "header": cfg, "evals": [r for r in recs if r["type"] == "eval"],
                     "diags": [r for r in recs if r["type"] == "diag"], "seconds": recs[-1]["seconds"]})
    return runs


def mean_sd(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    return stats.fmean(xs), (stats.stdev(xs) if len(xs) > 1 else 0.0)


def describe(cfg: dict) -> str:
    v = cfg["variant"]
    s = LABEL[v]
    if v == "band" and cfg["band"] != "low":
        s = f"F  DeMo, fixed {cfg['band']} band"
    if v != "adamw":
        s += f", {'2-D ' + str(cfg['chunk']) + 'x' + str(cfg['chunk']) if cfg['chunk_mode'] == '2d' else '1-D ' + str(cfg['chunk']) + '-point'} chunks"
    if v == "optics":
        s += f", ENOB {cfg['enob']:g}, {cfg['planes']} planes"
        fs = cfg.get("fs_every", 0)
        s += f", full scale {'re-set every ' + str(fs) + ' steps' if fs else 'fixed'}"
        s += f", feedback of the {'band sent' if cfg.get('feedback', 'exact') == 'sent' else 'exact band'}"
    return s


def summarise(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[r["group"]].append(r)
    base = [r for r in runs if r["cfg"]["variant"] == "adamw"]
    target = mean_sd([r["evals"][-1]["val_loss"] for r in base])[0] if base else None

    rows = []
    for g, rs in sorted(groups.items()):
        cfg, hdr = rs[0]["cfg"], rs[0]["header"]
        final, final_sd = mean_sd([r["evals"][-1]["val_loss"] for r in rs])
        reach = []
        for r in rs:
            hit = next((e["step"] for e in r["evals"] if target is not None and e["val_loss"] <= target), None)
            reach.append(hit)
        reached = [x for x in reach if x is not None]
        row = dict(group=g, label=describe(cfg), variant=cfg["variant"], seeds=len(rs),
                   final_val=final, final_val_sd=final_sd,
                   payload_per_step=hdr["payload_bytes_per_step"],
                   payload_int64_per_step=hdr["payload_bytes_int64_per_step"],
                   wire_per_step=hdr["wire_bytes_per_step"],
                   ratio_vs_dense=hdr["dense_bytes_per_step"] / hdr["payload_bytes_per_step"],
                   steps_to_target=stats.fmean(reached) if len(reached) == len(rs) else None,
                   reached=f"{len(reached)}/{len(rs)}",
                   seconds=stats.fmean(r["seconds"] for r in rs))
        if row["steps_to_target"] is not None:
            row["bytes_to_target"] = row["steps_to_target"] * row["payload_per_step"]
        clips = [r["evals"][-1].get("clip_rate") for r in rs if "clip_rate" in r["evals"][-1]]
        if clips:
            row["clip_rate"] = stats.fmean(clips)
            row["clip_rate_last"] = stats.fmean(r["evals"][-1]["clip_rate_interval"] for r in rs)
        rows.append(row)
    return rows, target


def energy(runs, after_step: int):
    """Mean captured energy of worker 0's momentum, per selector, over all diagnostics after a step."""
    out = {}
    for r in runs:
        if r["cfg"]["variant"] not in ("topk", "band", "calib") or r["cfg"]["band"] not in ("low", "calib"):
            continue
        key = (r["cfg"]["variant"], r["cfg"]["chunk_mode"], r["cfg"]["chunk"])
        acc = out.setdefault(key, defaultdict(list))
        for d in r["diags"]:
            if d["step"] <= after_step:
                continue
            for src, groups in d["energy"].items():
                for grp, sel in groups.items():
                    for s, v in sel.items():
                        acc[(src, grp, s)].append(v)
    return {k: {f"{src}/{g}/{s}": stats.fmean(v) for (src, g, s), v in acc.items()} for k, acc in out.items()}


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def write(dataset: str):
    runs = load(dataset)
    if not runs:
        raise SystemExit(f"no completed runs in results/{dataset}")
    rows, target = summarise(runs)
    hdr = runs[0]["header"]
    after = runs[0]["cfg"]["warmup"]
    en = energy(runs, after)

    L = [f"# Results: {dataset}\n",
         f"Generated by `experiment/analyse.py` from {len(runs)} completed runs in `results/{dataset}/`. "
         "Every number in the write-up comes from this file.\n",
         f"Hardware: {hdr['hardware']['processor']}, {hdr['hardware']['threads']} threads, PyTorch "
         f"{hdr['hardware']['torch']}, Python {hdr['hardware']['python']}. Model: {hdr['n_params']:,} parameters, "
         f"{runs[0]['cfg']['n_layer']} layers, width {runs[0]['cfg']['n_embd']}, context {runs[0]['cfg']['block']}. "
         f"{runs[0]['cfg']['workers']} simulated workers, {runs[0]['cfg']['batch']} sequences each per step, "
         f"{runs[0]['cfg']['steps']} steps. Data: {hdr['data']['tokeniser']}.\n",
         "Bytes are per worker per step, values in BF16 (2 bytes). *Payload* = what a worker originates; top-k "
         "indices are counted at the bytes needed for a position within a chunk (the reference code sends int64, "
         "shown separately). *Wire* = bytes a worker sends in a ring collective (all-reduce for A, all-gather "
         "for the DeMo variants).\n",
         "## 1. Loss against communication\n",
         f"Target = the AdamW baseline's mean final validation loss, {fmt(target)}.\n",
         "| Configuration | Seeds | Final val loss (mean ± sd) | Payload / step (bytes) | vs dense | Steps to target | Payload to target (MB) |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        btt = r.get("bytes_to_target")
        L.append(f"| {r['label']} | {r['seeds']} | {fmt(r['final_val'])} ± {fmt(r['final_val_sd'])} | "
                 f"{r['payload_per_step']:,} | {fmt(r['ratio_vs_dense'], 1)}x | "
                 f"{fmt(r['steps_to_target'], 0) if r['steps_to_target'] else 'not reached (' + r['reached'] + ')'} | "
                 f"{fmt(btt / 1e6, 1) if btt else 'n/a'} |")
    topk = [r for r in rows if r["variant"] == "topk"]
    if topk:
        t = topk[0]
        L.append(f"\n* Top-k with int64 indices, as the reference code sends them: {t['payload_int64_per_step']:,} "
                 f"bytes per step, {fmt(t['payload_int64_per_step'] / t['payload_per_step'], 1)}x the compact count.")

    L += ["\n## 2. Captured energy on real momentum\n",
          f"Share of energy kept by each selector at the same k, on worker 0's tensors, averaged over diagnostics "
          f"after step {after}. *momentum* = a plain momentum of the same decay, never compressed; *delta* = what "
          "DeMo actually compresses (momentum plus the residual that error feedback keeps); *grad* = the step's "
          "gradient. A random set keeps about the keep fraction.\n",
          "| Run | Tensor | Group | Top-k | Low band | Zig-zag | High band | Random | Calibrated |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (variant, mode, chunk), sel in sorted(en.items()):
        for src in ("momentum", "delta", "grad"):
            for grp in ("all", "attn", "mlp", "embed"):
                if f"{src}/{grp}/topk" not in sel:
                    continue
                g = lambda s: fmt(sel.get(f"{src}/{grp}/{s}"))  # noqa: E731
                L.append(f"| {LABEL[variant][:1]}, {mode} {chunk} | {src} | {grp} | {g('topk')} | {g('low')} | "
                         f"{g('zigzag')} | {g('high')} | {g('random')} | {g('calib')} |")

    opt = [r for r in rows if r["variant"] == "optics"]
    if opt:
        L += ["\n## 3. Emulated optics: ENOB and bit planes\n",
              "Input full scale kappa x RMS of the delta, either calibrated once over the calibration window "
              "(*fixed*) or re-set from the mean square over each interval (*re-set*); saturation, exact binary bit "
              "planes, noisy readout per plane at the given ENOB, weighted recombination. Planes = optical passes per "
              "chunk. *Feedback of the exact band*: each sender's error feedback removes the exact band, so the "
              "optics' errors are never corrected; *of the band sent*: it removes what the optics produced, which "
              "it receives back in the all-gather.\n",
              "| Configuration | Final val loss | Clip rate (whole run) | Clip rate (last interval) |",
              "|---|---|---|---|"]
        for r in sorted(opt, key=lambda r: r["label"]):
            L.append(f"| {r['label']} | {fmt(r['final_val'])} | {fmt(100 * r['clip_rate'], 2)}% | "
                     f"{fmt(100 * r['clip_rate_last'], 2)}% |")

    out = ROOT / "results" / dataset
    (out / "results.md").write_text("\n".join(L) + "\n")
    (out / "results.json").write_text(json.dumps(
        {"target": target, "rows": rows, "energy": {"/".join(map(str, k)): v for k, v in en.items()}},
        indent=1, default=lambda x: round(x, 6) if isinstance(x, float) else x))
    plots(runs, rows, out)
    print("\n".join(L))


def plots(runs, rows, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups = defaultdict(list)
    for r in runs:
        groups[r["group"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for g, rs in sorted(groups.items()):
        cfg = rs[0]["cfg"]
        if cfg["variant"] == "optics" or (cfg["variant"] == "band" and cfg["band"] != "low"):
            continue
        steps = [e["step"] for e in rs[0]["evals"]]
        loss = [stats.fmean(r["evals"][i]["val_loss"] for r in rs) for i in range(len(steps))]
        byts = [e["payload_bytes"] / 1e6 for e in rs[0]["evals"]]
        axes[0].plot(steps, loss, label=describe(cfg))
        axes[1].plot(byts, loss, label=describe(cfg))
    axes[0].set_xlabel("step")
    axes[1].set_xlabel("payload per worker (MB, log)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_ylabel("validation loss")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "loss_vs_steps_and_bytes.png", dpi=120)
    plt.close(fig)

    opt = sorted((r for r in rows if r["variant"] == "optics"), key=lambda r: r["label"])
    if opt:
        fig, ax = plt.subplots(figsize=(6, 4))
        by_planes = defaultdict(list)
        for g, rs in groups.items():
            c = rs[0]["cfg"]
            if c["variant"] == "optics":
                key = (c.get("feedback", "exact"), c.get("fs_every", 0), c["planes"])
                by_planes[key].append((c["enob"], stats.fmean(r["evals"][-1]["val_loss"] for r in rs)))
        for (fb, fs, p), pts in sorted(by_planes.items()):
            pts.sort()
            ax.plot([e for e, _ in pts], [v for _, v in pts], "o-" if fb == "sent" else "x:",
                    label=f"{p} planes, feedback {fb}, full scale {'re-set' if fs else 'fixed'}")
        ax.set_yscale("log")
        ref = [r for r in rows if r["variant"] == "band" and "1-D 256" in r["label"]]
        if ref:
            ax.axhline(ref[0]["final_val"], ls="--", color="grey", label="exact band (C, 1-D 256)")
        ax.set_xlabel("ENOB per plane readout")
        ax.set_ylabel("final validation loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "enob_tolerance.png", dpi=120)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shakespeare")
    write(ap.parse_args(argv).dataset)


if __name__ == "__main__":
    main()
