"""The DeMo update against the reference code, variant equivalences, determinism and resume."""

import dataclasses
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

import train as T

REF = Path(os.environ.get("DEMO_REF", Path(__file__).resolve().parents[2] / "_demo_ref_src" / "DeMo" / "demo.py"))


def _grads(params, workers, seed):
    """Stacked per-worker gradients, {name: (workers, *shape)}."""
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn((workers, *p.shape), generator=g) for n, p in params.items()}


@pytest.mark.skipif(not REF.exists(), reason="reference DeMo code not checked out (set DEMO_REF)")
def test_topk_variant_matches_the_reference_demo_update():
    """One worker, three steps: our B variant against bloc97/DeMo's optimizer (used, not copied:
    that repository has no licence, so it is only loaded here when a local checkout exists)."""
    pytest.importorskip("einops")
    spec = importlib.util.spec_from_file_location("demo_ref", REF)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    torch.manual_seed(0)
    shapes = {"w": (128, 192), "v": (128,)}
    init = {n: torch.randn(s) for n, s in shapes.items()}
    lr = 1e-3

    ref_params = [torch.nn.Parameter(init[n].clone()) for n in shapes]
    opt = ref.DeMo(ref_params, compression_decay=0.999, compression_topk=32, compression_chunk=64, lr=lr)
    opt._demo_all_gather = lambda idx, val: ([idx], [val])      # a single worker, no process group

    cfg = T.RunConfig(variant="topk", keep=32 / 4096, chunk=64, workers=1, decay=0.999)
    ours = {n: init[n].clone() for n in shapes}
    st = T.DeMoState(cfg, ours)
    # the 128-vector is chunked in runs of 64 by both: keep 32/4096 of 64 rounds to 1 here, so
    # match the reference's per-chunk k = 32 explicitly
    st.k["v"] = 32
    for step in range(3):
        grads = _grads(ours, 1, seed=step)
        for p, n in zip(ref_params, shapes):
            p.grad = grads[n][0].clone()
        opt.step()
        T.demo_step(st, ours, grads, lr, step, torch.Generator())
        for p, n in zip(ref_params, shapes):
            assert torch.allclose(p.detach(), ours[n], atol=1e-6), (step, n)
            assert torch.allclose(opt.demo_state[p]["delta"], st.delta[n][0], atol=1e-6), (step, n)


def test_optics_with_exact_input_and_readout_is_bit_identical_to_the_band():
    shapes = {"w": (128, 128), "v": (128,)}
    init = {n: torch.randn(s, generator=torch.Generator().manual_seed(5)) for n, s in shapes.items()}
    base = T.RunConfig(variant="band", band="low", workers=4, calib_start=1, calib_steps=2)
    exact = dataclasses.replace(base, variant="optics", enob=None, planes=None)
    a, b = {n: v.clone() for n, v in init.items()}, {n: v.clone() for n, v in init.items()}
    sa, sb = T.DeMoState(base, a), T.DeMoState(exact, b)
    for step in range(5):
        grads = _grads(a, 4, seed=10 + step)
        T.demo_step(sa, a, grads, 1e-3, step, torch.Generator())
        T.demo_step(sb, b, grads, 1e-3, step, torch.Generator())
    assert sb.emu and all(e.is_exact for e in sb.emu.values())
    for n in shapes:
        assert torch.equal(a[n], b[n])


def test_band_payload_has_no_index_bytes():
    params = {"w": torch.zeros(128, 192), "v": torch.zeros(128)}
    band = T.DeMoState(T.RunConfig(variant="band", chunk=64), params)
    topk = T.DeMoState(T.RunConfig(variant="topk", chunk=64), params)
    # w: 6 chunks x 256 kept; v: 2 runs of 64 x 4 kept
    assert band.payload_bytes() == (6 * 256 + 2 * 4) * 2
    assert topk.payload_bytes() == 6 * 256 * (2 + 2) + 2 * 4 * (2 + 1)
    assert topk.payload_bytes_int64() == (6 * 256 + 2 * 4) * (2 + 8)


def _tiny(**kw):
    return T.RunConfig(dataset="synthetic", workers=2, batch=2, block=64, n_layer=1, n_head=2, n_embd=64,
                       steps=4, eval_every=2, eval_batches=1, diag_every=2, ckpt_every=2, threads=1, **kw)


def _records(path):
    recs = [json.loads(ln) for ln in path.read_text().splitlines()]
    for r in recs:
        r.pop("seconds", None)
        r.pop("hardware", None)
    return recs


@pytest.mark.parametrize("variant", ["adamw", "topk", "band", "calib", "optics"])
def test_short_run_is_deterministic(tmp_path, variant):
    cfg = _tiny(variant=variant, calib_start=0, calib_steps=2, band="calib" if variant == "calib" else "low")
    a = T.run(cfg, tmp_path / "a.jsonl")
    b = T.run(cfg, tmp_path / "b.jsonl")
    ra, rb = _records(a), _records(b)
    for r in ra + rb:
        if r["type"] == "config":
            r.pop("name", None)
            r["config"] = None
    assert ra == rb
    assert ra[-1]["type"] == "done"


def test_resume_from_a_checkpoint_gives_the_same_run(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "ROOT", tmp_path)
    cfg = _tiny(variant="optics", calib_start=1, calib_steps=1)
    whole = _records(T.run(cfg, tmp_path / "whole.jsonl"))
    part = tmp_path / "part.jsonl"
    T.run(cfg, part, max_seconds=0.0)                 # stops after the first checkpoint
    assert (tmp_path / "results" / "ckpt" / "part.pt").exists()
    T.run(cfg, part)
    resumed = _records(part)
    assert [r for r in resumed if r["type"] == "eval"] == [r for r in whole if r["type"] == "eval"]


@pytest.mark.parametrize("variant", ["topk", "band"])
def test_batched_workers_match_a_loop_over_workers(variant):
    """The batched DeMo step against a plain loop over workers, written out from the paper."""
    import compressors as C
    shapes = {"w": (128, 192), "v": (128,)}
    W, lr, decay = 3, 1e-3, 0.999
    init = {n: torch.randn(s, generator=torch.Generator().manual_seed(7), dtype=torch.float64) for n, s in shapes.items()}
    ours = {n: v.clone() for n, v in init.items()}
    st = T.DeMoState(T.RunConfig(variant=variant, workers=W, decay=decay), ours)
    ref = {n: v.clone() for n, v in init.items()}
    deltas = {n: [torch.zeros(s, dtype=torch.float64) for _ in range(W)] for n, s in shapes.items()}
    for step in range(3):
        grads = {n: g.double() for n, g in _grads(ref, W, seed=20 + step).items()}
        T.demo_step(st, ours, grads, lr, step, torch.Generator())
        for n in shapes:
            ch, k = st.ch[n], st.k[n]
            idx_l, val_l = [], []
            for w in range(W):
                d = deltas[n][w]
                d.mul_(decay).add_(grads[n][w], alpha=lr)
                coef = ch.encode(d)
                if variant == "topk":
                    idx, val = C.topk_select(coef, k)
                else:
                    idx = st.pos[n].expand(ch.chunks, k)
                    val = coef[:, st.pos[n]]
                d.sub_(ch.decode(torch.zeros_like(coef).scatter_(-1, idx, val)))
                idx_l.append(idx)
                val_l.append(val)
            agg = C.scatter_mean(ch.chunks, ch.m, idx_l, val_l, torch.float64)
            ref[n].add_(ch.decode(agg).sign(), alpha=-lr)
            assert torch.allclose(ours[n], ref[n], atol=1e-12), (variant, step, n)
            for w in range(W):
                assert torch.allclose(st.delta[n][w], deltas[n][w], atol=1e-12)


def test_chunker_handles_a_batch_of_tensors():
    import compressors as C
    for mode, size in (("2d", 64), ("1d", 256)):
        ch = C.Chunker(torch.Size((128, 192)), mode, size, dtype=torch.float64)
        x = torch.randn(4, 128, 192, dtype=torch.float64)
        batched = ch.encode(x)
        assert torch.allclose(batched, torch.stack([ch.encode(x[i]) for i in range(4)]), atol=1e-12)
        assert torch.allclose(ch.decode(batched), x, atol=1e-12)


def test_optics_state_builds_for_width_384():
    """The tinystories-gpu width: every tensor must chunk to power-of-2 sides at setup, so a bad
    side fails at once rather than when calibration ends."""
    shapes = {"w": (384, 384), "fc": (1536, 384), "ln": (384,)}
    params = {n: torch.zeros(s) for n, s in shapes.items()}
    cfg = T.RunConfig(variant="optics", band="low", chunk_mode="1d", chunk=256, workers=2)
    st = T.DeMoState(cfg, params)
    assert st.ch["ln"].m == 128 and st.ch["w"].m == 256
    st.finish_calibration()


@pytest.mark.parametrize("feedback", ["exact", "sent"])
def test_optics_error_feedback_removes_exact_or_sent_band(feedback):
    """After calibration, the delta loses the exact band (feedback="exact") or the optics' noisy
    output (feedback="sent"): with "sent", delta + what was sent equals the pre-step delta."""
    shapes = {"w": (128, 128)}
    params = {"w": torch.zeros(shapes["w"])}
    cfg = T.RunConfig(variant="optics", band="low", chunk_mode="1d", chunk=256, workers=2,
                      calib_start=0, calib_steps=1, enob=6.0, planes=8, feedback=feedback, fs_every=2)
    st = T.DeMoState(cfg, params)
    T.demo_step(st, params, _grads(params, 2, seed=1), 1e-3, 0, torch.Generator())      # calibrates
    before = st.delta["w"].clone() * cfg.decay + _grads(params, 2, seed=2)["w"] * 1e-3
    ch, pos = st.ch["w"], st.pos["w"]
    gen = torch.Generator().manual_seed(3)
    sent = st.emu["w"](ch.to_chunks(before).reshape(-1, ch.m), torch.Generator().manual_seed(3))
    T.demo_step(st, params, _grads(params, 2, seed=2), 1e-3, 1, gen)
    left = ch.encode(st.delta["w"])[..., pos].reshape(-1, len(pos))
    exact = ch.encode(before)[..., pos].reshape(-1, len(pos))
    if feedback == "exact":
        assert left.abs().max() < 1e-6
    else:
        assert torch.allclose(left, exact - sent, atol=1e-6)
        assert (exact - sent).abs().max() > 1e-6
