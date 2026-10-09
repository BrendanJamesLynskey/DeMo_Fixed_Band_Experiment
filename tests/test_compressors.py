"""Transforms, bands, byte counting and the optics emulation."""

import math

import pytest
import torch

import compressors as C


@pytest.mark.parametrize("n", [4, 16, 64, 256])
def test_dct_is_orthonormal(n):
    d = C.dct_matrix(n)
    assert torch.allclose(d @ d.T, torch.eye(n, dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("shape,mode,size", [((128, 192), "2d", 64), ((64,), "2d", 64),
                                             ((128, 128), "1d", 256), ((128, 128), "1d", 64)])
def test_encode_decode_round_trip(shape, mode, size):
    ch = C.Chunker(torch.Size(shape), mode, size, dtype=torch.float64)
    x = torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(ch.decode(ch.encode(x)), x, atol=1e-12)
    # energy is preserved (Parseval), so captured-energy fractions are meaningful
    assert float((ch.encode(x) ** 2).sum()) == pytest.approx(float((x ** 2).sum()))


def test_encode_matches_the_kron_transform_matrix():
    ch = C.Chunker(torch.Size((64, 128)), "2d", 32, dtype=torch.float64)
    x = torch.randn(64, 128, dtype=torch.float64)
    assert torch.allclose(ch.encode(x), ch.to_chunks(x) @ ch.transform_matrix().T, atol=1e-12)


def _band_projection(ch, pos):
    def project(x):
        coef = ch.encode(x)
        kept = torch.zeros_like(coef)
        kept[:, pos] = coef[:, pos]
        return ch.decode(kept)
    return project


@pytest.mark.parametrize("mode,size", [("2d", 16), ("1d", 64)])
def test_fixed_band_is_an_orthogonal_projection(mode, size):
    ch = C.Chunker(torch.Size((64, 64)), mode, size, dtype=torch.float64)
    pos = C.band_order(ch.chunk_shape, "low")[: C.keep_count(ch.m, 1 / 16)]
    P = _band_projection(ch, pos)
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(64, 64, dtype=torch.float64, generator=g), torch.randn(64, 64, dtype=torch.float64, generator=g)
    assert torch.allclose(P(P(x)), P(x), atol=1e-12)                       # idempotent
    assert float((P(x) * y).sum()) == pytest.approx(float((x * P(y)).sum()))   # self-adjoint


def test_low_band_of_a_square_count_is_the_top_left_square():
    pos = C.band_order((8, 8), "low")[:16]
    assert sorted(pos.tolist()) == [i * 8 + j for i in range(4) for j in range(4)]
    assert C.band_order((8, 8), "high")[:1].tolist() == [63]
    assert sorted(C.band_order((8, 8), "random", seed=3).tolist()) == list(range(64))


def test_topk_never_captures_less_than_a_band():
    ch = C.Chunker(torch.Size((128, 128)), "2d", 64)
    coef = ch.encode(torch.randn(128, 128))
    k = C.keep_count(ch.m, 1 / 16)
    for shape in ("low", "zigzag", "high", "random"):
        band = C.captured_energy(coef, C.band_order(ch.chunk_shape, shape)[:k], k)
        assert C.captured_energy(coef, None, k) >= band - 1e-9


def test_byte_accounting_matches_a_hand_calculation():
    # a 128 x 192 matrix in 64 x 64 chunks: 2 x 3 = 6 chunks of 4,096 coefficients, keep 1/16 = 256
    ch = C.Chunker(torch.Size((128, 192)), "2d", 64)
    assert (ch.chunks, ch.m, C.keep_count(ch.m, 1 / 16)) == (6, 4096, 256)
    assert C.index_bytes(4096) == 2                       # 12 bits, rounded up to 2 bytes
    assert C.index_bytes(256) == 1
    # band: values only, BF16 = 6 x 256 x 2 = 3,072 bytes; top-k adds 2-byte indices: 6,144
    assert 6 * 256 * C.VALUE_BYTES == 3072
    assert 6 * 256 * (C.VALUE_BYTES + C.index_bytes(4096)) == 6144


def test_scatter_mean_averages_contributions_per_position():
    idx = [torch.tensor([[0, 1]]), torch.tensor([[1, 2]])]
    val = [torch.tensor([[2.0, 4.0]]), torch.tensor([[6.0, 8.0]])]
    out = C.scatter_mean(1, 4, idx, val, torch.float32)
    assert out.tolist() == [[2.0, 5.0, 8.0, 0.0]]


# ─────────────────────────────────────────────────── optics ──
def _emulator(planes, enob, mode="1d", size=64, kappa=4.0):
    ch = C.Chunker(torch.Size((64, 64)), mode, size, dtype=torch.float64)
    pos = C.band_order(ch.chunk_shape, "low")[: C.keep_count(ch.m, 1 / 16)]
    emu = C.OpticsEmulator(ch, pos, C.OpticsConfig(planes, enob, kappa, 256))
    return ch, pos, emu


def test_bit_planes_recombine_to_the_transform_of_the_quantised_input():
    ch, pos, emu = _emulator(planes=10, enob=None)
    x = torch.randn(64, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    emu.calibrate(float(x.pow(2).mean().sqrt()))
    got = emu(ch.to_chunks(x), torch.Generator())
    qmax = 2 ** 9 - 1
    q = torch.round(ch.to_chunks(x) / emu.scale_in * qmax).clamp(-qmax, qmax)
    want = (q * emu.scale_in / qmax) @ ch.transform_matrix()[pos].T
    assert torch.allclose(got, want, atol=1e-9)


def test_more_planes_means_less_quantisation_error():
    errs = []
    x = torch.randn(64, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    for p in (4, 8, 12):
        ch, pos, emu = _emulator(planes=p, enob=None)
        emu.calibrate(float(x.pow(2).mean().sqrt()))
        exact = ch.encode(x)[:, pos]
        errs.append(float((emu(ch.to_chunks(x), torch.Generator()) - exact).pow(2).mean().sqrt()))
    assert errs[0] > errs[1] > errs[2]


def test_readout_noise_follows_the_enob_rule():
    ch, pos, emu = _emulator(planes=None, enob=8.0)
    x = torch.randn(64, 64, dtype=torch.float64)
    exact = ch.encode(x)[:, pos]
    noisy = torch.cat([emu(ch.to_chunks(x), torch.Generator().manual_seed(s)) for s in range(20)])
    rms = float((noisy - exact.repeat(20, 1)).pow(2).mean().sqrt())
    assert rms == pytest.approx(emu.fs_out * 2 ** -8 * 2 / math.sqrt(12), rel=0.05)


def test_saturation_is_counted():
    ch, pos, emu = _emulator(planes=8, enob=None, kappa=1.0)
    x = torch.randn(64, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    emu.calibrate(float(x.pow(2).mean().sqrt()))
    emu(ch.to_chunks(x), torch.Generator())
    # full scale at 1 sigma: about 32% of a Gaussian lies beyond it
    assert 0.25 < emu.clip_rate() < 0.40


def test_native_size_limit():
    ch = C.Chunker(torch.Size((512, 512)), "1d", 512)
    with pytest.raises(ValueError):
        C.OpticsEmulator(ch, torch.arange(4), C.OpticsConfig(8, 8, 4.0, 256))


def test_pow2_chunking_for_widths_that_are_not_powers_of_2():
    # a 384-vector (LayerNorm at width 384): DeMo's rule gives runs of 192, the optics needs 128
    assert C.Chunker(torch.Size((384,)), "1d", 256).m == 192
    ch = C.Chunker(torch.Size((384,)), "1d", 256, pow2=True)
    assert ch.m == 128
    C.OpticsEmulator.check(ch, 256)
    assert C.Chunker(torch.Size((384, 384)), "2d", 64, pow2=True).chunk_shape == (64, 64)
    assert C.Chunker(torch.Size((384, 1536)), "1d", 256, pow2=True).m == 256
