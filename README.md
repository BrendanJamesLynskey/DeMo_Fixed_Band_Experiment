# DeMo Fixed-Band Experiment

Can DeMo's top-k selection be replaced by a **fixed frequency band**, which needs no comparisons and so suits a photonic link module, without losing convergence? And how much error can an emulated analogue optical transform add before it hurts?

DeMo (Peng et al., [arXiv:2411.19870](https://arxiv.org/abs/2411.19870), ICLR 2026) compresses each worker's momentum with a chunked DCT and keeps the top-k coefficients of each chunk. Optalysys's compute-in-transit module (Kundu et al., [arXiv:2608.21536](https://arxiv.org/abs/2608.21536)) lists FFT, NTT, add, multiply, MAC and modular arithmetic, but no comparisons, so it cannot do top-k. A fixed band is a gather: the same positions in every chunk, every step. This repository trains small GPTs with simulated data parallelism and measures what that substitution costs.

**Status: planned runs in progress.** Results will appear in `results/<dataset>/results.md`, written by `experiment/analyse.py`; every number in the write-up will come from there.

Part of the [Compute in Transit series](https://github.com/BrendanJamesLynskey/LLM_Hub_Compute_in_Transit); the motivation is in [CiT 02, slides 09 and 10](https://brendanjameslynskey.github.io/CiT_02_Data_Movement_in_LLM_Training/#slide-09).

## Variants

All compressed variants keep the same fraction of coefficients (1/16).

| | Variant | What is sent |
|---|---|---|
| A | AdamW on all-reduced gradients (the DDP baseline) | dense BF16 gradients |
| B | DeMo as published: chunked DCT, per-chunk top-k | values and indices |
| C | DeMo with a fixed low-frequency band | values only |
| D | DeMo with a calibrated band: the mean energy map of a short window chooses the positions once, then they are frozen | values only |
| E | C with the band computed by an emulated optical transform | values only |
| F | Other fixed shapes of the same size: zig-zag, high band, random (the control) | values only |

Chunks are DeMo's 2-D 64x64 tiles, or 1-D runs of 64 or 256 values (the module does 1xN radix-4 transforms; its FHE run split 65,536-point NTTs into 256-point ones).

**Emulated optics (E).** Each value enters as a P-bit two's-complement integer against a fixed full scale, calibrated once as kappa times the RMS of the delta over the calibration window (a sum of squares: no comparisons), with saturation; the clip rate is logged. Each bit plane passes through the transform separately, as binary (NRZ) input, and its readout gets Gaussian noise of RMS FS x 2^-ENOB x 2/sqrt(12), FS the readout full scale (fixed by the transform). The planes are recombined with weights 2^b, the top plane negative. ENOB is swept over 6, 8, 10 and 12; P over 4, 8 and 12 (P is also the number of optical passes). Error feedback comes in two forms. As first designed, each sender removes the exact band, so the optics' quantisation, clipping and noise are never corrected; on the GPU runs every such configuration diverged, because error feedback leaves the delta almost all out-of-band residual (about 0.2% of its energy is in the band) and the errors repeat step after step. In an all-gather each worker receives its own values back, so it can instead remove what the optics actually produced (`feedback="sent"`); with that, and the input full scale optionally re-set every N steps from the delta's mean square (`fs_every`, still only multiply-accumulates), the optics runs are repeated. The optics accepts only power-of-2 chunk sides, so E cuts each tensor into the largest power-of-2 runs that divide it; at width 384 that changes only the 17 LayerNorm vectors (6,528 parameters), which E cuts into runs of 128 where C, following DeMo's rule, uses 192.

**Diagnostics.** Every 100 steps, on worker 0: the share of energy kept by top-k and by each band shape at the same k, for a plain momentum (never compressed), for the delta DeMo actually compresses, and for the gradient.

## Setup

- Simulated data parallelism: W = 8 workers in one process, each with its own shard, momentum and DeMo state, sharing one set of weights. Bytes are counted analytically, BF16 values, top-k indices at the bytes needed for a position in a chunk (the reference code sends int64; both are reported).
- TinyStories ([roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), V2 GPT-4 stories), the first 200 MB of the training file, a byte-level BPE tokeniser of 4,096 tokens trained here. A 6-layer GPT of width 192 (3.5M parameters), context 256, 4 sequences per worker per step, 1,500 steps. TinyShakespeare at character level is the smoke test.
- A learning-rate sweep chooses each optimiser's rate first (`results/tinystories-lr/`, `results/tinystories/lr_choice.json`).
- Three seeds for A to D and C with 1-D chunks; one seed for E and F.
- Hardware: an Intel Core i7-3770 (4 cores, AVX, no AVX2), CPU only. About 3.6 s per step for the DeMo variants. The model size and run length were chosen to fit it; the directions' 10 to 20M parameters would allow only about 600 steps in the same time.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests                                        # 36 tests
.venv/bin/python experiment/sweep.py --grid smoke --dataset shakespeare
.venv/bin/python experiment/sweep.py --grid all --dataset tinystories --commit   # the whole experiment
.venv/bin/python experiment/analyse.py --dataset tinystories
```

**On a GPU (Colab):** open [colab/run_sweep.ipynb](https://colab.research.google.com/github/BrendanJamesLynskey/DeMo_Fixed_Band_Experiment/blob/main/colab/run_sweep.ipynb), choose a GPU runtime and *Run all*. It keeps the repository, data and checkpoints in Google Drive and runs the `tinystories-gpu` profile (8 layers, width 384, about 16M parameters, 3,000 steps); a reconnect and *Run all* resumes.

Runs resume from checkpoints, and the sweep retries a run that crashed. The Jenkinsfile runs lint, the tests and a smoke run of every variant on synthetic data, with an optional nightly sweep.

## The reference DeMo code

DeMo is implemented here from the paper. The authors' code ([bloc97/DeMo](https://github.com/bloc97/DeMo)) carries no licence, so none of it is copied. One test loads it from a local checkout (set `DEMO_REF` to its `demo.py`) and checks that variant B gives the same parameters and momentum as the reference optimizer over three steps; without the checkout the test is skipped.

## Licence

Code: MIT ([LICENSE](LICENSE)). Prose: CC BY 4.0 ([LICENSE-TEXT](LICENSE-TEXT)).
