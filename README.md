# DeMo Fixed-Band Experiment

Can DeMo's top-k selection be replaced by a **fixed frequency band**, which needs no comparisons and so suits a photonic link module, without losing convergence? And how much error can an emulated analogue optical transform add before it hurts?

DeMo (Peng et al., [arXiv:2411.19870](https://arxiv.org/abs/2411.19870), ICLR 2026) compresses each worker's momentum with a chunked DCT and keeps the top-k coefficients of each chunk. Optalysys's compute-in-transit module (Kundu et al., [arXiv:2608.21536](https://arxiv.org/abs/2608.21536)) lists FFT, NTT, add, multiply, MAC and modular arithmetic, but no comparisons, so it cannot do top-k. A fixed band is a gather: the same positions in every chunk, every step. This repository trains small GPTs with simulated data parallelism and measures what that substitution costs.

**Status: complete.** The full results are in [results/tinystories-gpu/results.md](results/tinystories-gpu/results.md), written by `experiment/analyse.py`; every number below comes from there.

## Results

TinyStories, a 16M-parameter GPT, 8 simulated workers, 3,000 steps. Final validation loss (mean ± sd over seeds), and the bytes each worker originates per step:

| | Variant | Seeds | Final loss | Payload / step | vs dense |
|---|---|---|---|---|---|
| A | AdamW, dense all-reduce | 3 | 1.646 ± 0.005 | 31.7 MB | 1x |
| B | DeMo, top-k (2-D 64x64) | 3 | 1.785 ± 0.007 | 3.96 MB | 8x |
| C | Fixed low band (2-D 64x64) | 3 | 3.158 ± 0.009 | 1.98 MB | 16x |
| C | Fixed low band (1-D 256) | 3 | 3.124 ± 0.008 | 1.98 MB | 16x |
| D | Calibrated band (2-D 64x64) | 3 | 3.087 ± 0.027 | 1.98 MB | 16x |
| E | Emulated optics, repaired, ENOB 6 / 8 / 10 / 12 | 1 each | 3.423 / 3.017 / 2.878 / 2.905 | 1.98 MB | 16x |
| F | Fixed random band (2-D 64x64) | 1 | 2.700 | 1.98 MB | 16x |

- **A fixed band does not replace top-k.** The low, zig-zag, high and calibrated bands end 1.24 to 1.46 nats above top-k, and even a random fixed set of positions ends 0.92 above, at half top-k's bytes. The saving in bytes does not buy the loss back: no fixed band comes near top-k's loss at any point in the run.
- **Per byte, top-k still wins.** Top-k reaches the 1-D fixed band's final loss (3.124) by step 400, after 1.58 GB per worker; the fixed band itself needs 2,667 steps and 5.28 GB (section 1 of results.md, *Steps to reach a loss*).
- **Why:** on the momentum of the fixed-band runs, top-k at the same k would keep 0.64 of the energy in 2-D chunks; the fixed low band keeps 0.013, less than a random set of positions (0.065). The gradient's low band is better (0.31), but DeMo compresses a momentum, and with error feedback the delta it compresses holds almost nothing in the band (section 2 of results.md).
- **Emulated optics:** as first designed, every run diverged; the cause and the fix (error feedback of the values actually sent) are in [docs/optics_investigation.md](docs/optics_investigation.md). Repaired, the optics band trains, and from ENOB 8 it ends slightly *below* the exact band it emulates; a perfect-optics control (3.141) rules out a chunking difference as the cause. The optics does not rescue the fixed band.

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

**Emulated optics (E).** Each value enters as a P-bit two's-complement integer against a fixed full scale, calibrated once as kappa times the RMS of the delta over the calibration window (a sum of squares: no comparisons), with saturation; the clip rate is logged. Each bit plane passes through the transform separately, as binary (NRZ) input, and its readout gets Gaussian noise of RMS FS x 2^-ENOB x 2/sqrt(12), FS the readout full scale (fixed by the transform). The planes are recombined with weights 2^b, the top plane negative. ENOB is swept over 6, 8, 10 and 12; P over 4, 8 and 12 (P is also the number of optical passes). Error feedback comes in two forms. As first designed, each sender removes the exact band, so the optics' quantisation, clipping and noise are never corrected; on the GPU runs every such configuration diverged, because error feedback leaves the delta almost all out-of-band residual (about 0.2% of its energy is in the band) and the errors repeat step after step. In an all-gather each worker receives its own values back, so it can instead remove what the optics actually produced (`feedback="sent"`); with that, and the input full scale optionally re-set every N steps from the delta's mean square (`fs_every`, still only multiply-accumulates), the optics runs are repeated. The whole diagnosis, with the CPU check that isolated it, is in [docs/optics_investigation.md](docs/optics_investigation.md). The optics accepts only power-of-2 chunk sides, so E cuts each tensor into the largest power-of-2 runs that divide it; at width 384 that changes only the 17 LayerNorm vectors (6,528 parameters), which E cuts into runs of 128 where C, following DeMo's rule, uses 192.

**Diagnostics.** Every 200 steps, on worker 0: the share of energy kept by top-k and by each band shape at the same k, for a plain momentum (never compressed), for the delta DeMo actually compresses, and for the gradient.

## Setup

- Simulated data parallelism: W = 8 workers in one process, each with its own shard, momentum and DeMo state, sharing one set of weights. Bytes are counted analytically, BF16 values, top-k indices at the bytes needed for a position in a chunk (the reference code sends int64; both are reported).
- TinyStories ([roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), V2 GPT-4 stories), the first 200 MB of the training file, a byte-level BPE tokeniser of 4,096 tokens trained here. The `tinystories-gpu` profile: an 8-layer GPT of width 384 (15.8M parameters), context 256, 8 sequences per worker per step, 3,000 steps (about 49M tokens per run).
- TinyShakespeare (about 1 MB of Shakespeare's plays, read one character at a time) is the small, fast test set: the smoke test, and the CPU check that isolated the optics problem.
- A learning-rate sweep (800 steps) chose each optimiser's rate first: 1e-3 for DeMo, the middle of three, and 3e-3 for AdamW, the higher of the two tried, so AdamW might do slightly better still (`results/tinystories-gpu-lr/`, `results/tinystories-gpu/lr_choice.json`).
- Three seeds for A to D and C with 1-D chunks; one seed for E and F.
- Hardware: Google Colab GPUs: an A100 (40 or 80 GB) for most runs, three on an L4 and three on a T4 (the GPU is recorded in each log). About 0.17 s per step for AdamW and 0.3 s for DeMo on the A100. GPU runs are close but not bit-identical between runs (scatter_reduce has no deterministic CUDA kernel). The work started on a CPU profile (an i7-3770, 3.5M parameters, 1,500 steps, `tinystories`), stopped after its learning-rate sweep when the GPU became available.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests                                        # 36 tests
.venv/bin/python experiment/sweep.py --grid smoke --dataset shakespeare
.venv/bin/python experiment/sweep.py --grid all --dataset tinystories-gpu --commit   # the whole experiment (GPU)
.venv/bin/python experiment/analyse.py --dataset tinystories-gpu
.venv/bin/python scripts/optics_feedback_check.py                                   # the CPU check of the optics fix
```

**On a GPU (Colab):** open [colab/run_sweep.ipynb](https://colab.research.google.com/github/BrendanJamesLynskey/DeMo_Fixed_Band_Experiment/blob/main/colab/run_sweep.ipynb), choose a GPU runtime and *Run all*. It keeps the repository, data and checkpoints in Google Drive and runs the `tinystories-gpu` profile (8 layers, width 384, about 16M parameters, 3,000 steps); a reconnect and *Run all* resumes.

Runs resume from checkpoints, and the sweep retries a run that crashed. The Jenkinsfile runs lint, the tests and a smoke run of every variant on synthetic data, with an optional nightly sweep.

## The reference DeMo code

DeMo is implemented here from the paper. The authors' code ([bloc97/DeMo](https://github.com/bloc97/DeMo)) carries no licence, so none of it is copied. One test loads it from a local checkout (set `DEMO_REF` to its `demo.py`) and checks that variant B gives the same parameters and momentum as the reference optimizer over three steps; without the checkout the test is skipped.

## Licence

Code: MIT ([LICENSE](LICENSE)). Prose: CC BY 4.0 ([LICENSE-TEXT](LICENSE-TEXT)).
