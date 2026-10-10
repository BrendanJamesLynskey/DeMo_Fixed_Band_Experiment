# Why the first optics runs diverged, and the fix

Variant E computes DeMo's fixed band with an emulated optical transform. In the first GPU sweep, all six E runs diverged, while the same band computed exactly (C, 1-D 256-point chunks) trained normally. This note records what went wrong, how it was diagnosed, and what changed. The numbers come from the run logs in `results/tinystories-gpu/` and the CPU check in `results/shakespeare-optics-check/results.md` (written by `scripts/optics_feedback_check.py`).

## 1. What happened (TinyStories, 16M parameters, GPU)

E as first designed: input full scale fixed once, as kappa = 4 times the RMS of the delta over the calibration window (steps 200 to 299), then saturation; 8-bit two's-complement input in exact binary bit planes; per-plane readout noise at the given ENOB; error feedback removed the **exact** band.

| Run | Loss at 400 | Loss at 3000 | Clip rate 300–400 | Clip rate 400–600 |
|---|---|---|---|---|
| C, exact band, 1-D 256 (seed 0) | 4.07 | 3.12 | – | – |
| E, ENOB 6, 8 planes | 4.56 | 11.15 | 0.8% | 38.6% |
| E, ENOB 8, 8 planes | 4.59 | 214.65 | 1.0% | 41.5% |
| E, ENOB 10, 8 planes | 4.47 | 42.18 | 0.8% | 34.9% |
| E, ENOB 12, 8 planes | 4.51 | 336.98 | 0.8% | 42.9% |
| E, ENOB 8, 4 planes | 5.20 | 344.39 | 0.9% | 50.8% |
| E, ENOB 8, 12 planes | 4.49 | 558.83 | 0.8% | 37.0% |

Two signs that this was not simply analogue noise: the damage did not fall as ENOB rose (ENOB 12 ended worse than ENOB 6), and 12 planes ended worse than 8. And the clip rate jumped from about 1% to 35–51% within 200 steps of the end of calibration.

## 2. The residual: what the optics actually sees

DeMo's error feedback keeps everything not sent in the delta, decaying only by 0.999 per step. With a fixed low band, the band is removed every step and the out-of-band residual accumulates. The diagnostics (energy share of the low band in the delta, worker 0, the three C 1-D 256 seeds) show how little of the optics' input is the signal it extracts:

| Step | 200 | 400 | 1000 | 2000 | 3000 |
|---|---|---|---|---|---|
| Band share of delta energy | 1.4–2.3e-3 | 2.3–2.9e-4 | 1.1–1.5e-4 | 3.9–4.2e-5 | 1.6–1.8e-5 |

For comparison, the band holds 11–31% of a plain momentum's energy over the same steps. So the transform's input range is set by the residual, a quantity that is never sent, and the band it extracts is a few parts in 10^5 of the input energy by the end.

Two consequences:

- **The full scale goes stale.** The residual keeps growing long after a short calibration window, so a full scale fixed at step 300 saturates: 30–50% of inputs clipped.
- **The errors are large relative to the band and nearly the same each step.** Per band coefficient, the RMS is about sqrt(16 x share) times the input RMS: 0.18 at share 2e-3, 0.016 at share 1.6e-5. With kappa 4 and 8 planes the quantisation step is 4/127 = 0.031 of the input RMS, an error of 0.009 per coefficient. The readout noise per plane is FS x 2^-ENOB x 2/sqrt(12), FS = 16 for a 256-point orthonormal DCT (the DC row), so 0.036 plane units at ENOB 8; the planes are recombined with weights 2^b, which multiplies it by sqrt(sum 4^b) = 147.8 for 8 planes, 0.17 of the input RMS (0.011 at ENOB 12). So late in training the band is at or below the error. The residual changes slowly, so quantisation and clipping errors repeat almost identically from step to step: a bias, not noise that averages out.

With error feedback of the exact band, the sender believes it has sent the exact band, while every receiver applies the optics' version. The difference is never corrected, and DeMo applies the **sign** of the update, so a consistent error moves every weight by a full step in the wrong direction, step after step.

The emulator itself is accurate: on white input (where the band holds 1/16 of the energy), 8 planes without readout noise reproduce the exact band to 0.9% RMS error, 12 planes to 0.2%, and with ENOB 8 noise to 16%.

## 3. Diagnosis on the CPU (TinyShakespeare)

TinyShakespeare is about 1 MB of Shakespeare's plays, read one character at a time: a small, fast test set, so each idea could be tried in about 12 minutes on the CPU without spending GPU time. Each change was tested on a small model first (4 layers, width 128, 600 steps; calibration 50 to 99, so the optics starts at step 100). Final validation loss at step 600 (clip rate over the last interval):

| Configuration (8 planes, ENOB 8 unless stated) | Final loss |
|---|---|
| C, exact band (reference) | 2.464 |
| E as first designed: full scale fixed, feedback of the exact band | 8.809 (41.8%) |
| Full scale re-set every 50 steps, feedback of the exact band | 3.404 (0.6%) |
| ... ENOB 12 | 3.069 (0.7%) |
| ... no readout noise at all | 2.967 (0.7%) |
| Full scale re-set every 50 steps, **feedback of the band sent** | 2.535 (0.5%) |
| Full scale fixed, **feedback of the band sent** | 2.520 (5.7%) |

Reading the table in order:

1. The failure reproduces at small scale: clipping rises to 42% and the loss climbs from 2.73 to 8.81.
2. Re-setting the full scale removes most of the clipping, but the loss still drifts up. So saturation was not the only problem.
3. Raising ENOB to 12 helps a little, and removing the readout noise entirely still leaves the loss 0.5 above the exact band. So the noise was not the main problem either: quantisation and clipping errors, never fed back, are enough.
4. Feeding back what was actually sent fixes it, with or without re-setting the full scale. With the full scale fixed, about 6% of inputs clip, but the clipping error is now fed back too, and the result (2.520) is within 0.06 of the exact band.

## 4. The fix, and why it is physically allowed

In DeMo the workers exchange their compressed values with an **all-gather**: every worker receives every worker's values, its own included. So a sender can see what the optics produced for its own data, noise and all, and remove that from its delta instead of the exact band. That is `feedback="sent"`. The earlier statement in this repository that "the sender cannot see the optics' noise" was wrong for an all-gather.

The optional full-scale re-set (`fs_every`) uses the delta's mean square over the last interval: a sum of squares, so multiply-accumulates only, with no comparisons, in keeping with the module's operation set.

## 5. Other problems met on the way

- **Chunk sizes at width 384.** DeMo's rule (the largest divisor not above 256) cut the 17 LayerNorm vectors of 384 values into runs of 192, which the optics refuses (power-of-2 sides only). Every optics run crashed when calibration ended. Fixed in commit fe28f19: E uses power-of-2 runs (128 for those vectors, 6,528 of about 16M parameters); C keeps DeMo's rule. The check now runs at start-up.
- **Colab sessions.** The sweep ran over several Colab sessions (T4, then A100, L4 and A100 again), with results and checkpoints copied to Google Drive every 10 minutes and restored at the start of each session. Two sessions disconnected without a recorded cause. Checkpoints of finished runs were never deleted from Drive and piled up (roughly 1 GB each); they are now pruned.

## 6. The rerun (TinyStories, GPU)

The optics stage was rerun with the full scale re-set every 100 steps and feedback of the band sent, plus feedback of the band sent with the full scale fixed, and a control with perfect optics (no quantisation, no noise), which is C with E's power-of-2 chunking. One seed each. From `results/tinystories-gpu/results.md`, sections 1 and 3:

| Configuration | Final loss | Clip rate (whole run) |
|---|---|---|
| C, exact band, 1-D 256 (3 seeds) | 3.124 ± 0.008 | – |
| E, perfect optics (control) | 3.141 | 0% |
| E, ENOB 6, 8 planes | 3.423 | 0.28% |
| E, ENOB 8, 8 planes | 3.017 | 0.29% |
| E, ENOB 10, 8 planes | 2.878 | 0.30% |
| E, ENOB 12, 8 planes | 2.905 | 0.30% |
| E, ENOB 8, 4 planes | 3.137 | 0.22% |
| E, ENOB 8, 12 planes | 3.024 | 0.30% |
| E, ENOB 8, 8 planes, full scale fixed | 3.021 | 1.34% |

What it shows:

1. **Feedback of the band sent is the fix.** Every repaired run trains; clipping stays near 0.3%.
2. **Re-setting the full scale adds nothing** once the errors are fed back: 3.017 with it, 3.021 without (ENOB 8). With the full scale fixed, about 1.3% of inputs clip, and the clipping error is corrected like any other.
3. **Below ENOB 8 the noise costs loss** (ENOB 6: 3.423, 0.3 above the exact band). Four planes cost 0.12 against eight; twelve gain nothing.
4. **From ENOB 8 up, the optics band ends below the exact band**, by up to 0.25 (ENOB 10). The control rules out the chunking difference: perfect optics gives 3.141, 0.017 above the exact band's mean, about two of its seed standard deviations and far from the optics runs' gains (the control ran on a T4, the others on an A100, and GPU runs are not bit-identical).

Why would an inexact band do better than the exact one? A hypothesis, not tested here: quantisation and clipping are non-linear, so the values the optics produces carry a little of the large out-of-band residual (section 2) into the 16 positions sent; with feedback of the band sent, that part of the residual is then removed from the sender's delta, so over many steps some out-of-band information reaches the model that the exact band never sends. That fits the random fixed band (2.700) beating the low band (3.158): a fixed band loses by sending the same positions forever, and anything that leaks other information helps. Against it: on TinyShakespeare the repaired optics ended slightly *above* the exact band (2.520 to 2.535 against 2.464), and every E rerun is a single seed. Testing it would take several seeds and a measurement of how much out-of-band energy the sent values carry.

None of this rescues the fixed band: the best optics run (2.878) is still 1.09 above top-k (1.785).
