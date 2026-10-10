# Optics feedback check (TinyShakespeare, CPU)

Written by `scripts/optics_feedback_check.py`. 4 layers, width 128, 8 workers, 600 steps, 1-D 256-point chunks, keep 1/16, learning rate 1e-3, calibration over steps 50 to 99. Optics: 8 planes and ENOB 8 unless stated, kappa 4. Validation loss, with the clip rate over the preceding interval in brackets.

| Run | step 100 | step 200 | step 300 | step 400 | step 500 | step 600 |
|---|---|---|---|---|---|---|
| C, exact band (reference) | 2.730 | 2.601 | 2.545 | 2.508 | 2.475 | 2.464 |
| E, full scale fixed, feedback of the exact band (as first designed) | 2.730 | 3.298 (6.4%) | 3.586 (30.9%) | 4.192 (35.4%) | 6.442 (39.7%) | 8.809 (41.8%) |
| E, full scale re-set every 50 steps, feedback of the exact band | 2.730 | 2.938 (1.7%) | 3.581 (10.9%) | 3.331 (0.7%) | 3.362 (0.6%) | 3.404 (0.6%) |
| E, as above, ENOB 12 | 2.730 | 2.775 (1.6%) | 4.019 (12.8%) | 3.218 (0.9%) | 3.073 (0.7%) | 3.069 (0.7%) |
| E, as above, no readout noise | 2.730 | 2.780 (1.6%) | 3.346 (3.4%) | 3.116 (1.6%) | 2.985 (0.7%) | 2.967 (0.7%) |
| E, full scale re-set every 50 steps, feedback of the band sent | 2.730 | 2.620 (0.8%) | 2.568 (0.8%) | 2.545 (0.7%) | 2.537 (0.6%) | 2.535 (0.5%) |
| E, full scale fixed, feedback of the band sent | 2.730 | 2.624 (1.7%) | 2.567 (3.6%) | 2.537 (5.4%) | 2.523 (6.1%) | 2.520 (5.7%) |
