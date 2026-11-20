# Sequence 5 summary (690 fits on disk)
## T1  E2a: FLW qubit process (lam = 0.8) against HMMs

### flw-T50000-s0  (Hankel rank of the truth: words <= 6 3, gap law 3; OOM_3 exact, QSM_2 r = 1 exact, no finite HMM)
| model | inits | held-out KL ± SE | train − truth | d_16 | KL − KL(QSM_2) ± SE (paired) | params | flags |
|---|---|---|---|---|---|---|---|
| truth | 1 | 0.00e+00 ± 0.0e+00 | 0.00e+00 | 0.00e+00 | -3.06e-05 ± 2.3e-05 | – |  |
| qsm-D2-r1 | 6 | 3.06e-05 ± 2.3e-05 | -3.02e-05 | 2.43e-05 | – | 16 |  |
| qsm-D2-r2 | 6 | 3.06e-05 ± 2.3e-05 | -3.02e-05 | 2.43e-05 | -1.68e-09 ± 2.9e-10 | 32 |  |
| hmm-2 | 7 | 5.05e-02 ± 5.1e-04 | 5.04e-02 | 4.79e-02 | +5.05e-02 ± 5.0e-04 | 6 |  |
| hmm-3 | 7 | 2.73e-02 ± 3.6e-04 | 2.71e-02 | 2.48e-02 | +2.73e-02 ± 3.5e-04 | 15 |  |
| hmm-4 | 7 | 1.86e-02 ± 2.5e-04 | 1.80e-02 | 1.63e-02 | +1.86e-02 ± 2.4e-04 | 28 |  |
| hmm-6 | 7 | 1.49e-03 ± 8.7e-05 | 1.35e-03 | 1.00e-03 | +1.46e-03 ± 8.6e-05 | 66 |  |
| hmm-8 | 7 | 7.49e-04 ± 1.4e-04 | -2.48e-05 | 5.48e-04 | +7.18e-04 ± 1.4e-04 | 120 |  |
| hmm-12 | 7 | 8.43e-04 ± 1.6e-04 | -3.70e-04 | 6.78e-04 | +8.12e-04 ± 1.6e-04 | 276 | it-limit |
| hmm-16 | 7 | 8.75e-04 ± 1.8e-04 | -5.42e-04 | 7.54e-04 | +8.44e-04 ± 1.8e-04 | 496 | it-limit |
| markov-1 | 1 | 6.11e-02 ± 5.5e-04 | 6.04e-02 | 5.85e-02 | +6.11e-02 ± 5.4e-04 | 2 |  |
| markov-2 | 1 | 3.21e-02 ± 4.0e-04 | 3.18e-02 | 2.96e-02 | +3.21e-02 ± 4.0e-04 | 4 |  |
| markov-3 | 1 | 2.53e-02 ± 3.0e-04 | 2.49e-02 | 2.29e-02 | +2.53e-02 ± 2.9e-04 | 8 |  |
| markov-4 | 1 | 2.54e-02 ± 3.1e-04 | 2.46e-02 | 2.29e-02 | +2.54e-02 ± 3.0e-04 | 16 |  |
| markov-5 | 1 | 2.07e-02 ± 2.9e-04 | 1.94e-02 | 1.84e-02 | +2.07e-02 ± 2.8e-04 | 32 |  |
| markov-6 | 1 | 1.63e-02 ± 3.1e-04 | 1.51e-02 | 1.44e-02 | +1.63e-02 ± 2.9e-04 | 64 |  |
| markov-7 | 1 | 1.66e-02 ± 3.0e-04 | 1.49e-02 | 1.46e-02 | +1.65e-02 ± 2.8e-04 | 128 |  |
| markov-8 | 1 | 1.03e-02 ± 2.5e-04 | 8.15e-03 | 8.43e-03 | +1.03e-02 ± 2.3e-04 | 256 |  |

S* (parsimony, finite sample: smallest S with KL_HMM − ε < 2 SE) = >16; practical relevance (HMM_8 KL < 1e-4): False

### flw-T50000-s1  (Hankel rank of the truth: words <= 6 3, gap law 3; OOM_3 exact, QSM_2 r = 1 exact, no finite HMM)
| model | inits | held-out KL ± SE | train − truth | d_16 | KL − KL(QSM_2) ± SE (paired) | params | flags |
|---|---|---|---|---|---|---|---|
| truth | 1 | 0.00e+00 ± 0.0e+00 | 0.00e+00 | 0.00e+00 | -3.24e-05 ± 1.5e-05 | – |  |
| qsm-D2-r1 | 6 | 3.24e-05 ± 1.5e-05 | -1.99e-05 | 1.88e-05 | – | 16 |  |
| qsm-D2-r2 | 6 | 3.24e-05 ± 1.5e-05 | -1.99e-05 | 1.88e-05 | -1.53e-09 ± 7.3e-10 | 32 |  |
| hmm-2 | 7 | 5.05e-02 ± 5.1e-04 | 4.94e-02 | 4.79e-02 | +5.04e-02 ± 5.0e-04 | 6 |  |
| hmm-3 | 7 | 2.73e-02 ± 3.6e-04 | 2.60e-02 | 2.48e-02 | +2.73e-02 ± 3.6e-04 | 15 |  |
| hmm-4 | 7 | 1.86e-02 ± 2.5e-04 | 1.75e-02 | 1.63e-02 | +1.86e-02 ± 2.5e-04 | 28 |  |
| hmm-6 | 7 | 1.73e-03 ± 1.0e-04 | 1.02e-03 | 1.17e-03 | +1.70e-03 ± 1.0e-04 | 66 |  |
| hmm-8 | 7 | 2.54e-03 ± 4.8e-04 | 1.02e-05 | 1.68e-03 | +2.51e-03 ± 4.8e-04 | 120 |  |
| hmm-12 | 7 | 1.38e-03 ± 2.3e-04 | -2.01e-04 | 1.21e-03 | +1.35e-03 ± 2.3e-04 | 276 |  |
| hmm-16 | 7 | 6.27e-04 ± 1.1e-04 | -4.17e-04 | 6.10e-04 | +5.94e-04 ± 1.1e-04 | 496 |  |
| markov-1 | 1 | 6.11e-02 ± 5.5e-04 | 6.00e-02 | 5.85e-02 | +6.11e-02 ± 5.5e-04 | 2 |  |
| markov-2 | 1 | 3.21e-02 ± 4.1e-04 | 3.06e-02 | 2.96e-02 | +3.21e-02 ± 4.0e-04 | 4 |  |
| markov-3 | 1 | 2.53e-02 ± 3.1e-04 | 2.41e-02 | 2.29e-02 | +2.53e-02 ± 3.1e-04 | 8 |  |
| markov-4 | 1 | 2.54e-02 ± 3.1e-04 | 2.38e-02 | 2.29e-02 | +2.53e-02 ± 3.1e-04 | 16 |  |
| markov-5 | 1 | 2.07e-02 ± 2.9e-04 | 1.96e-02 | 1.85e-02 | +2.07e-02 ± 2.9e-04 | 32 |  |
| markov-6 | 1 | 1.64e-02 ± 3.5e-04 | 1.49e-02 | 1.45e-02 | +1.63e-02 ± 3.5e-04 | 64 |  |
| markov-7 | 1 | 1.66e-02 ± 3.5e-04 | 1.47e-02 | 1.47e-02 | +1.66e-02 ± 3.5e-04 | 128 |  |
| markov-8 | 1 | 1.03e-02 ± 3.0e-04 | 7.86e-03 | 8.41e-03 | +1.02e-02 ± 3.0e-04 | 256 |  |

S* (parsimony, finite sample: smallest S with KL_HMM − ε < 2 SE) = >16; practical relevance (HMM_8 KL < 1e-4): False

### flw-T5000-s0  (Hankel rank of the truth: words <= 6 3, gap law 3; OOM_3 exact, QSM_2 r = 1 exact, no finite HMM)
| model | inits | held-out KL ± SE | train − truth | d_16 | KL − KL(QSM_2) ± SE (paired) | params | flags |
|---|---|---|---|---|---|---|---|
| truth | 1 | 0.00e+00 ± 0.0e+00 | 0.00e+00 | 0.00e+00 | -2.19e-04 ± 7.0e-05 | – |  |
| qsm-D2-r1 | 6 | 2.19e-04 ± 7.0e-05 | -2.79e-04 | 2.14e-04 | – | 16 |  |
| qsm-D2-r2 | 6 | 2.19e-04 ± 7.0e-05 | -2.79e-04 | 2.14e-04 | +2.35e-09 ± 5.3e-10 | 32 |  |
| hmm-2 | 7 | 5.07e-02 ± 5.1e-04 | 4.93e-02 | 4.81e-02 | +5.05e-02 ± 4.8e-04 | 6 |  |
| hmm-3 | 7 | 2.81e-02 ± 3.5e-04 | 2.68e-02 | 2.58e-02 | +2.79e-02 ± 3.2e-04 | 15 |  |
| hmm-4 | 7 | 1.97e-02 ± 3.2e-04 | 1.87e-02 | 1.78e-02 | +1.95e-02 ± 2.9e-04 | 28 |  |
| hmm-6 | 7 | 3.43e-03 ± 2.8e-04 | 1.04e-04 | 3.39e-03 | +3.21e-03 ± 2.7e-04 | 66 |  |
| hmm-8 | 7 | 1.02e-02 ± 1.1e-03 | -9.60e-04 | 1.09e-02 | +1.00e-02 ± 1.1e-03 | 120 |  |
| hmm-12 | 7 | 2.32e-02 ± 1.9e-03 | -3.78e-03 | 1.95e-02 | +2.30e-02 ± 1.9e-03 | 276 |  |
| hmm-16 | 7 | 4.10e-02 ± 1.7e-03 | -5.83e-03 | 3.07e-02 | +4.08e-02 ± 1.8e-03 | 496 |  |
| markov-1 | 1 | 6.13e-02 ± 5.6e-04 | 6.10e-02 | 5.87e-02 | +6.11e-02 ± 5.3e-04 | 2 |  |
| markov-2 | 1 | 3.26e-02 ± 4.2e-04 | 3.13e-02 | 3.00e-02 | +3.23e-02 ± 3.9e-04 | 4 |  |
| markov-3 | 1 | 2.61e-02 ± 3.3e-04 | 2.50e-02 | 2.36e-02 | +2.59e-02 ± 3.0e-04 | 8 |  |
| markov-4 | 1 | 2.64e-02 ± 3.5e-04 | 2.50e-02 | 2.40e-02 | +2.62e-02 ± 3.2e-04 | 16 |  |
| markov-5 | 1 | 2.19e-02 ± 3.2e-04 | 2.00e-02 | 1.97e-02 | +2.16e-02 ± 2.9e-04 | 32 |  |
| markov-6 | 1 | 1.80e-02 ± 3.8e-04 | 1.67e-02 | 1.62e-02 | +1.78e-02 ± 3.3e-04 | 64 |  |
| markov-7 | 1 | 1.97e-02 ± 4.7e-04 | 1.61e-02 | 1.80e-02 | +1.95e-02 ± 4.2e-04 | 128 |  |
| markov-8 | 1 | 1.48e-02 ± 4.6e-04 | 9.34e-03 | 1.33e-02 | +1.46e-02 ± 4.3e-04 | 256 |  |

S* (parsimony, finite sample: smallest S with KL_HMM − ε < 2 SE) = >16; practical relevance (HMM_8 KL < 1e-4): False

### flw-T5000-s1  (Hankel rank of the truth: words <= 6 3, gap law 3; OOM_3 exact, QSM_2 r = 1 exact, no finite HMM)
| model | inits | held-out KL ± SE | train − truth | d_16 | KL − KL(QSM_2) ± SE (paired) | params | flags |
|---|---|---|---|---|---|---|---|
| truth | 1 | 0.00e+00 ± 0.0e+00 | 0.00e+00 | 0.00e+00 | -6.43e-04 ± 7.0e-05 | – |  |
| qsm-D2-r1 | 6 | 6.43e-04 ± 7.0e-05 | -4.93e-04 | 3.93e-04 | – | 16 |  |
| qsm-D2-r2 | 6 | 6.56e-04 ± 6.8e-05 | -5.08e-04 | 3.95e-04 | +1.29e-05 ± 6.4e-06 | 32 |  |
| hmm-2 | 7 | 5.05e-02 ± 5.0e-04 | 4.78e-02 | 4.79e-02 | +4.99e-02 ± 5.0e-04 | 6 |  |
| hmm-3 | 7 | 2.80e-02 ± 3.4e-04 | 2.36e-02 | 2.56e-02 | +2.73e-02 ± 3.5e-04 | 15 |  |
| hmm-4 | 7 | 1.91e-02 ± 2.5e-04 | 1.49e-02 | 1.69e-02 | +1.85e-02 ± 2.5e-04 | 28 |  |
| hmm-6 | 7 | 2.97e-03 ± 2.5e-04 | -9.07e-04 | 2.55e-03 | +2.33e-03 ± 2.1e-04 | 66 |  |
| hmm-8 | 7 | 5.16e-03 ± 4.6e-04 | -1.90e-03 | 3.03e-03 | +4.52e-03 ± 4.1e-04 | 120 |  |
| hmm-12 | 7 | 7.60e-03 ± 3.8e-04 | -5.29e-03 | 6.07e-03 | +6.95e-03 ± 3.5e-04 | 276 |  |
| hmm-16 | 7 | 1.60e-02 ± 8.7e-04 | -7.15e-03 | 9.67e-03 | +1.54e-02 ± 8.3e-04 | 496 |  |
| markov-1 | 1 | 6.12e-02 ± 5.4e-04 | 5.89e-02 | 5.85e-02 | +6.05e-02 ± 5.5e-04 | 2 |  |
| markov-2 | 1 | 3.22e-02 ± 4.1e-04 | 2.84e-02 | 2.97e-02 | +3.15e-02 ± 4.1e-04 | 4 |  |
| markov-3 | 1 | 2.59e-02 ± 3.2e-04 | 2.16e-02 | 2.34e-02 | +2.52e-02 ± 3.2e-04 | 8 |  |
| markov-4 | 1 | 2.61e-02 ± 3.3e-04 | 2.16e-02 | 2.37e-02 | +2.55e-02 ± 3.3e-04 | 16 |  |
| markov-5 | 1 | 2.22e-02 ± 3.2e-04 | 1.51e-02 | 2.01e-02 | +2.16e-02 ± 3.1e-04 | 32 |  |
| markov-6 | 1 | 1.96e-02 ± 3.8e-04 | 1.16e-02 | 1.77e-02 | +1.90e-02 ± 3.8e-04 | 64 |  |
| markov-7 | 1 | 2.07e-02 ± 4.0e-04 | 1.10e-02 | 1.88e-02 | +2.01e-02 ± 4.0e-04 | 128 |  |
| markov-8 | 1 | 1.64e-02 ± 3.9e-04 | 2.21e-03 | 1.45e-02 | +1.57e-02 ± 3.7e-04 | 256 |  |

S* (parsimony, finite sample: smallest S with KL_HMM − ε < 2 SE) = >16; practical relevance (HMM_8 KL < 1e-4): False

### flw-T5000-s2  (Hankel rank of the truth: words <= 6 3, gap law 3; OOM_3 exact, QSM_2 r = 1 exact, no finite HMM)
| model | inits | held-out KL ± SE | train − truth | d_16 | KL − KL(QSM_2) ± SE (paired) | params | flags |
|---|---|---|---|---|---|---|---|
| truth | 1 | 0.00e+00 ± 0.0e+00 | 0.00e+00 | 0.00e+00 | -4.77e-04 ± 1.0e-04 | – |  |
| qsm-D2-r1 | 6 | 4.77e-04 ± 1.0e-04 | -5.26e-04 | 3.82e-04 | – | 16 |  |
| qsm-D2-r2 | 6 | 4.77e-04 ± 1.0e-04 | -5.26e-04 | 3.82e-04 | +2.50e-09 ± 3.2e-10 | 32 |  |
| hmm-2 | 7 | 5.07e-02 ± 5.1e-04 | 4.72e-02 | 4.81e-02 | +5.02e-02 ± 4.9e-04 | 6 |  |
| hmm-3 | 7 | 3.36e-02 ± 7.9e-04 | 2.43e-02 | 3.11e-02 | +3.31e-02 ± 8.1e-04 | 15 |  |
| hmm-4 | 7 | 2.52e-02 ± 9.0e-04 | 1.67e-02 | 2.28e-02 | +2.47e-02 ± 9.2e-04 | 28 |  |
| hmm-6 | 7 | 7.84e-03 ± 7.6e-04 | 5.21e-04 | 7.21e-03 | +7.36e-03 ± 7.9e-04 | 66 |  |
| hmm-8 | 7 | 9.21e-03 ± 9.5e-04 | -2.00e-03 | 8.89e-03 | +8.74e-03 ± 9.8e-04 | 120 |  |
| hmm-12 | 7 | 1.70e-02 ± 1.2e-03 | -4.56e-03 | 1.50e-02 | +1.65e-02 ± 1.2e-03 | 276 |  |
| hmm-16 | 7 | 3.41e-02 ± 2.0e-03 | -7.67e-03 | 1.28e-02 | +3.36e-02 ± 2.1e-03 | 496 |  |
| markov-1 | 1 | 6.12e-02 ± 5.6e-04 | 5.91e-02 | 5.86e-02 | +6.08e-02 ± 5.4e-04 | 2 |  |
| markov-2 | 1 | 3.24e-02 ± 4.0e-04 | 2.85e-02 | 2.99e-02 | +3.19e-02 ± 3.8e-04 | 4 |  |
| markov-3 | 1 | 2.57e-02 ± 3.2e-04 | 2.22e-02 | 2.33e-02 | +2.52e-02 ± 2.9e-04 | 8 |  |
| markov-4 | 1 | 2.62e-02 ± 3.3e-04 | 2.18e-02 | 2.38e-02 | +2.57e-02 ± 3.1e-04 | 16 |  |
| markov-5 | 1 | 2.20e-02 ± 3.2e-04 | 1.85e-02 | 1.98e-02 | +2.16e-02 ± 3.0e-04 | 32 |  |
| markov-6 | 1 | 1.93e-02 ± 3.5e-04 | 1.58e-02 | 1.74e-02 | +1.88e-02 ± 3.0e-04 | 64 |  |
| markov-7 | 1 | 2.07e-02 ± 3.9e-04 | 1.53e-02 | 1.87e-02 | +2.02e-02 ± 3.3e-04 | 128 |  |
| markov-8 | 1 | 1.58e-02 ± 3.6e-04 | 7.36e-03 | 1.39e-02 | +1.53e-02 ± 3.2e-04 | 256 |  |

S* (parsimony, finite sample: smallest S with KL_HMM − ε < 2 SE) = >16; practical relevance (HMM_8 KL < 1e-4): False

### Decisive (T = 5e4, seeds 0 and 1): QSM_2 r = 1 vs HMM_k
| comparison | outcome | KL_HMM − KL_QSM seed 0 | seed 1 |
|---|---|---|---|
| HMM_2 (equal memory) | Positive | +5.05e-02 ± 5.0e-04 | +5.04e-02 ± 5.0e-04 |
| HMM_4 (equal filter cost) | Positive | +1.86e-02 ± 2.4e-04 | +1.86e-02 ± 2.5e-04 |

### E2a exact-law arm (HMM_S fitted to the exact law of 16-symbol words; QSM_2 r = 1 reaches 0)
| model | init | D_16 | d_8 | d_12 | d_16 | held-out KL ± SE | flags |
|---|---|---|---|---|---|---|---|
| hmm-4 | fit | 7.13e-03 | 3.64e-03 | 1.27e-02 | 1.63e-02 | 1.86e-02 ± 2.5e-04 | d_L rising at 16  |
| hmm-8 | fit | 3.67e-05 | 1.06e-05 | 4.62e-05 | 1.83e-04 | 4.20e-04 ± 5.3e-05 | d_L rising at 16  |
| hmm-12 | lump | 1.04e-05 | 1.44e-07 | 7.63e-07 | 1.04e-04 | 3.97e-04 ± 3.1e-05 | d_L rising at 16  |
| hmm-16 | lump | 5.32e-09 | 6.57e-10 | 4.80e-09 | 3.87e-08 | 1.70e-03 ± 9.2e-05 | horizon artefact d_L rising at 16  |

S*_approx (smallest S with held-out rate < 1e-4 within 2 SE) = >16; HMM_8 d_16 >= 1e-4: True; outcome: **Positive**

## T2  E2b: uniform renewal at equal filter cost

### renewal16-s0
| model | inits | held-out KL ± SE | train − truth | d_16 | params | flags |
|---|---|---|---|---|---|---|
| truth | 1 | 0.00e+00 ± 0.0e+00 | 0.00e+00 | 0.00e+00 | – |  |
| qsm-D2-r1 | 4 | 1.29e-02 ± 2.3e-04 | 1.21e-02 | 1.43e-02 | 16 |  |
| qsm-D4-r1 | 6 | 4.96e-03 ± 1.5e-04 | 3.98e-03 | 6.40e-03 | 64 |  |
| hmm-2 | 7 | 2.37e-02 ± 2.6e-04 | 2.26e-02 | 2.32e-02 | 6 |  |
| hmm-4 | 7 | 1.53e-02 ± 2.3e-04 | 1.43e-02 | 1.45e-02 | 28 |  |
| hmm-8 | 7 | 9.23e-03 ± 2.2e-04 | 8.06e-03 | 8.50e-03 | 120 |  |
| hmm-16 | 7 | 1.64e-03 ± 1.6e-04 | -1.83e-03 | 5.84e-04 | 496 | it-limit |
| markov-1 | 1 | 3.36e-02 ± 2.6e-04 | 3.29e-02 | 3.38e-02 | 2 |  |
| markov-2 | 1 | 3.16e-02 ± 3.2e-04 | 3.09e-02 | 3.15e-02 | 4 |  |
| markov-3 | 1 | 2.94e-02 ± 3.2e-04 | 2.83e-02 | 2.93e-02 | 8 |  |
| markov-4 | 1 | 2.74e-02 ± 2.9e-04 | 2.57e-02 | 2.71e-02 | 16 |  |
| markov-5 | 1 | 2.55e-02 ± 3.1e-04 | 2.35e-02 | 2.50e-02 | 32 |  |
| markov-6 | 1 | 2.34e-02 ± 2.9e-04 | 2.13e-02 | 2.30e-02 | 64 |  |
| markov-7 | 1 | 2.15e-02 ± 2.8e-04 | 1.93e-02 | 2.10e-02 | 128 |  |
| markov-8 | 1 | 1.95e-02 ± 3.2e-04 | 1.68e-02 | 1.92e-02 | 256 |  |

### renewal32-s0
| model | inits | held-out KL ± SE | train − truth | d_16 | params | flags |
|---|---|---|---|---|---|---|
| truth | 1 | 0.00e+00 ± 0.0e+00 | 0.00e+00 | 0.00e+00 | – |  |
| qsm-D2-r1 | 4 | 6.23e-03 ± 1.5e-04 | 6.94e-03 | 5.15e-04 | 16 |  |
| qsm-D4-r1 | 6 | 2.84e-03 ± 8.6e-05 | 1.77e-03 | 3.36e-04 | 64 |  |
| qsm-D8-r1 | 3 | 4.68e-03 ± 2.5e-04 | -5.51e-04 | 4.39e-04 | 256 |  |
| hmm-2 | 7 | 1.23e-02 ± 1.8e-04 | 1.28e-02 | 2.94e-03 | 6 |  |
| hmm-4 | 7 | 7.92e-03 ± 1.5e-04 | 8.64e-03 | 5.23e-04 | 28 |  |
| hmm-8 | 7 | 5.33e-03 ± 1.3e-04 | 5.48e-03 | 2.16e-04 | 120 |  |
| hmm-16 | 7 | 3.53e-03 ± 1.0e-04 | 2.95e-03 | 3.02e-04 | 496 |  |
| hmm-32 | 3 | 2.65e-03 ± 9.4e-05 | 1.30e-03 | 3.79e-04 | 2016 | it-limit |
| markov-1 | 1 | 1.80e-02 ± 1.8e-04 | 1.85e-02 | 8.16e-03 | 2 |  |
| markov-2 | 1 | 1.75e-02 ± 1.8e-04 | 1.79e-02 | 7.59e-03 | 4 |  |
| markov-3 | 1 | 1.70e-02 ± 2.0e-04 | 1.74e-02 | 7.02e-03 | 8 |  |
| markov-4 | 1 | 1.65e-02 ± 2.1e-04 | 1.70e-02 | 6.59e-03 | 16 |  |
| markov-5 | 1 | 1.59e-02 ± 2.2e-04 | 1.65e-02 | 6.13e-03 | 32 |  |
| markov-6 | 1 | 1.54e-02 ± 2.3e-04 | 1.58e-02 | 5.66e-03 | 64 |  |
| markov-7 | 1 | 1.50e-02 ± 2.3e-04 | 1.53e-02 | 5.25e-03 | 128 |  |
| markov-8 | 1 | 1.46e-02 ± 2.1e-04 | 1.46e-02 | 4.90e-03 | 256 |  |

### renewal32-s1
| model | inits | held-out KL ± SE | train − truth | d_16 | params | flags |
|---|---|---|---|---|---|---|
| truth | 1 | 0.00e+00 ± 0.0e+00 | 0.00e+00 | 0.00e+00 | – |  |
| qsm-D2-r1 | 0 | – | – | – | – |  |
| qsm-D4-r1 | 6 | 2.63e-03 ± 9.7e-05 | 2.18e-03 | 2.13e-04 | 64 |  |
| hmm-2 | 0 | – | – | – | – |  |
| hmm-4 | 0 | – | – | – | – |  |
| hmm-8 | 0 | – | – | – | – |  |
| hmm-16 | 7 | 3.36e-03 ± 9.9e-05 | 2.60e-03 | 1.87e-04 | 496 |  |
| markov-1 | 0 | – | – | – | – |  |
| markov-2 | 0 | – | – | – | – |  |
| markov-3 | 0 | – | – | – | – |  |
| markov-4 | 0 | – | – | – | – |  |
| markov-5 | 0 | – | – | – | – |  |
| markov-6 | 0 | – | – | – | – |  |
| markov-7 | 0 | – | – | – | – |  |
| markov-8 | 0 | – | – | – | – |  |

### Outcomes
| test | outcome | detail |
|---|---|---|
| b1 QSM_4 vs HMM_16, N = 32 (decisive) | **Positive** | seed 0: KL_HMM − KL_QSM = +6.83e-04 ± 1.3e-04; seed 1: KL_HMM − KL_QSM = +7.28e-04 ± 4.8e-05 |
| b2 control N = 16 | control OK (HMM_16 beats QSM_4) | KL_QSM − KL_HMM = +3.33e-03 ± 2.2e-04 |
| b3 HMM_32 vs QSM_8, N = 32 | non-decisive (HMM_32 not converged: iteration limit) | KL_QSM − KL_HMM = +2.03e-03 ± 2.3e-04 |

## T3  Ablations of the C-QSM memory (A1 switch, A2 dephasing)

### msar  (reference: re-evaluated seq4 C-QSM m = 1 + 2 anc)
| model | inits | held-out KL ± SE | ratio to C-QSM m=1 | KL − KL(C-QSM m=1) ± SE (paired) | params | filter dim | flags |
|---|---|---|---|---|---|---|---|
| C-QSM m=1 +2 anc (seq4) | 1 | 1.90e-02 ± 3.3e-04 | 1.00 | – | 269 | 4 |  |
| C-QSM m=2 +2 anc (seq4) | 1 | 1.74e-02 ± 2.4e-04 | 0.92 | -1.55e-03 ± 2.5e-04 | 329 | 16 |  |
| C-QSM m=1 +2 anc, dephased γ=1 | 5 | 2.07e-02 ± 3.2e-04 | 1.09 | +1.71e-03 ± 4.1e-04 | 269 | 4 | it-limit |
| switch anc2 S=2 | 5 | 4.97e-03 ± 2.0e-04 | 0.26 | -1.40e-02 ± 3.6e-04 | 494 | 2 | it-limit |
| switch B0 S=2 | 5 | 2.06e-02 ± 4.0e-04 | 1.09 | +1.62e-03 ± 3.6e-04 | 302 | 2 | it-limit |
| IOHMM_2 factored (seq4) | 1 | 9.79e-03 ± 2.5e-04 | 0.52 | -9.20e-03 ± 4.0e-04 | 2048 | 2 |  |
| Arch. B anc2 (seq4) | 1 | 1.53e-01 ± 1.3e-03 | 8.06 | +1.34e-01 ± 1.3e-03 | 215 | 1 |  |

| pre-registered pair | outcome | KL ratio | KL difference ± SE (paired) |
|---|---|---|---|
| A1 equal memory (switch anc2 S = 2) vs C-QSM m = 1 + 2 anc | **Negative for quantum memory** | 0.26 | -1.40e-02 ± 3.6e-04 |
| A1 parameter-matched (switch B0 S = 2) vs C-QSM m = 1 + 2 anc | **Negative for quantum memory** | 1.09 | +1.62e-03 ± 3.6e-04 |
| A2 dephased γ = 1 vs γ = 0 (C-QSM m = 1 + 2 anc) | **uses its memory classically** | 1.09 | +1.71e-03 ± 4.1e-04 |

### smsar16  (reference: re-evaluated seq4 C-QSM m = 1 + 2 anc)
| model | inits | held-out KL ± SE | ratio to C-QSM m=1 | KL − KL(C-QSM m=1) ± SE (paired) | params | filter dim | flags |
|---|---|---|---|---|---|---|---|
| C-QSM m=1 +2 anc (seq4) | 1 | 2.52e-02 ± 5.8e-04 | 1.00 | – | 269 | 4 |  |
| C-QSM m=2 +2 anc (seq4) | 1 | 2.50e-02 ± 4.9e-04 | 0.99 | -2.53e-04 ± 4.3e-04 | 329 | 16 |  |
| C-QSM m=1 +2 anc, dephased γ=1 | 5 | 2.68e-02 ± 4.9e-04 | 1.06 | +1.62e-03 ± 3.6e-04 | 269 | 4 | it-limit |
| switch anc2 S=2 | 5 | 1.51e-02 ± 2.6e-04 | 0.60 | -1.01e-02 ± 4.3e-04 | 494 | 2 | it-limit |
| switch anc2 S=4 | 3 | 1.62e-02 ± 3.9e-04 | 0.64 | -8.98e-03 ± 4.4e-04 | 1244 | 4 | it-limit |
| switch B0 S=2 | 5 | 2.38e-02 ± 4.0e-04 | 0.95 | -1.38e-03 ± 4.7e-04 | 302 | 2 | it-limit |
| IOHMM_2 factored (seq4) | 1 | 2.10e-02 ± 3.9e-04 | 0.83 | -4.22e-03 ± 4.4e-04 | 2048 | 2 |  |
| Arch. B anc2 (seq4) | 1 | 6.97e-02 ± 8.4e-04 | 2.77 | +4.45e-02 ± 7.1e-04 | 215 | 1 |  |

| pre-registered pair | outcome | KL ratio | KL difference ± SE (paired) |
|---|---|---|---|
| A1 equal memory (switch anc2 S = 2) vs C-QSM m = 1 + 2 anc | **Negative for quantum memory** | 0.60 | -1.01e-02 ± 4.3e-04 |
| A1 equal filter cost (switch anc2 S = 4) vs C-QSM m = 1 + 2 anc | **Negative for quantum memory** | 0.64 | -8.98e-03 ± 4.4e-04 |
| A1 parameter-matched (switch B0 S = 2) vs C-QSM m = 1 + 2 anc | **Negative for quantum memory** | 0.95 | -1.38e-03 ± 4.7e-04 |
| A2 dephased γ = 1 vs γ = 0 (C-QSM m = 1 + 2 anc) | **uses its memory classically** | 1.06 | +1.62e-03 ± 3.6e-04 |

## T4  E1: coherence knob (dephasing gamma) on the QSM memory

### renewal6-s0 D=4 r=16
| γ | inits | held-out KL ± SE | train − truth | flags |
|---|---|---|---|---|
| 0 | 4 | 3.05e-03 ± 7.6e-05 | 2.70e-03 |  |
| 0.05 | 4 | 4.71e-03 ± 9.6e-05 | 4.45e-03 |  |
| 0.1 | 4 | 6.27e-03 ± 1.2e-04 | 6.10e-03 |  |
| 0.25 | 4 | 1.05e-02 ± 1.9e-04 | 1.06e-02 |  |
| 0.5 | 4 | 1.52e-02 ± 3.0e-04 | 1.57e-02 |  |
| 1 | 4 | 2.75e-02 ± 4.0e-04 | 2.71e-02 |  |
| r = 1, γ = 0 | 4 | 2.64e-03 ± 6.8e-05 | 2.16e-03 |  |
| HMM_4 | 7 | 2.75e-02 ± 4.0e-04 | 2.71e-02 |  |

validity (γ = 1 = HMM_4 within 2 SE): True; train-NLL monotonicity violations: none; γ_½ = 0.211; r = 16 vs r = 1 at γ = 0: held-out ΔKL = +4.05e-04 ± 3.9e-05, train ΔNLL = +5.34e-04 (negative = stochasticity helps)

### renewal8-s0 D=4 r=16
| γ | inits | held-out KL ± SE | train − truth | flags |
|---|---|---|---|---|
| 0 | 4 | 5.86e-03 ± 1.7e-04 | 4.95e-03 |  |
| 0.05 | 4 | 7.66e-03 ± 1.9e-04 | 6.75e-03 |  |
| 0.1 | 4 | 9.03e-03 ± 2.0e-04 | 8.07e-03 |  |
| 0.25 | 4 | 1.16e-02 ± 2.5e-04 | 1.06e-02 |  |
| 0.5 | 4 | 1.50e-02 ± 3.1e-04 | 1.45e-02 |  |
| 1 | 4 | 2.45e-02 ± 4.0e-04 | 2.45e-02 |  |
| r = 1, γ = 0 | 4 | 3.33e-03 ± 9.7e-05 | 2.72e-03 |  |
| HMM_4 | 7 | 2.45e-02 ± 4.0e-04 | 2.45e-02 |  |

validity (γ = 1 = HMM_4 within 2 SE): True; train-NLL monotonicity violations: none; γ_½ = 0.282; r = 16 vs r = 1 at γ = 0: held-out ΔKL = +2.53e-03 ± 1.0e-04, train ΔNLL = +2.23e-03 (negative = stochasticity helps)

### renewal8-s0 D=2 r=4
| γ | inits | held-out KL ± SE | train − truth | flags |
|---|---|---|---|---|
| 0 | 4 | 2.19e-02 ± 3.7e-04 | 2.18e-02 |  |
| 0.05 | 4 | 2.40e-02 ± 3.8e-04 | 2.38e-02 |  |
| 0.1 | 4 | 2.55e-02 ± 3.9e-04 | 2.54e-02 |  |
| 0.25 | 4 | 2.90e-02 ± 4.0e-04 | 2.87e-02 |  |
| 0.5 | 4 | 3.33e-02 ± 3.9e-04 | 3.28e-02 |  |
| 1 | 4 | 4.28e-02 ± 3.8e-04 | 4.17e-02 |  |
| r = 1, γ = 0 | 4 | 2.19e-02 ± 3.7e-04 | 2.18e-02 |  |
| HMM_2 | 7 | 4.28e-02 ± 3.8e-04 | 4.17e-02 |  |

validity (γ = 1 = HMM_2 within 2 SE): False; train-NLL monotonicity violations: none; γ_½ = 0.352; r = 4 vs r = 1 at γ = 0: held-out ΔKL = +2.17e-09 ± 5.4e-10, train ΔNLL = +1.26e-13 (negative = stochasticity helps)
