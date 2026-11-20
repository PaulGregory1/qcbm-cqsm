# Architecture B — Minimal Baseline Spec

## Goal and scope

One end-to-end run of Architecture B on a single OU process, with at most 10 qubits and exact statevector simulation, to decide whether the full comparison study is worth launching.

The question is narrow: can a conditional QCBM with 5 input and 5 output qubits learn the 32 × 32 transition kernel of a discretized OU process well enough to reproduce its stationary law and autocorrelation?

- **In scope:** one process, two encodings (standard binary and Gray code), one circuit, one loss, three seeds per encoding, cheap closed-form reference points.
- **Not in scope:** the hypernetwork QCBM, time dependence, trained classical baselines, ablations, finite shots. They belong to the full study.

The result is a go/no-go note plus the pieces the full study reuses: data generator, exact kernel, circuit simulator, loss and metrics.

## Process and data

A stationary OU process, simulated through its exact one-step recursion and encoded on 5 bits, with a ground-truth kernel computed in closed form.

```latex
x_{t+1} = c\,x_t + m + s\,\xi_t,\qquad c = e^{-\kappa\Delta t},\quad m = \mu(1-c),\quad s^2 = \frac{\sigma^2}{2\kappa}\left(1-e^{-2\kappa\Delta t}\right)
```

| Setting | Value |
| --- | --- |
| κ, μ, σ, Δt | 1, 0, 1, 0.1 |
| Resulting c, s, stationary sd | 0.905, 0.301, 0.707 |
| Encoding | uniform fixed-point, MSB first, range \[−3, 3\], 5 bits; run in standard binary and in Gray code |
| Grid | 32 bins of width 0.1875; s spans about 1.6 bins |
| Data | 2,000 trajectories × 50 steps from the stationary law, about 98,000 transitions |
| Clipping | values outside the range go to the edge bins; log the clipped fraction |

**Gray-code variant.** The same 32 bins get new bit labels: the MSB is kept and each lower bit is the XOR of two neighbouring binary bits.

```latex
g_1 = b_1,\qquad g_k = b_k \oplus b_{k-1}\quad (k = 2,\dots,5)
```

Neighbouring bins then differ by exactly one bit. In binary, a one-bin move can flip every bit (01111 → 10000), so the circuit must learn carries; in Gray code it is a single bit flip. Loss and metrics use decoded values, so both runs are scored the same way; only the circuit's task changes.

**Ground truth.** With bin edges u\_j and centres z\_i, the exact kernel integrates the Gaussian transition over each bin, the two edge bins absorbing the tails:

```latex
P^\star_{ij} = \Phi\!\left(\frac{u_{j+1} - c\,z_i - m}{s}\right) - \Phi\!\left(\frac{u_j - c\,z_i - m}{s}\right)
```

The empirical pair counts N\_ij and the empirical marginal of x\_t are the only data the models see.

## Circuit

Ten qubits, no ancilla: 5 input qubits hold x\_t, 5 output qubits are measured to give x\_{t+1}.

```latex
P_\theta(\mathbf b' \mid \mathbf b) = \sum_{\mathbf c \in \{0,1\}^5} \Big|\langle \mathbf c, \mathbf b' \,|\, U(\theta)\,|\mathbf b, \mathbf 0\rangle\Big|^2
```

**Input qubits are controls only.** They are never rotated, so the input register stays in |b⟩ and the trace is trivial. Each row is then the Born distribution of a 5-qubit output state:

```latex
P_\theta(\mathbf b' \mid \mathbf b) = \big|\langle \mathbf b' \,|\, U_{\mathbf b}(\theta)\,|\mathbf 0\rangle\big|^2
```

The simulator can therefore hold 32 output states of 32 amplitudes each, one per input. This is exact for the 10-qubit circuit, not an approximation. The rows are still general: the input register plays the role of the auxiliary system, so the kernel is not restricted to doubly stochastic matrices.

**Ansatz (one layer).** Real gates only, so amplitudes stay real:

| Block | Gates | Angles |
| --- | --- | --- |
| Local | R\_y on each output qubit | 5 |
| Input → output | controlled R\_y from every input bit to every output bit | 25 |
| Output chain | controlled R\_y between neighbouring output bits, both directions | 8 |

Three layers plus a final R\_y layer give 3 × 38 + 5 = 119 angles. For reference, a free 32 × 32 stochastic matrix has 992 parameters.

**Initialization.** In the first layer, set the controlled R\_y from input bit q to output bit q to π, which copies x\_t to the output. The model then starts near the identity kernel, close to the truth since c ≈ 0.9. All other angles start at N(0, 0.2²).

Avoid an all-zero start: the output then stays |00000⟩ for every input, a stationary point where most gradients vanish.

## Training

A debiased Sinkhorn divergence between two exact weighted measures on the 1,024 grid points (z\_i, z\_j), with gradients from automatic differentiation.

```latex
w_{ij}(\theta) = \hat\pi_i\,P_\theta(j \mid i),\qquad v_{ij} = \frac{N_{ij}}{\sum_{kl} N_{kl}},\qquad \mathcal L(\theta) = \mathcal S_\varepsilon(w(\theta), v)
```

Here π̂ is the empirical marginal of x\_t. The cost between points is the squared Euclidean distance on decoded values.

| Setting | Value |
| --- | --- |
| Blur ε | 0.01 (decoded units squared; one bin width squared is 0.035) |
| Sinkhorn solver | log-domain; potentials warm-started from the previous step, 25 iterations per step |
| Gradient | autodiff through the circuit; Sinkhorn potentials held fixed (envelope theorem) |
| Optimizer | Adam, learning rate 0.03 |
| Steps | 800, or stop when the loss changes by less than 0.1% over 100 steps |
| Seeds | 3 per encoding (data fixed, initialization varies) |

With exact probabilities, the model side needs no sampling, so the loss and its gradient carry no shot noise. Gradients equal what parameter shift would give on hardware with infinite shots.

## Reference points

Four kernels that need no training, computed from the same data, to put the QCBM's numbers on a scale. They are anchors, not the baselines of the full study.

| Reference | How it is built | What it tells us |
| --- | --- | --- |
| Exact kernel P\* | closed form above | the target; its metrics are the best possible |
| Gaussian AR(1) | least-squares fit of c, m, s on bin centres, then discretized | near-best parametric fit; the QCBM should approach it |
| Tabular counts | (N\_ij + 0.1) normalized per row | nonparametric, 992 free numbers; strong in the bulk, noisy in the tails |
| i.i.d. oracle | every row equals the empirical marginal | the best a static, Coyle-style QCBM can do; zero autocorrelation |

The QCBM must clearly beat the i.i.d. oracle; landing between it and the AR(1) fit is the interesting zone.

## Outputs

Six numbers per model and four figures, all computed exactly from the 32 × 32 kernels, so no test-set sampling noise.

**Metrics** (for the QCBM and each reference):

| Metric | Definition |
| --- | --- |
| Weighted TV | Σ\_i π\*(i) · ½ Σ\_j \|P(j\|i) − P\*(j\|i)\| |
| Tail TV | mean row TV over states beyond 2 stationary sd (\|z\| > 1.41) |
| Stationary TV | TV between the kernel's leading left eigenvector and π\* |
| Lag-1 autocorrelation | from the kernel's stationary chain; compare with the exact chain's value |
| ACF error | max absolute error over lags 1 to 20 |
| Conditional-mean slope | slope of E\[x\_{t+1} \| x\_t\] against x\_t; compare with c = 0.905 |

**Figures.**

1. Kernel heatmaps side by side: P\*, QCBM, tabular counts.
2. Conditional mean ± one sd against x\_t: truth, QCBM, AR(1).
3. Autocorrelation, lags 1 to 20: truth, QCBM, i.i.d. oracle.
4. Training loss per step, one line per seed.

Report every number and figure for both encodings. Also save the learned angles and kernels, so the full study can reuse them as warm starts.

## Go/no-go criteria

Go to the full study when all five checks pass on at least 2 of 3 seeds for at least one encoding; the thresholds are first guesses, to revisit once the reference numbers are known.

- [ ] Stationary TV below 0.05.
- [ ] Lag-1 autocorrelation within 0.02 of the exact chain's value.
- [ ] Conditional-mean slope within 0.03 of c = 0.905.
- [ ] Weighted TV at most half that of the i.i.d. oracle.
- [ ] Weighted TV varies by less than 20% across seeds.

**If a check fails, look here first.**

| Symptom | Likely cause | First thing to try |
| --- | --- | --- |
| Loss stalls early | stationary start or flat gradients | check the copy initialization; lower the noise on other angles |
| Good bulk, bad tails | tail rows carry little weight in the loss | inspect tail rows; try ε = 0.005 |
| Right mean, wrong spread | ansatz cannot spread mass across a carry (e.g. 01111 → 10000) | compare with the Gray-code run; add one layer |
| Slope near 1 | stuck at the identity start | longer training or a higher learning rate |
| Large seed-to-seed spread | rugged landscape | more seeds; report the best and the median |

## Out of scope and effort

About 2 to 3 working days for one person, one script of a few hundred lines; everything else waits for the full study.

**Deferred to the full study:** hypernetwork QCBM, time-varying volatility, fat tails, MPS and neural baselines, ablations (ancillas, depth, loss), finite shots, hardware.

**One optional extra (half a day):** the single-register model on 5 qubits, P(b'|b) = |⟨b'|U|b⟩|². Its kernel is doubly stochastic, so its stationary law must come out uniform. It shows in one plot why the input register is needed.

**Stack:** Python with NumPy, SciPy, JAX (autodiff) and Matplotlib. No quantum SDK needed at this stage.
