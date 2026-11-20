# Conditional QCBMs and quantum memories for time series

Exact-statevector study of conditional quantum circuit Born machines (QCBMs) as transition kernels of
discretized time series, and of quantum sequence models with a never-measured memory register. Values sit on
5-bit grids (32 bins). Every likelihood and KL rate is computed exactly against the true process. The
circuits are compared with classical models of matched size: HMMs, input-output HMMs, Markov chains, MLPs,
mixtures, conditional RBMs, Markov-switching AR and GARCH.

## Setup

Python 3.12, CPU only.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
.venv/bin/python -m pytest -q          # 226 tests, about 5 min
```

## Reproduce

```bash
.venv/bin/python scripts/reproduce.py   # about 25 min on a laptop
```

This rebuilds every summary (`runs/**/summary.json`), table, figure (`figs/`) and note (`reports/*.md`) from
the fits stored in `runs/`. No training happens. Afterwards `git status` shows only floating-point noise
(relative error below 1e-5) and timing fields. `--only seq7 seq11` limits the run to some sequences.

To retrain, run `scripts/run_<seq>.py` for that sequence; `--help` lists its options. Fits are resumable and
skip finished runs, and most runners accept `--workers` and `--smoke`. The baseline (`run_all.py`) takes
about 8 minutes, and sequences 1 to 6 take hours on a multi-core CPU. Sequences 7 to 10 ran on a GPU node, and
sequence 11 ran overnight on a laptop. The C-QSM m = 2 fits of sequence 9 need a GPU.

## Main results

KL rates are held-out KL-divergence rates against the true process, in nats per step. "> 2 SE" means a
paired difference larger than twice its standard error over 20 segments of the held-out path.

| Sequence | Question | Result | Source |
| --- | --- | --- | --- |
| Baseline | Can a 10-qubit conditional QCBM learn the 32 × 32 kernel of an Ornstein-Uhlenbeck process? | GO, Gray encoding only: 2 of 3 seeds pass every check, binary 0 of 3. | `runs/summary.json` |
| 1–2 | What limits the kernel? | Ancilla qubits; likelihood training beats Sinkhorn. On Merton and VAR(2D) targets the circuit stays behind transition counts. | `runs/seq1`, `runs/seq2` |
| 3 | Does a quantum memory compress hidden states (QSM vs HMM)? | Renewal processes: the QSM beats the HMM with the same memory dimension in 10 of 10 (N, D) pairs. Two-regime process: the HMM wins. | `runs/seq3` |
| 4 | Conditional QSM (C-QSM) on regime-switching AR | MS-AR: KL 1.7e-2, 8.8× better than the memoryless kernel and 8.4× better than the QSM. Real daily returns: loses to GARCH-t by > 2 SE on 5 of 6 datasets. | `runs/seq4`, `runs/seq4_real` |
| 5 | Can a stochastic or dephased QSM beat positive classical models? | FLW process: one qubit is exact, while HMM_2 and HMM_4 stay at 5.0e-2 and 1.9e-2 (> 2 SE, 2 seeds). Renewal N = 32: QSM_4 beats HMM_16 (2 seeds). On regimes the memory is used classically. | `runs/seq5` |
| 6 | QCBM kernels vs compact classical kernels at equal size | Synthetic targets: QCBM ahead of the MLP (8 to 5), behind the Gaussian mixture (0 to 9), level with the conditional RBM (3 to 4). Real data: 7 wins, 70 ties, 25 losses. | `runs/seq6` |
| 7 | QSM_4 vs larger HMMs, renewal N = 32 | Better than HMM_16 on 5 of 5 strings (all > 2 SE). Better than early-stopped HMM_32 on 5 of 5 (3 at > 2 SE), median KL ratio 0.91. | `runs/seq7` |
| 8 | Does the C-QSM need coherence? | No consistent gain from coherence (5 datasets per target, > 2 SE counts). msar: the dephased C-QSM is level or better (m = 2: better on 4 of 5). smsar16: the coherent one is better on 4 of 5 (m = 1). | `runs/seq8` |
| 9 | Four FX rates on 8 qubits | Test log-score, nats per day: C-QSM m = 2 −4.706 and m = 1 −4.726, circuit kernel −4.752, conditional RBM −4.750, GARCH-t + DCC copula −4.550. | `runs/seq9/real/fx4-q/eval.json` |
| 10 | Parsimony on rare events (exact tail scores) | At equal parameter budget a classical family is better than the circuit in nearly every comparison (at most 1 of 60 the other way). | `runs/seq10` |
| 11 | Persistence and volatility-contrast grid of MS-AR | The classical switch beats the C-QSM by > 2 SE on 17 of 17 datasets, excluding the A = 0.5 negative control. | `runs/seq11` |

Taken together: a conditional QCBM is a compact and competitive parametrization of transition kernels, with
no edge over well-chosen classical families. The quantum memory wins only on processes built around phase
structure (renewal, FLW), not on volatility regimes of returns.

## Layout

| Path | Content |
| --- | --- |
| `src/archb/` | simulators (conditional QCBM, QSM, C-QSM), targets, classical baselines, training, metrics |
| `scripts/run_*.py` | training runners, one per sequence |
| `scripts/*_summary.py`, `*_report.py` | summaries, figures and notes from the stored fits |
| `scripts/reproduce.py` | all of the above in order |
| `runs/` | stored fits (`result.json`, `params.npz`) and summaries |
| `figs/` | figures |
| `data/raw/` | daily FRED series: DEXUSEU, DEXUSUK, DEXSZUS, DEXJPUS, NASDAQCOM |
| `docs/spec.md` | settings of the baseline |
| `tests/` | unit tests |

Conventions: 5 bits, most significant bit first. Bin i has binary label i and Gray label i ^ (i >> 1). Kernel
rows and columns are bin indices, so `P[i, j] = P(x_{t+1} in bin j | x_t in bin i)` whatever the encoding.
