# CNN for Time-Series Prediction

A self-directed learning project exploring how a **1D Convolutional Neural
Network (CNN)** can forecast a synthetic time series. The model is shown the
first 900 steps of a sequence and learns to predict the next 101.

> **Why this repo exists.** This is one of my early personal investigations into
> deep learning. I built it to develop an intuition for how convolutional
> networks — usually associated with images — apply to *sequential* data, and to
> get hands-on with the full loop: generating data, framing a supervised
> problem, designing an architecture, training, and critically interpreting the
> results. I keep it as a record of that progression rather than as a
> production tool.

![Forecast: actual vs. predicted continuation](forecast.png)

*The model is given everything left of the dashed line (steps 0–900) and
predicts the 101 steps to the right. Orange is the prediction; blue is the true
continuation.*

---

## Table of contents

- [Learning goals](#learning-goals)
- [The problem](#the-problem)
- [The data](#the-data-synthetic-by-design)
- [Model architecture](#model-architecture)
- [Results & interpretation](#results--interpretation)
- [Running it](#running-it)
- [Project structure](#project-structure)
- [What I learned](#what-i-learned)
- [Known limitations & next steps](#known-limitations--next-steps)

---

## Learning goals

When I started this, I wanted to answer a few concrete questions for myself:

1. **Can a CNN — not an RNN/LSTM — do sequence forecasting?** CNNs are most
   famous for images, and I wanted to see *why* 1D convolutions are a natural
   fit for time series (they learn local, shift-invariant temporal patterns).
2. **How do you frame "predict the future" as supervised learning?** i.e. the
   sliding-window idea: input = a window of the past, target = the next chunk.
3. **What does a model do when part of the signal is genuinely
   unpredictable?** I deliberately mixed a *learnable* trend with *irreducible*
   noise to see how the network behaves at that boundary.

## The problem

Given a window of past observations `x = [x₀, …, x₈₉₉]`, predict the
continuation `y = [x₉₀₀, …, x₁₀₀₀]` (101 future values) in a **single forward
pass** — a direct multi-step forecast, not an autoregressive one-step-at-a-time
loop.

## The data (synthetic by design)

Rather than download a dataset, I generate the data myself so I control exactly
how much of the signal is predictable. Each series is built step-by-step
(`src/random_walker.py`):

```
xₜ = xₜ₋₁ + sinₜ + (±fluctuation)
```

where:

- **`sinₜ`** is a sinusoidal trend, `amplitude · sin(2π · period · t / n_steps)`.
  With `period = 10` over `1000` steps, the trend completes ~10 oscillations —
  these are the regular waves you see in the plot. **This part is fully
  learnable.**
- **`±fluctuation`** is a random step (`uniform(min_step, max_step)` in a random
  direction). Accumulated over time this is a **random walk** — **pure noise,
  and fundamentally unpredictable.**

So every series is a deterministic, periodic backbone with stochastic jitter
layered on top. The split into (input, target) pairs happens in
`src/data_generator.py`. I generate **1000** independent series for training,
each with a different random seed, and one held-out series (different seed) for
testing.

> A subtle but important detail I fixed along the way: the **training and test
> data must use the same `amplitude`**. Early on they didn't, so the model was
> being evaluated on a series several times taller than anything it had seen —
> a small bug that produced misleadingly poor forecasts and a good lesson in
> train/test distribution matching.

## Model architecture

A compact 1D CNN built in Keras (`main.py`):

| Layer            | Output shape   | Params      | Role |
|------------------|----------------|-------------|------|
| `Input`          | (900, 1)       | 0           | one window, single feature |
| `Conv1D(64, k=2, relu)` | (899, 64) | 192      | learns local 2-step temporal patterns across 64 filters |
| `MaxPooling1D(2)`| (449, 64)      | 0           | downsamples, adds translation tolerance |
| `Flatten`        | (28736,)       | 0           | flattens for the dense head |
| `Dense(50, relu)`| (50,)          | 1,436,850   | mixes features globally |
| `Dense(101)`     | (101,)         | 5,151       | linear output: the 101-step forecast |

**Total: ~1.44M trainable parameters** — almost all of them in the first dense
layer, since flattening a length-449 × 64-channel feature map into a 50-unit
layer is where the weights concentrate. Trained with the **Adam** optimizer and
**mean-squared-error** loss for **200 epochs**.

## Results & interpretation

The forecast (right of the dashed line in the plot above) tracks the true
continuation closely: it gets the **phase** (when the next peak occurs) and the
**amplitude** (~54 predicted vs. ~51 actual) right. What it *can't* reproduce is
the fine step-to-step jitter — and that's the headline lesson:

> **The model learns the predictable structure (the sinusoidal trend) and
> sensibly "averages out" the unpredictable part (the random walk).** Training
> loss plateaus rather than going to zero, because zero loss is impossible by
> construction — you cannot predict noise. The smoother, slightly damped
> prediction curve is the network doing exactly the right thing.

This was the most valuable takeaway of the whole exercise: a flat loss isn't
always a failure to learn — sometimes it's the model correctly hitting the
**irreducible error floor** of the problem.

### A note on the forecast boundary

While investigating the plot I noticed what looked like a jump in the series
right where the forecast begins. Checking the raw numbers showed the *actual*
data is perfectly continuous there — the step across the boundary is ordinary,
and the split at step 900 is just where the series is sliced into (input,
target). The apparent jump was in the *predicted* line: the model emits all 101
future values **jointly**, with no constraint that its first prediction continue
smoothly from the last observed value, so the prediction started slightly offset
from the input. The plot now **anchors the forecast line to the last observed
point** so it reads honestly as a continuation. This is an inherent property of
direct multi-step forecasting (an autoregressive model would instead start
exactly from the last known value) — and a good reminder to verify a surprising
visual against the underlying data before trusting it.

## Running it

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
uv sync          # create .venv and install pinned dependencies
uv run main.py   # generate data, train the CNN, and write forecast.png
```

The run prints the model summary and per-epoch loss, then saves `forecast.png`
(input window + actual vs. predicted continuation, with the forecast boundary
marked).

Tested with Python 3.13, TensorFlow 2.21, and NumPy 2.4. Dependency versions are
pinned in `uv.lock` for reproducibility.

## Project structure

```
.
├── main.py                 # build, train, and evaluate the CNN; plot the forecast
├── src/
│   ├── random_walker.py    # RandomWalk: generates one sinusoid + random-walk series
│   └── data_generator.py   # Data: builds many (input, future) training pairs
├── pyproject.toml          # project metadata and dependencies
├── uv.lock                 # pinned, reproducible dependency tree
└── forecast.png            # output plot (regenerated on each run)
```

## What I learned

- **1D convolutions are a real tool for sequences**, not just an image trick —
  the `kernel_size=2` filters learn local "what comes next" patterns that
  generalise across the whole window.
- **Framing matters as much as modelling.** Most of the thinking went into the
  sliding-window setup and generating data with a *known* predictable/random
  split, not into the network itself.
- **Read the loss curve critically.** A non-zero plateau can be the correct
  answer, not a bug.
- **Train/test distributions must match** — the amplitude bug was a concrete,
  memorable example of how a tiny mismatch corrupts evaluation.

## Known limitations & next steps

This is a learning artefact, so several things are intentionally simple. If I
were to extend it, the natural progression would be:

- **Add a validation split and early stopping** instead of a fixed 200 epochs,
  and plot training vs. validation loss.
- **Quantify the forecast** with a metric (RMSE/MAE) against the true
  continuation, rather than judging it by eye.
- **Compare architectures** — stack more `Conv1D` layers, add dropout, or
  benchmark against an LSTM and a naive baseline (e.g. "repeat the last value").
- **Try real data** — apply the same sliding-window framing to an actual
  time series (weather, finance, sensor data) where the signal/noise split is
  unknown.
- **Tidy the data pipeline** — `Data` mutates instance state across calls; a
  cleaner functional generator would scale better.
