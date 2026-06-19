"""Run the full model-comparison study.

Trains every model on a shared train/val/test split, evaluates each with the
same metrics, quantifies uncertainty (MC Dropout) for the neural models, and
writes a results table plus comparison plots to ``results/``.

Quick smoke-test mode:  STUDY_QUICK=1 uv run run_study.py
Full study:             uv run run_study.py
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data import build_splits
from src import models as M
from src import metrics as Mt
from src.uncertainty import mc_dropout_predict, intervals

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
tf.keras.utils.set_random_seed(SEED)

QUICK = os.environ.get("STUDY_QUICK") == "1"
if QUICK:
    N_STEPS, WINDOW = 120, 100
    N_TRAIN, N_VAL, N_TEST = 24, 8, 16
    EPOCHS, PATIENCE, MC_PASSES = 3, 2, 5
else:
    N_STEPS, WINDOW = 400, 300
    N_TRAIN, N_VAL, N_TEST = 300, 75, 150
    EPOCHS, PATIENCE, MC_PASSES = 80, 8, 50

PERIOD = 10
AMPLITUDE = 1.0
SEASON = N_STEPS // PERIOD          # one full oscillation, in steps
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train, val, test = build_splits(
        N_TRAIN, N_VAL, N_TEST, N_STEPS, WINDOW,
        amplitude=AMPLITUDE, period=PERIOD)
    Xtr, Ytr = train
    Xva, Yva = val
    Xte, Yte = test
    horizon = Ytr.shape[1]
    print(f"data: train{Xtr.shape} val{Xva.shape} test{Xte.shape} "
          f"horizon={horizon} season={SEASON}")

    rows = []                       # one metrics dict per model
    horizon_curves = {}             # name -> per-horizon RMSE
    keras_models = {}               # name -> trained model (for uncertainty plot)
    histories = {}                  # name -> Keras training history (for learning curves)

    # ---- baselines ---------------------------------------------------------
    for base in (M.Persistence(), M.SeasonalNaive(SEASON), M.LinearRidge()):
        base.fit(Xtr, Ytr)
        pred = base.predict(Xte, horizon)
        rows.append(dict(model=base.name, params=0,
                         rmse=Mt.rmse(Yte, pred), mae=Mt.mae(Yte, pred),
                         smape=Mt.smape(Yte, pred),
                         picp=float("nan"), mpiw=float("nan")))
        horizon_curves[base.name] = Mt.per_horizon_rmse(Yte, pred)
        print(f"[baseline] {base.name:14s} rmse={rows[-1]['rmse']:.3f}")

    # ---- neural networks ---------------------------------------------------
    for name, builder in M.NEURAL_BUILDERS.items():
        model = builder(WINDOW, horizon)
        hist = M.train_keras(model, (Xtr, Ytr), (Xva, Yva),
                             epochs=EPOCHS, patience=PATIENCE)
        pred = model.predict(Xte, verbose=0)
        mean, std = mc_dropout_predict(model, Xte, n_passes=MC_PASSES)
        lo, hi = intervals(mean, std)
        rows.append(dict(model=name, params=model.count_params(),
                         rmse=Mt.rmse(Yte, pred), mae=Mt.mae(Yte, pred),
                         smape=Mt.smape(Yte, pred),
                         picp=Mt.picp(Yte, lo, hi), mpiw=Mt.mpiw(lo, hi)))
        horizon_curves[name] = Mt.per_horizon_rmse(Yte, pred)
        keras_models[name] = model
        histories[name] = hist.history
        print(f"[neural]   {name:14s} rmse={rows[-1]['rmse']:.3f} "
              f"picp={rows[-1]['picp']:.2f} epochs={len(hist.history['loss'])}")

    rows.sort(key=lambda r: r["rmse"])
    _write_csv(rows)
    _print_table(rows)
    _plot_comparison(rows)
    _plot_horizon_curves(horizon_curves, rows)
    _plot_learning_curves(histories)
    _plot_uncertainty(keras_models, rows, Xte, Yte, horizon)
    print(f"\nWrote results and plots to {RESULTS_DIR}/")


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
_COLS = ["model", "rmse", "mae", "smape", "picp", "mpiw", "params"]


def _write_csv(rows):
    with open(os.path.join(RESULTS_DIR, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r[c] for c in _COLS})


def _fmt(v):
    if isinstance(v, float):
        return "n/a" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def _print_table(rows):
    print("\n=== Test-set results (sorted by RMSE) ===")
    header = f"{'model':14s} {'RMSE':>7s} {'MAE':>7s} {'sMAPE%':>7s} " \
             f"{'PICP':>6s} {'MPIW':>7s} {'params':>9s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:14s} {_fmt(r['rmse']):>7s} {_fmt(r['mae']):>7s} "
              f"{_fmt(r['smape']):>7s} {_fmt(r['picp']):>6s} "
              f"{_fmt(r['mpiw']):>7s} {r['params']:>9d}")


def _plot_comparison(rows):
    names = [r["model"] for r in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(names, [r["rmse"] for r in rows], color="C0")
    plt.ylabel("test RMSE (lower is better)")
    plt.title("Model comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=120)
    plt.close()


def _plot_horizon_curves(curves, rows):
    plt.figure(figsize=(10, 5))
    for r in rows:                          # plot in ranked order for a clean legend
        name = r["model"]
        plt.plot(curves[name], label=name, linewidth=1.3)
    plt.xlabel("forecast step (further ahead ->)")
    plt.ylabel("RMSE at step")
    plt.title("Error growth over the forecast horizon")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "horizon_error.png"), dpi=120)
    plt.close()


def _plot_learning_curves(histories):
    n = len(histories)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                             squeeze=False)
    flat = axes.flatten()
    for ax, (name, h) in zip(flat, histories.items()):
        ax.plot(h["loss"], label="train", linewidth=1.3)
        ax.plot(h["val_loss"], label="validation", linewidth=1.3)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    for ax in flat[n:]:                 # hide unused axes
        ax.set_visible(False)
    fig.suptitle("Training vs. validation loss (log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"), dpi=120)
    plt.close(fig)


def _plot_uncertainty(keras_models, rows, Xte, Yte, horizon):
    # best neural model = first ranked row whose model has a trained net
    best = next(r["model"] for r in rows if r["model"] in keras_models)
    model = keras_models[best]

    sample = Xte[:1]
    mean, std = mc_dropout_predict(model, sample, n_passes=50)
    lo, hi = intervals(mean[0], std[0])

    history = sample[0].flatten()
    truth = Yte[0]
    split = len(history)
    fx = np.arange(split - 1, split + horizon)
    anchor = history[-1]

    plt.figure(figsize=(12, 5))
    plt.plot(np.concatenate([history, truth]), label="actual", linewidth=1)
    plt.plot(fx, np.concatenate([[anchor], mean[0]]),
             color="C1", label=f"{best} mean forecast", linewidth=1)
    plt.fill_between(fx[1:], lo, hi, color="C1", alpha=0.25,
                     label="95% MC-Dropout interval")
    plt.axvline(split, color="grey", linestyle="--", label="forecast start")
    plt.title(f"Forecast with uncertainty -- {best}")
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "uncertainty.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
