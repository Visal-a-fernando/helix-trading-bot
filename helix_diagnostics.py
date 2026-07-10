"""
helix_diagnostics.py
====================

Turns the multi-model debate about the HELIX sizing formula into numbers.

It answers the one question the whole debate reduced to:
    "What fraction of trades exit on the timeout barrier, what is their mean R,
     and how badly does binary Kelly mis-size versus sizing on the real payoff
     distribution?"

and it runs the two validation gates the models agreed on:
    - the zero-alpha test (replace alpha with noise; sizes and p must collapse)
    - a purged, embargoed walk-forward split (no look-ahead through overlapping
      triple-barrier labels)
plus a calibration report (Brier / ECE / reliability), because Kelly reacts to
the absolute value of p, not its ranking.

CONVENTIONS
-----------
A "trade ledger" is a pandas DataFrame with one row per closed trade and:
    R            : realized return in RISK units. A full stop-out is -1.0,
                   a take-profit at reward:risk b is +b. This is pnl / risk_per_trade.
    exit_reason  : one of {"tp", "sl", "timeout"} (case-insensitive).
    t0           : integer bar index when the trade was entered (event time).
    t1           : integer bar index when the label/trade resolved (label end).
    p            : (optional) meta-model win-probability used at entry.

Everything is numpy/pandas; sklearn/scipy are used only where noted.

This is an engineering/quant diagnostic for research and backtesting. It is not
financial advice. Validate on your own data and dry-run before risking capital.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Core Kelly primitives
# ---------------------------------------------------------------------------

def binary_kelly(p: float, b: float, c: float = 0.0, exact_cost: bool = False) -> float:
    """Binary Kelly fraction (full, unscaled).

    Default: the practical form f = p - (1-p)/b - c/b, a linear cost haircut.
    exact_cost=True uses the correct asymmetric-binary Kelly when a round-trip
    cost c is paid on BOTH outcomes (win pays b-c, loss costs 1+c):
        f = [p(b-c) - (1-p)(1+c)] / [(b-c)(1+c)]
    The linear form overstates f when c is non-trivial relative to b; for crypto
    (c ~ 0.001) the two agree to ~4 decimals, but expose the option so it can't
    silently mislead when costs are large. (DeepSeek and Gemini both flagged this.)
    """
    if b <= 0:
        return 0.0
    if exact_cost:
        W, L = b - c, 1.0 + c
        if W <= 0 or L <= 0:
            return 0.0
        return (p * W - (1.0 - p) * L) / (W * L)
    return p - (1.0 - p) / b - c / b


def log_growth_at(f: float, R: np.ndarray) -> float:
    """Expected per-trade log growth E[log(1 + f*R)] on an empirical R sample.

    Returns -inf if any 1 + f*R <= 0 (a bet that can be wiped out on one trade).
    """
    R = np.asarray(R, dtype=float)
    R = R[np.isfinite(R)]
    if R.size == 0:
        return 0.0
    v = 1.0 + f * R
    if np.any(v <= 0):
        return -np.inf
    return float(np.mean(np.log(v)))


def empirical_kelly(R: np.ndarray, f_max: float = 2.0, shrink_bootstrap: bool = False,
                    n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """Growth-optimal *full* Kelly fraction on the EMPIRICAL payoff distribution.

    Numerically maximizes E[log(1 + f*R)] over f in (0, f_upper). This is the
    "generalized Kelly" that respects a 3-outcome / continuous payoff, including
    timeout exits, without pretending the payoff is binary.

    CAVEAT (raised by ChatGPT): this fraction is fit IN-SAMPLE on the same R it is
    scored on, so it is biased UPWARD -- the optimizer exploits lucky trades. Do
    not size live off this number directly. Two guards are provided:
      * f_max default lowered to 2.0 so a thin-loss ledger (small |min(R)|) can't
        push the search bound to absurd leverage;
      * shrink_bootstrap=True returns a conservative lower-ish fraction: the mean
        of per-resample optimal f minus one std (a crude uncertainty haircut).

    Returns (f_star, growth_at_f_star). f_star is FULL Kelly; apply lambda after.
    """
    R = np.asarray(R, dtype=float)
    R = R[np.isfinite(R)]
    if R.size == 0:
        return 0.0, 0.0
    min_R = R.min()
    f_upper = (0.999 / (-min_R)) if min_R < 0 else f_max
    f_upper = min(f_upper, f_max)
    if f_upper <= 0:
        return 0.0, 0.0

    def _argmax_f(sample):
        grid = np.linspace(0.0, f_upper, 4001)[1:]
        g = np.array([log_growth_at(f, sample) for f in grid])
        i = int(np.nanargmax(g))
        lo = grid[max(0, i - 1)]
        hi = grid[min(len(grid) - 1, i + 1)]
        fine = np.linspace(lo, hi, 501)
        gg = np.array([log_growth_at(f, sample) for f in fine])
        j = int(np.nanargmax(gg))
        return float(fine[j]), float(gg[j])

    f_star, g_star = _argmax_f(R)
    if shrink_bootstrap:
        rng = np.random.default_rng(seed)
        fs = []
        for _ in range(n_boot):
            samp = R[rng.integers(0, R.size, R.size)]
            try:
                fs.append(_argmax_f(samp)[0])
            except Exception:
                pass
        if fs:
            f_shrunk = max(0.0, float(np.mean(fs) - np.std(fs)))
            return f_shrunk, log_growth_at(f_shrunk, R)
    return f_star, g_star


def gaussian_kelly(R: np.ndarray) -> float:
    """Mean-variance Kelly approximation f ~= E[R] / Var(R) (Gemini's proposal).

    Convenient, but assumes roughly Gaussian returns. Crypto returns are fat
    tailed, so this OVER-sizes when variance is underestimated in calm periods
    before a jump. Reported for comparison, not recommended as the primary rule.
    """
    R = np.asarray(R, dtype=float)
    R = R[np.isfinite(R)]
    var = np.var(R)
    if var <= 0:
        return 0.0
    return float(np.mean(R) / var)


# ---------------------------------------------------------------------------
# 2. Timeout structure + the binary-vs-empirical sizing gap  (the crux)
# ---------------------------------------------------------------------------

@dataclass
class TimeoutReport:
    n_trades: int
    timeout_rate: float
    tp_rate: float
    sl_rate: float
    mean_timeout_R: float
    timeout_R_quantiles: dict
    win_rate: float                 # P(R > 0)
    b_design: float                 # u/d you configured
    b_empirical: float              # realized mean-win / mean-|loss|
    f_binary_full: float            # full binary Kelly using p_used and b_design
    f_empirical_full: float         # full empirical (generalized) Kelly
    f_gaussian_full: float
    p_used: float
    growth_empirical: float         # E[log(1+f*R)] at empirical Kelly
    growth_binary: float            # E[log(1+f*R)] at binary Kelly fraction
    growth_capture: float           # growth_binary / growth_empirical  (ratio; see caveat)
    growth_gap_abs: float           # growth_empirical - growth_binary  (absolute, per trade)
    sizing_direction: str           # binary Kelly OVER- or UNDER-sizes vs empirical
    structural_trigger: float       # timeout_rate * |mean_timeout_R - (-1)|
    verdict: str

    def summary(self) -> str:
        q = self.timeout_R_quantiles
        lines = [
            "=" * 66,
            "TIMEOUT STRUCTURE + SIZING GAP",
            "=" * 66,
            f"trades                 : {self.n_trades}",
            f"exit mix               : tp {self.tp_rate:.1%} | sl {self.sl_rate:.1%} | timeout {self.timeout_rate:.1%}",
            f"win rate P(R>0)        : {self.win_rate:.3f}",
            f"mean timeout R         : {self.mean_timeout_R:+.3f}   (full stop = -1.000)",
            f"timeout R q10/50/90    : {q.get('q10', float('nan')):+.2f} / {q.get('q50', float('nan')):+.2f} / {q.get('q90', float('nan')):+.2f}",
            "-" * 66,
            f"b design (u/d)         : {self.b_design:.3f}",
            f"b empirical            : {self.b_empirical:.3f}",
            f"p used                 : {self.p_used:.3f}",
            "-" * 66,
            f"full Kelly (binary)    : {self.f_binary_full:+.4f}",
            f"full Kelly (empirical) : {self.f_empirical_full:+.4f}   <- growth-optimal on real payoff",
            f"full Kelly (gaussian)  : {self.f_gaussian_full:+.4f}   (fat-tail caveat, not primary)",
            f"sizing error binary-emp: {self.f_binary_full - self.f_empirical_full:+.4f}   ({self.sizing_direction})",
            f"growth gap (absolute)  : {self.growth_gap_abs:+.5f} log-growth/trade   <- judge economic size here",
            f"growth capture (ratio) : {self.growth_capture:.1%}   (ratio can look scary while the abs gap is tiny)",
            "-" * 66,
            f"structural trigger     : {self.structural_trigger:.3f}   (rate * |mean_timeout_R - (-1)|)",
            f"VERDICT                : {self.verdict}",
            "=" * 66,
        ]
        return "\n".join(lines)


def analyze_timeout_structure(
    ledger: pd.DataFrame,
    b_design: float,
    p_used: Optional[float] = None,
    c: float = 0.0,
    trigger_threshold: float = 0.15,
) -> TimeoutReport:
    """The crux diagnostic. Compares binary Kelly (what the formula assumes) to
    empirical Kelly (what the realized payoff distribution actually implies).

    trigger_threshold: above this, the timeout misspecification is treated as
    first-order (consensus landed near 0.15-0.20). Note the trigger multiplies
    frequency by how far mean timeout R sits from a full stop, which is the
    correct axis, not frequency alone.
    """
    df = ledger.copy()
    df["exit_reason"] = df["exit_reason"].str.lower()
    R = df["R"].to_numpy(dtype=float)
    n = len(df)

    reason = df["exit_reason"].to_numpy()
    timeout_mask = reason == "timeout"
    tp_rate = float(np.mean(reason == "tp"))
    sl_rate = float(np.mean(reason == "sl"))
    timeout_rate = float(np.mean(timeout_mask))

    timeout_R = R[timeout_mask]
    mean_timeout_R = float(np.mean(timeout_R)) if timeout_R.size else 0.0
    if timeout_R.size:
        qs = {
            "q10": float(np.quantile(timeout_R, 0.10)),
            "q50": float(np.quantile(timeout_R, 0.50)),
            "q90": float(np.quantile(timeout_R, 0.90)),
        }
    else:
        qs = {"q10": float("nan"), "q50": float("nan"), "q90": float("nan")}

    win_rate = float(np.mean(R > 0))
    wins = R[R > 0]
    losses = R[R < 0]
    b_emp = float(np.mean(wins) / -np.mean(losses)) if wins.size and losses.size else float("nan")

    # DeepSeek's catch: defaulting p to P(R>0) counts small-positive timeout
    # exits as "wins" and inflates p -> inflates f_binary. Prefer the
    # barrier-conditioned win rate tp/(tp+sl), which matches the binary payoff.
    n_barrier = int(np.sum((reason == "tp") | (reason == "sl")))
    barrier_win = float(np.sum(reason == "tp") / n_barrier) if n_barrier else win_rate
    p_eff = float(p_used) if p_used is not None else barrier_win
    f_binary = binary_kelly(p_eff, b_design, c)
    f_binary = max(f_binary, 0.0)
    f_emp, g_emp = empirical_kelly(R)
    # DeepSeek: silently capping f_max hides a high RAW Kelly, which is itself an
    # overfitting / instability signal. Compute the uncapped optimum and surface
    # it when the cap binds.
    f_emp_raw, _ = empirical_kelly(R, f_max=50.0)
    f_gauss = max(gaussian_kelly(R), 0.0)

    g_binary = log_growth_at(f_binary, R)
    capture = (g_binary / g_emp) if g_emp > 0 else float("nan")
    growth_gap_abs = (g_emp - g_binary) if np.isfinite(g_binary) else float("nan")
    if f_binary > f_emp + 1e-9:
        sizing_direction = "binary OVER-sizes vs empirical (DANGER: over-bet)"
    elif f_binary < f_emp - 1e-9:
        sizing_direction = "binary UNDER-sizes vs empirical (conservative, safe)"
    else:
        sizing_direction = "binary ~ empirical"

    trigger = timeout_rate * abs(mean_timeout_R - (-1.0))

    if trigger >= trigger_threshold:
        verdict = ("FIRST-ORDER: replace binary Kelly with empirical/generalized "
                   "Kelly. Timeouts are frequent and/or far from a full stop.")
    elif not np.isnan(capture) and capture < 0.85:
        verdict = ("BORDERLINE: trigger is low but binary Kelly still leaves "
                   f">15% of growth on the table. Prefer empirical Kelly.")
    else:
        verdict = ("SECOND-ORDER: binary Kelly is an acceptable approximation "
                   "here. Spend effort on p-calibration and non-stationarity.")

    if f_emp_raw > 2.0 + 1e-6:
        verdict += (f" | WARNING: uncapped empirical Kelly is {f_emp_raw:.2f} (cap 2.0 binds) "
                    "-> extreme raw Kelly is an overfitting/instability signal, not a green light.")

    return TimeoutReport(
        n_trades=n, timeout_rate=timeout_rate, tp_rate=tp_rate, sl_rate=sl_rate,
        mean_timeout_R=mean_timeout_R, timeout_R_quantiles=qs, win_rate=win_rate,
        b_design=b_design, b_empirical=b_emp, f_binary_full=f_binary,
        f_empirical_full=f_emp, f_gaussian_full=f_gauss, p_used=p_eff,
        growth_empirical=g_emp, growth_binary=g_binary, growth_capture=capture,
        growth_gap_abs=growth_gap_abs, sizing_direction=sizing_direction,
        structural_trigger=trigger, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# 3. Zero-alpha gate (Gemini's test, strengthened by DeepSeek)
# ---------------------------------------------------------------------------

@dataclass
class ZeroAlphaReport:
    base_rate: float
    base_expectancy: float          # mean R over all trades (no selection)
    p_mean_real: float
    p_mean_noise: float
    p_std_real: float
    p_std_noise: float
    lift_real: float                # E[R | selected by real p] - base_expectancy
    lift_noise: float               # E[R | selected by noise p] - base_expectancy
    passed: bool
    notes: str

    def summary(self) -> str:
        return "\n".join([
            "=" * 66,
            "ZERO-ALPHA GATE  (alpha -> noise; selection lift and p must collapse)",
            "=" * 66,
            f"base rate (uncond. win)   : {self.base_rate:.3f}",
            f"base expectancy (all R)   : {self.base_expectancy:+.4f}",
            f"p mean  real / noise      : {self.p_mean_real:.3f} / {self.p_mean_noise:.3f}",
            f"p std   real / noise      : {self.p_std_real:.3f} / {self.p_std_noise:.3f}",
            f"selection lift real/noise : {self.lift_real:+.4f} / {self.lift_noise:+.4f}",
            f"RESULT                    : {'PASS' if self.passed else 'FAIL'}",
            f"notes                     : {self.notes}",
            "=" * 66,
        ])


def zero_alpha_test(
    make_estimator: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    R: np.ndarray,
    alpha_cols: Sequence[str],
    select_frac: float = 0.4,
    n_shuffles: int = 5,
    random_state: int = 0,
) -> ZeroAlphaReport:
    """Refit the meta-model with the alpha feature(s) replaced by white noise.

    The meta-model's job is to FILTER trades. Its value is the SELECTION LIFT:
    the mean R of the trades it picks, minus the mean R of all trades. We select
    a fixed top fraction by p (so sample sizes stay stable under real and noise,
    avoiding the tiny-sample artifact of a hard probability threshold).

    Pass requires BOTH:
      (a) the noise selection lift collapses toward zero (noise picks no better
          than random), and
      (b) the p distribution collapses toward the base rate (lower spread).
    A positive absolute expectancy under noise is NOT a failure by itself: the
    trade population can be positive-EV independently of alpha. Only the *lift*
    over base matters.
    """
def zero_alpha_test(
    make_estimator: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    R: np.ndarray,
    alpha_cols: Sequence[str],
    t0: np.ndarray,
    t1: np.ndarray,
    select_frac: float = 0.4,
    n_shuffles: int = 5,
    n_splits: int = 6,
    embargo_pct: float = 0.01,
    random_state: int = 0,
) -> ZeroAlphaReport:
    """Refit the meta-model with the alpha feature(s) replaced by white noise.

    The meta-model's job is to FILTER trades. Its value is the SELECTION LIFT:
    the mean R of the trades it picks, minus the mean R of all trades. We select
    a fixed top fraction by p (so sample sizes stay stable under real and noise,
    avoiding the tiny-sample artifact of a hard probability threshold).

    PANEL FIX: predictions are PURGED FORWARD-ONLY, not sklearn KFold. Plain
    KFold trained each test fold on the OTHER folds -- including FUTURE folds
    (Gemini) -- and never purged overlapping triple-barrier labels (ChatGPT,
    DeepSeek). Both leak information backward and inflate lift, giving a false
    PASS. Here each test block is predicted by a model trained only on earlier
    data, with overlapping-label purge + embargo via purged_walk_forward_splits.

    LEAKAGE CAVEAT (ChatGPT + Gemini): a purged split makes the CV *split*
    leak-free, but the PIPELINE is only leak-free if every fit-requiring step is
    inside the fold too -- feature scaling/normalization, imputation, probability
    calibration, feature selection, and hyperparameter tuning. This harness fits
    the estimator per fold, but if your features (X) were engineered with global
    statistics or your labels leak into X, the OOF p is still contaminated. Pass
    here means "the split did not leak," not "the whole pipeline is clean."
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    R = np.asarray(R, dtype=float)
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    base_rate = float(np.mean(y))
    base_expectancy = float(np.mean(R))

    splits = list(purged_walk_forward_splits(t0, t1, n_splits=n_splits,
                                             embargo_pct=embargo_pct))

    def _oof_p(Xmat):
        p = np.full(len(y), np.nan)
        for tr, te in splits:
            est = make_estimator()
            est.fit(Xmat[tr], y[tr])
            p[te] = est.predict_proba(Xmat[te])[:, 1]
        return p

    def _lift(p):
        mask = np.isfinite(p)
        if mask.sum() < 10:
            return 0.0
        pv, Rv = p[mask], R[mask]
        k = max(1, int(round(select_frac * len(pv))))
        top = np.argsort(pv)[-k:]
        return float(np.mean(Rv[top])) - float(np.mean(Rv))

    p_real = _oof_p(X.values)
    lift_real = _lift(p_real)

    p_noise_runs, lift_noise_runs = [], []
    for _ in range(n_shuffles):
        Xn = X.copy()
        for col in alpha_cols:
            Xn[col] = rng.permutation(Xn[col].to_numpy())
        p_n = _oof_p(Xn.values)
        p_noise_runs.append(p_n[np.isfinite(p_n)])
        lift_noise_runs.append(_lift(p_n))

    p_noise = np.concatenate(p_noise_runs) if p_noise_runs else np.array([base_rate])
    p_real_valid = p_real[np.isfinite(p_real)]
    lift_noise = float(np.mean(lift_noise_runs)) if lift_noise_runs else 0.0

    p_collapses = np.std(p_noise) < (np.std(p_real_valid) * 0.7 + 1e-9)
    # Selection lift is RANK-based (top select_frac by p), so it is blind to how
    # wide the p distribution is. Gemini + DeepSeek showed the loophole: a model
    # whose p collapses to a tiny band (e.g. 0.51-0.53) can still rank-order that
    # band and score a high lift_real, passing on lift alone. So p-spread collapse
    # is a REQUIRED gate again, not a soft warning.
    lift_collapses = lift_noise < max(0.25 * lift_real, 0.03)
    inconclusive = lift_real <= 0.03
    passed = bool(lift_collapses and p_collapses and not inconclusive)

    notes = []
    if inconclusive:
        notes.append("no real selection edge detected (lift_real ~ 0); test inconclusive")
    if not lift_collapses:
        notes.append("noise produced POSITIVE selection lift -> leakage or overfitting (FAIL)")
    if not p_collapses:
        notes.append("p spread did NOT collapse under noise -> model overfits non-alpha "
                     "features; rank-based lift can hide this, so this is a FAIL")
    if passed:
        notes.append("noise adds ~zero selection lift AND p spread collapsed; edge comes from alpha")

    return ZeroAlphaReport(
        base_rate=base_rate, base_expectancy=base_expectancy,
        p_mean_real=float(np.mean(p_real_valid)), p_mean_noise=float(np.mean(p_noise)),
        p_std_real=float(np.std(p_real_valid)), p_std_noise=float(np.std(p_noise)),
        lift_real=lift_real, lift_noise=lift_noise, passed=passed,
        notes="; ".join(notes),
    )


# ---------------------------------------------------------------------------
# 4. Purged, embargoed walk-forward (no look-ahead via overlapping labels)
# ---------------------------------------------------------------------------

def purged_walk_forward_splits(
    t0: np.ndarray,
    t1: np.ndarray,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    expanding: bool = True,
):
    """Walk-forward splits with purge + embargo (Lopez de Prado style).

    t0, t1 : per-sample event start and label-end bar indices (same length as data).
    A training sample is PURGED if its [t0, t1] interval overlaps the test block's
    [min t0, max t1] span. An additional EMBARGO of embargo_pct * (bar range) bars
    after the test span is also dropped from training.

    Yields (train_idx, test_idx) with test blocks in chronological order.
    """
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    n = len(t0)
    order = np.argsort(t0, kind="mergesort")
    folds = np.array_split(order, n_splits)
    bar_range = float(np.max(t1) - np.min(t0)) if n else 0.0
    embargo = int(round(embargo_pct * bar_range))

    for k in range(1, n_splits):  # start at 1 so there is history to train on
        test_idx = np.sort(folds[k])
        test_start = t0[test_idx].min()
        test_end = t1[test_idx].max()

        if expanding:
            candidate = np.concatenate(folds[:k])
        else:
            candidate = folds[k - 1]
        candidate = np.sort(candidate)

        # DeepSeek's catch: in an expanding window, train is always BEFORE
        # test_start, so an embargo added AFTER test_end does nothing. The real
        # leak is a training label that resolves too close to (or into) the test
        # period. So require each training label to fully resolve at least
        # `embargo` bars BEFORE the test begins. This both purges overlaps
        # (t1 >= test_start) and enforces a genuine gap.
        keep = t1[candidate] < (test_start - embargo)
        train_idx = candidate[keep]
        if train_idx.size == 0:
            continue
        yield train_idx, test_idx


# ---------------------------------------------------------------------------
# 5. Calibration report (Kelly needs the absolute p, not the ranking)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    brier: float
    ece: float
    bins: pd.DataFrame

    def summary(self) -> str:
        return "\n".join([
            "=" * 66,
            "CALIBRATION  (reliability of p; Kelly uses the absolute value)",
            "=" * 66,
            f"Brier score : {self.brier:.4f}   (lower is better)",
            f"ECE         : {self.ece:.4f}   (expected calibration error; <0.05 target)",
            "reliability bins:",
            self.bins.to_string(index=False),
            "=" * 66,
        ])


def calibration_report(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> CalibrationReport:
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    brier = float(np.mean((p_pred - y_true) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece = [], 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p_pred >= lo) & (p_pred < hi if i < n_bins - 1 else p_pred <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "conf": np.nan, "acc": np.nan})
            continue
        conf = float(np.mean(p_pred[m]))
        acc = float(np.mean(y_true[m]))
        ece += (cnt / n) * abs(acc - conf)
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": cnt, "conf": round(conf, 3), "acc": round(acc, 3)})
    return CalibrationReport(brier=brier, ece=float(ece), bins=pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# 6. Synthetic demo so the module runs out of the box
# ---------------------------------------------------------------------------

def _make_synthetic(n_trades: int = 3000, timeout_rate: float = 0.30, seed: int = 7):
    """Generate a toy ledger + feature matrix with a real but weak alpha and a
    non-trivial timeout population, so every diagnostic has something to chew on.
    """
    rng = np.random.default_rng(seed)
    b = 2.0  # design reward:risk (tp = 2R, sl = -1R)

    # a weak alpha that genuinely predicts direction, plus noise features
    alpha = rng.normal(0, 1, n_trades)
    noise1 = rng.normal(0, 1, n_trades)
    noise2 = rng.normal(0, 1, n_trades)
    true_win_prob = 1 / (1 + np.exp(-(0.6 * alpha)))  # alpha -> edge

    t0 = np.sort(rng.integers(0, 100_000, n_trades))
    hold = rng.integers(5, 25, n_trades)      # holding period in bars (overlap!)
    t1 = t0 + hold

    reasons = np.empty(n_trades, dtype=object)
    R = np.empty(n_trades, dtype=float)
    for i in range(n_trades):
        if rng.random() < timeout_rate:
            reasons[i] = "timeout"
            # timeout exits: continuous P&L, mean slightly negative, NOT a full stop
            R[i] = float(np.clip(rng.normal(-0.15, 0.45), -0.99, b))
        else:
            win = rng.random() < true_win_prob[i]
            if win:
                reasons[i] = "tp"
                R[i] = b
            else:
                reasons[i] = "sl"
                R[i] = -1.0

    y = (R > 0).astype(int)  # meta-label: was the trade a win
    ledger = pd.DataFrame({"t0": t0, "t1": t1, "exit_reason": reasons, "R": R})
    X = pd.DataFrame({"alpha": alpha, "noise1": noise1, "noise2": noise2})
    return ledger, X, y, b


def _demo():
    from sklearn.ensemble import GradientBoostingClassifier

    ledger, X, y, b = _make_synthetic()

    # a meta-model p for the calibration + zero-alpha demos
    # regularized so it does not overfit the two pure-noise features (stumps,
    # few trees, leaf floor, subsampling) -- this is what a disciplined meta-model
    # should look like, and it lets the stricter p-spread gate pass honestly.
    def _make_gbt():
        return GradientBoostingClassifier(
            max_depth=1, n_estimators=40, min_samples_leaf=80,
            subsample=0.7, random_state=0)

    gbt = _make_gbt()
    gbt.fit(X.values, y)
    p_hat = gbt.predict_proba(X.values)[:, 1]
    p_used = float(np.mean(p_hat[p_hat > 0.5])) if (p_hat > 0.5).any() else float(np.mean(p_hat))

    print(analyze_timeout_structure(ledger, b_design=b, p_used=p_used, c=0.001).summary())
    print()
    print(calibration_report(y, p_hat).summary())
    print()
    print(zero_alpha_test(
        make_estimator=_make_gbt,
        X=X, y=y, R=ledger["R"].to_numpy(),
        t0=ledger["t0"].to_numpy(), t1=ledger["t1"].to_numpy(),
        alpha_cols=["alpha"], select_frac=0.4,
    ).summary())
    print()

    # walk-forward sanity: show fold sizes and that purge removes overlaps
    print("=" * 66)
    print("PURGED WALK-FORWARD SPLITS (fold sizes; purge drops overlapping labels)")
    print("=" * 66)
    t0 = ledger["t0"].to_numpy()
    t1 = ledger["t1"].to_numpy()
    for k, (tr, te) in enumerate(purged_walk_forward_splits(t0, t1, n_splits=5, embargo_pct=0.01), 1):
        # naive (unpurged) train would be everything before the test block
        test_start = t0[te].min()
        naive = int(np.sum(t0 < test_start))
        print(f"fold {k}: train {tr.size:5d} (naive {naive:5d}, purged {naive - tr.size:4d})  test {te.size:5d}")
    print("=" * 66)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    _demo()
