"""Split-conformal prediction intervals for the multi-step forecast, per lead time.

The half-width is the finite-sample-corrected (1 - alpha) quantile of the
absolute forecast errors that the walk-forward backtest already measured for
that lead-time bucket. Calibration uses earlier folds; the last fold is held out
to report honest empirical coverage (calibrating and scoring on the same errors
would trivially hit the nominal level). Forecast errors at neighbouring hours are
correlated, so coverage is measured rather than assumed.
"""
import numpy as np
import pandas as pd

DEFAULT_ALPHA = 0.10


def conformal_halfwidth(abs_errors, alpha: float = DEFAULT_ALPHA) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest absolute error."""
    errs = np.sort(np.abs(np.asarray(abs_errors, dtype=float)))
    errs = errs[~np.isnan(errs)]
    if len(errs) == 0:
        raise ValueError("need at least one error")
    k = min(len(errs), int(np.ceil((len(errs) + 1) * (1 - alpha))))
    return float(errs[k - 1])


def _in_bucket(errors: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return errors[(errors.index >= lo) & (errors.index <= hi)]


def build_intervals(errors: pd.DataFrame, targets: list, buckets: list, horizon: int,
                    alpha: float = DEFAULT_ALPHA) -> dict:
    """`errors`: absolute errors indexed by lead hours, with a `fold` column and one
    column per target (the recursive-forecast errors from the backtest).

    Returns half-widths per target and bucket (calibrated on all folds, for
    serving) plus held-out coverage (calibrated on folds before the last, scored
    on the last fold)."""
    folds = sorted(errors["fold"].unique())
    calib, held_out = errors[errors["fold"] != folds[-1]], errors[errors["fold"] == folds[-1]]
    out = {"alpha": alpha, "method": "split conformal on walk-forward absolute errors, per lead-time bucket",
           "calibration_folds_for_coverage": [int(f) for f in folds[:-1]], "held_out_fold": int(folds[-1]),
           "targets": {}}
    for target in targets:
        entry = {}
        for lo, hi in buckets:
            hi = min(hi, horizon)
            if lo > horizon:
                continue
            serve = _in_bucket(errors, lo, hi)[target]
            cal, test = _in_bucket(calib, lo, hi)[target], _in_bucket(held_out, lo, hi)[target]
            hw = conformal_halfwidth(serve, alpha)
            cov = float((test <= conformal_halfwidth(cal, alpha)).mean()) if len(cal) and len(test) else None
            entry[f"{lo}-{hi}"] = {"lo": lo, "hi": hi, "halfwidth_mw": hw, "held_out_coverage": cov,
                                   "n_calibration": int(serve.notna().sum())}
        out["targets"][target] = entry
    return out


def halfwidth_for_lead(intervals: dict, target: str, lead: int):
    """Half-width (MW) for a lead time in hours, or None if no bucket covers it."""
    for b in intervals.get("targets", {}).get(target, {}).values():
        if b["lo"] <= lead <= b["hi"]:
            return b["halfwidth_mw"]
    return None
