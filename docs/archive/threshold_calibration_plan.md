# Per-class threshold calibration (planned 2026-07-03)

## Context

Baseline (ovr multi-label, 470-sample frozen val): macro F1 0.56 — precision 0.73 but
recall 0.47, because one flat `conf_threshold=0.9` gates all 18 sigmoid heads. Heads are
independently calibrated differently: vocal is precise at far lower scores, synth isn't
precise even at 0.9. Goal: per-class thresholds picked on the frozen val set to hold a
target precision, converting surplus precision into recall/coverage. No new labels needed.

## Design

- `ml_params` gains `target_precision` (default 0.9). `conf_threshold` stays as the
  fallback for classes that can't be calibrated.
- Calibration happens in **train.py**, right after the val evaluation (it already has
  `Y_v` and val probabilities):
  - per class j, sweep t over a grid (0.30 → 0.98, step 0.02);
    precision_j(t) = TP/(TP+FP) among val rows where p_j ≥ t
  - pick the **lowest t** with precision ≥ target_precision
  - fallbacks: val support < 10 → global `conf_threshold`; target never reached →
    global `conf_threshold`
- Thresholds ship inside `head.joblib` (`"thresholds": {class: t}`) — they're model
  state, not config — and are echoed into the metrics row (`coverage` JSON) so
  `/api/ml/metrics` exposes them.
- **predict.py** uses the per-class vector: `model_labels` membership = `p_j ≥ thr[j]`;
  resolve `source='model'` when top-1 prob ≥ its own class threshold.
- The val report in train.py switches from the flat threshold to the calibrated vector,
  so the headline macro F1 measures the deployed operating point.

## Steps

1. **[edit]** `sampletagger/ml/export.py`: add `"target_precision": 0.9` to
   `ML_PARAM_DEFAULTS` (existing labels.db rows: one INSERT OR IGNORE via
   `ensure_ml_tables` — extend it to top up missing keys, it currently only seeds
   an empty table).
2. **[edit]** `sampletagger/ml/train.py`: after the val eval, add
   `calibrate_thresholds(Y_v, probs_v, classes, target, fallback, min_support=10)`
   → dict; re-run `_report` with the vector (report both flat and calibrated once,
   for the before/after in the log); include `"thresholds"` in the joblib dump and
   in the metrics `coverage` JSON.
3. **[edit]** `sampletagger/ml/predict.py`: load `thresholds` from the joblib
   (fallback: flat `conf_threshold` for old models); build `thr` vector aligned to
   `classes`; use it for `model_labels` membership and the resolve condition.
4. **[edit]** `sampletagger/ml/train.py` `_report`: accept a threshold vector.
5. **[cmd]** `./venv/bin/python -m sampletagger.ml.cli pipeline samples.db`
6. **[check]** New metrics row: macro F1 should beat 0.56 mainly via recall (> 0.47);
   precision should hold ≈ target. Thresholds visible in `/api/ml/metrics`.
7. **[check]** `sqlite3 samples.db "SELECT source, COUNT(*) FROM samples GROUP BY source"`
   — `model` should rise above 112.7k / `none` fall below 47.9k, at measured precision.
8. **[ui]** Map: spot-check ~10 newly model-resolved files in the weakest calibrated
   classes (synth, pad); if a class sounds wrong, raise `target_precision` and re-run
   (train+predict ≈ 2 min).

## Honest caveat

Thresholds are selected on the same val set the F1 is reported on, so the calibrated
numbers are mildly optimistic. Acceptable at 470 samples; the next gold top-up (or any
future labels) provides untouched data to confirm. Don't re-tune target_precision by
staring at val F1 repeatedly — that's how eval sets die.

## Not in scope

Per-class thresholds for the *review queue* ordering (margin already handles uncertainty)
and probability recalibration (Platt/isotonic) — only worth it if the threshold sweep
shows heads badly miscalibrated near the operating point.
