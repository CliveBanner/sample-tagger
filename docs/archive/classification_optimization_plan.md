# Classification optimization plan (2026-07-02)

## Context

Current state of the pipeline (measured, not guessed):

- **Coverage:** of 226,443 files the final `instrument` comes from: path priors 82,713 (37%), model 61,500 (27%), nothing 73,050 (32%), human 9,280 (4%).
- **Model quality:** logreg on PANNs embeddings; CV accuracy **0.35** (on the 185 single-human labels), frozen-val accuracy 0.59 — but the val set is **29 samples**, so neither number is trustworthy. Average `model_conf` is 0.493; 60k files have margin < 0.1.
- **Labels:** 9,280 human labels, but only **185 are `single`** (individually auditioned); 8,508 are bulk (`cluster` 7,460 / `map` 593 / `propagate` 455) and 587 have no `label_source`. Labeled clusters routinely contain up to 4 distinct human labels → bulk labels are noisy.
- **Training-data plumbing is broken right now:** `config.json` lost its `"ml"` section (settings-save bug), so training silently uses `weak_weight=0.2` (was 0.007) and an empty `weak_label_map`. And `panns_instrument` uses the *old* taxonomy (`tonal` 49k, `drums` 32k, `fx` 27k, `snare`/`clap`/`hihat`/`cymbal` ~8k) — none of these are in `labels.db`, so wherever panns is the weak fallback it's dropped from training. `path_instrument` is already on the new taxonomy and unaffected.

Scope agreed with user: **existing PANNs embeddings only** (no re-embed), **~1000 new manual labels** available.

The strategy in one line: *fix the plumbing → build an eval set you can trust → improve the model with free methods → spend the remaining label budget where the model is most confused → convert confidence into coverage with calibrated thresholds.*

---

## Phase 0 — Repair the training-data plumbing (no labeling, ~1h work)

1. **Restore the `ml` section in `config.json`** and make `save_config` stop destroying it. Either add an `ml: dict` field to the `Config` dataclass (`sampletagger/config.py`) or make `save_config` (`sampletagger/web/state.py:98`) merge-preserve unknown top-level keys. Restore:
   ```json
   "ml": {"weak_weight": 0.007, "conf_threshold": 0.6}
   ```
2. **Add the old→new taxonomy map** so panns weak labels stop being dropped (`weak_label_map` is already consumed in `ml/train.py:23`):
   ```json
   "weak_label_map": {"snare": "snare_clap", "clap": "snare_clap",
     "hihat": "hats_cymbals", "cymbal": "hats_cymbals",
     "fx": "sfx", "drums": "perc", "808": "bass"}
   ```
   Leave `tonal` unmapped (deliberately dropped — it's too generic to be a label).
3. **Assign `label_source` to the 587 NULL-source human labels** (they predate provenance): `UPDATE samples SET label_source='bulk_legacy' WHERE human_instrument IS NOT NULL AND label_source IS NULL` — so `build_dataset` weights them as bulk (0.5), not accidental gold.

## Phase 1 — An eval set you can trust (~400 of the label budget)

You cannot optimize what you can't measure; 29 val samples measure nothing.

1. **Build a stratified gold campaign**: ~23 per class × 17 classes ≈ 400 files, sampled per *predicted* class (`model_instrument`) plus a slice from the `source='none'` pool so the eval covers the space the model currently punts on. Extend `scripts/freeze_val.py` (or a new `scripts/sample_gold.py`) to write the candidate list.
2. **Label them in the review UI as `single`s**, then set `is_val=1` on all of them. These are *never* trained on (`train.py:38` already excludes them).
3. **Make frozen-val macro-F1 the headline metric**: `ml/train.py` currently leads with CV-on-185; move the val report first and also append one JSON line per trained model (`version, macro_f1, per_class_f1, coverage@threshold`) to `models/metrics.jsonl` so improvements across this plan are tracked, not remembered.

## Phase 2 — Free improvements (no new labels; measure each on the Phase-1 val set)

Do these one at a time, keep what wins:

1. **Bulk-label denoising via cluster purity.** For `label_source IN ('cluster','bulk_legacy')`, compute each cluster's label agreement among its human-labeled members (`GROUP BY cluster_id`); weight those rows by purity (e.g. `bulk_weight * purity²`) instead of a flat 0.5. Clusters where 25 members carry 4 different labels are currently poisoning training at half the weight of gold. (`ml/train.py:build_dataset`, purity query in `ml/export.py` or inline.)
2. **One confident-learning pass over bulk labels.** Train, then flag training rows where the model contradicts its own bulk label with high confidence (`pred != y AND conf ≥ 0.8`). Down-weight them (×0.1) in the next fit, and surface the top ~100 as a "suspect labels" review-queue mode — correcting them is the cheapest labeling there is (each fixes a *wrong* training signal, not just adds one).
3. **Try a stronger head.** Sklearn `MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True)` next to the logreg; 2048-d × ~100k rows fits in seconds–minutes on this box. Also sweep logreg `C` (0.01–10). Keep whichever wins val macro-F1. (`ml/train.py`, model choice via `config.json` `ml.head`.)
4. **Self-training round.** Add model predictions with `conf ≥ 0.9` (currently 13,134 files) as pseudo-labels at `weak_weight`, retrain once, measure. Cheap to try, easy to revert; guard against feedback loops by only ever pseudo-labeling from a model that was trained *without* pseudo-labels.
5. **kNN label propagation (the most promising free method).** The fp16 sidecar + `SimIndex` matmul already do exactly the needed lookup. For every unlabeled file, find its k=15 nearest labeled neighbors (blocked matmul over the 9k labeled rows — 226k × 9k is trivial); if ≥ 70% agree on a class with mean sim ≥ threshold, emit a pseudo-label weighted by agreement×sim. Add as a new weak tier (weight ~0.05–0.1). This directly attacks the 73k `none` files sitting near labeled neighbors in embedding space.

## Phase 3 — Spend the remaining ~600 labels with active learning

The infrastructure mostly exists (`model_margin` is computed and stored; the review queue has modes).

1. **Class-stratified uncertainty queue** (this is A2 from `classification_efficiency_plan.md`): round-robin across predicted classes, within each class order by ascending `model_margin`. Prevents the 60k low-margin files from all being the same three confusable classes. (`sampletagger/web/labeling.py:review_queue`, one window-function query.)
2. **Work in rounds, not one marathon:** label ~120 → run `ml pipeline` (fit is 29s, predict 0.5s) → next batch comes from the *new* model's confusion. 5 rounds ≈ 600 labels. Between rounds, watch `models/metrics.jsonl`: **stop when val macro-F1 flattens** — remaining budget is better spent on the Phase-2.2 suspect-label queue.
3. All these go through single-file review → `label_source='single'` → full-weight gold, which also grows the CV pool from 185 toward ~800.

## Phase 4 — Convert model confidence into library coverage

1. **Per-class thresholds instead of the global 0.6.** After the final train, compute per-class precision on the val set across thresholds and pick, per class, the lowest threshold that holds precision ≥ 0.9 (config: `ml.target_precision`). Classes the model is good at (probably kick, vocal) get aggressive thresholds; confusables stay conservative. (`ml/predict.py:37` currently one global `conf_threshold`.)
2. **Re-resolve and report.** Target: `source='none'` well under 73k → ideally < 30k, model share from 27% toward 50%+, at measured ≥ 0.9 precision.
3. **Optional purity-gated bulk round:** re-run cluster bulk labeling but only offer clusters whose already-labeled members agree ≥ 90% — high-yield, low-noise, and now measurable against the val set.

---

## What this doesn't touch (and why)

- **Re-embedding with CLAP/MERT** — excluded by scope; it remains the biggest lever if Phase 2+3 plateau below what you want. The plan's val set and metrics log make that future comparison trivial.
- **`relabel-panns`** — still a no-op (DB blobs NULLed; sidecar is L2-normalized so `fc_audioset` outputs would be distorted). Unrelated to this pipeline; only matters if you want raw AudioSet labels refreshed.

## Order & effort

| Phase | Effort | Labels used | Expected effect |
|---|---|---|---|
| 0 plumbing | ~1h | 0 | recovers tuned weights + ~100k dropped panns weak labels |
| 1 eval set | ~1–2h labeling + small scripts | ~400 | trustworthy macro-F1; everything below becomes measurable |
| 2 free wins | ~1 day hacking, minutes per experiment | 0 | denoised training set, better head, pseudo-labels |
| 3 active learning | ~5 rounds × (20 min labeling + 30 s train) | ~600 | gold set 185 → ~1200, targeted at confusion |
| 4 thresholds | ~2h | 0 | coverage: 27% → 50%+ model-labeled at measured precision |

## Verification

- After Phase 0: `sample-tagger-ml export && train` — training-set size printed by `train.py` should jump (weak rows recovered); confirm `weak_weight=0.007` in the log.
- After Phase 1: `models/metrics.jsonl` gets its first honest baseline line; val report shows ~400 samples, every class ≥ 15 support.
- Each Phase 2/3 experiment: one `ml pipeline` run, compare `macro_f1` in `metrics.jsonl` against the baseline; keep only wins.
- After Phase 4: `SELECT source, COUNT(*) FROM samples GROUP BY source` — the before/after of this whole plan in one query; spot-check 20 newly model-labeled files in the map UI with audio playback.
