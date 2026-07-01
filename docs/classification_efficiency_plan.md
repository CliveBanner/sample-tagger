# Classification efficiency plan — review & cluster logic

The scarce resource is **human labeling attention**, so "efficiency" splits into
**label-efficiency** (most accuracy per label) and **compute**. This plan targets
both, and closes gaps where the implementation drifted from `classification_plan.md`
(the planned 4-factor `active.py` batch selector was never fully built).

Critical files: `sampletagger/web/api.py` (`review_queue`, `build_clusters`,
`clusters_list`), `sampletagger/ml/predict.py`, `sampletagger/sim.py`.

---

## Part A — Label-efficiency (the real bottleneck)

### A1. Review queue re-surfaces already-labeled samples (do first, one-line fix)

Every mode in `review_queue` (`api.py:769`+) filters `source != 'human'` but never
checks `human_instrument`. `label_api` (`api.py:293`) writes `human_instrument` +
`label_source` but **not** `source`; `source` only flips to `'human'` on the next
`predict` run (`predict.py:69,88`). So from the moment you label a sample until you
retrain + re-predict, it keeps reappearing in the queue.

The cluster and propagate paths already guard correctly with
`AND (human_instrument IS NULL OR human_instrument='')` (`api.py:449,351`) —
`review_queue` is just inconsistent.

**Fix:** add `AND (human_instrument IS NULL OR human_instrument='')` to the `where`
of all four modes (`disagree`, `active`/`uncertain`, `class_*`, default).

**Verify:** label a sample, re-fetch the same review mode, confirm it no longer
appears (before retrain/predict).

### A2. No class stratification in the uncertainty queue

`mode="active"` is a pure global `ORDER BY model_conf ASC` (`api.py:781`).
`classification_plan.md` ("Which files to surface" #1) calls for
stratify-by-predicted-class so rare classes (rhodes, organ, beep) get surfaced.
Global lowest-confidence is dominated by the big confusable classes
(snare↔clap, synth catch-all), so the rare melodic classes — which the plan says
should get ~60% of the budget — almost never appear.

**Fix:** round-robin / per-class `LIMIT` across `model_instrument` (window function
`ROW_NUMBER() OVER (PARTITION BY model_instrument ORDER BY model_conf ASC)` then take
the lowest-confidence N per class, interleaved), instead of one global sort.

**Verify:** the `active` batch contains examples from rare predicted classes, not
just the dominant confusable ones.

### A3. No similarity dedup before serving a batch

The library has conflicted-copy duplicates (e.g. `... (conflicted 2).wav`).
`model_conf ASC` will hand you a run of near-identical copies → wasted labels **and**
train/val leakage (plan #4). `SimIndex` already provides the embedding matrix.

**Fix:** while assembling a batch, drop a candidate if its cosine-sim to an
already-selected candidate exceeds a threshold (reuse `SimIndex` / `mat @ mat[i]`).

**Verify:** a batch never contains two samples above the sim threshold of each other.

### A4. Uncertainty uses only top-1 prob, not margin (optional)

`predict.py:50` keeps only `max(prob)`; the top1−top2 margin is a stronger
uncertainty signal but is discarded.

**Fix:** in `predict.py`, also store top-2 prob (or the margin) as an additive
column; order the uncertainty queue by margin. Additive migration, same pattern as
existing `model_*` columns.

**Verify:** queue ordering changes to prefer small-margin samples; column populated
after a predict run.

---

## Part B — Compute-efficiency

### B1. `build_clusters` aggregates ~226k rows in Python; cache dies on every label

`build_clusters` (`api.py:362-403`) pulls every clustered row and folds with
`Counter`/`defaultdict`, while `_CLUSTERS=None` is set on **every**
`label_api`/`label_propagate`/`label_cluster` write. During active labeling that's a
full 226k-row scan + Python aggregation on basically every `/api/clusters` call.

All of it is expressible in SQL with the existing `idx_cluster`:
- sizes / unlabeled counts: `GROUP BY cluster_id`
- medoid: `MIN(cluster_d)` per cluster (join back for the path)
- dominant unlabeled model label + agreement: grouped count of `model_instrument`
  among `human_instrument IS NULL` rows, pick the max.

**Fix:** push the aggregation into SQL; stop materializing 226k rows. Consider not
nuking the whole cache for a single-cluster change (a label only affects one
cluster's counts) — lazy/partial invalidation or accept slightly stale agreement.

**Verify:** `/api/clusters` returns the same shape/values as before; timing drops
substantially on a cold cache.

### B2. `ORDER BY RANDOM()` scans+sorts the whole matching set

The `disagree`/`class_*`/default modes use `ORDER BY RANDOM()` (`api.py`), scanning
and sorting the full matching set each call, and are non-deterministic (you can
re-see items across refetches). Less critical once A1 lands.

**Fix (if these modes get heavy):** precomputed random-bucket column or anchor-based
sampling (`WHERE rowid > :anchor ... LIMIT`).

---

## Sequencing

1. **Now (quick, low-risk):** A1 (re-surfacing bug) + B1 (SQL-push clusters).
2. **Next (biggest accuracy-per-label lever):** a single `active.py`-style batch
   builder combining A2 + A3 (+ A4) into one queue, replacing the separate modes —
   i.e. finish the 4-factor selector that `classification_plan.md` Part A always
   specified.

## Out of scope

Dashboard graph reframing (`graphs_refactor_plan.md`), taxonomy/threshold tuning,
and the linear-head→MLP upgrade (`classification_plan.md` deferred decisions).
