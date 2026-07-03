# Sample classification — labeling plan & training pipeline

## Context

We're building the final `instrument` label for the sample catalog by training a small
classifier on **human labels**, using the frozen 2048-d PANNs embedding (already stored for
all ~226k files) as the feature. This is transfer learning: a small head on a frozen
embedding needs hundreds of labels, not tens of thousands.

The target taxonomy is **parameterized** — it lives in `labels.db` (`labels` table, editable
live via the webapp's `/api/labels/add|delete`). The pipeline reads its class set from there
at train time; changing the taxonomy = edit the list + retrain, no code changes.

### What we know about separability (drives everything below)

Measured kNN-purity in embedding space (a point's neighbors sharing its label ≈ how often
the classifier will agree). Path-labeled classes are honest; melodic classes were
pseudo-labeled from raw PANNs so their numbers are **optimistic/circular** until confirmed by
real labels.

| Tier | Classes | Status |
|---|---|---|
| **Reliable** (independent evidence) | vocal, kick, tom, cymbal, clap | trust; train from weak labels |
| **Promising** (distinct region, unconfirmed) | organ, strings, rhodes, guitar, piano, bass | **label here** |
| **Confirmed weak** | hihat, snare (mutual bleed); synth, drums, perc, fx (catch-alls) | low confidence; threshold/route to human |

Key principle: **probability-of-correct tracks timbral distinctiveness, not musical
importance.** Identity classes ≫ role/character classes (lead/pad/pluck have no AudioSet
signal and subjective ground truth — out of scope).

---

## Part A — Labeling plan (~700 labels)

Spend the budget where it moves accuracy: the **melodic split**, which path priors are blind
to and which only human labels can confirm. Percussion/coarse classes come nearly free from
path + PANNs agreement and need only spot-checking.

### Budget allocation

| Group | Classes | Human labels | Source of the rest |
|---|---|---:|---|
| **Percussion/coarse** | kick, snare, hihat, clap, cymbal, tom, perc, drums | ~120 (spot-check/correct only) | path weak labels (folder names are trustworthy here) |
| **Vocal / FX** | vocal, fx | ~40 | path + PANNs |
| **Melodic — confirm clusters** | organ, strings, rhodes, guitar, piano | ~70 each = **~350** | active labeling (primary spend) |
| **Bass family** | bass (+ 808→bass) | ~60 | active labeling; disambiguate from synth/sub/kick |
| **Boundary / hard cases** | piano↔rhodes, snare↔clap, synth↔melodic, bass↔808 | ~80 | uncertainty + disagreement sampling |
| **Catch-alls** | synth (heterogeneous), beep (rare), tonal | ~50 | anchor boundaries only |

≈ 700. The melodic tier (organ/strings/rhodes/guitar/piano/bass) gets ~60% of the budget.

### Workflow — active-learning loop (reuses the existing webapp review UI + `/api/label`)

- **Round 0 — seed (no human):** build weak labels = path→taxonomy for percussion/coarse +
  high-confidence raw-PANNs for melodic. Train an initial head. It'll be strong on percussion,
  rough on the melodic split.
- **Round 1+ — confirm/correct:** model predicts all → writes `model_instrument`/`model_conf`.
  The review UI shows each candidate **pre-filled with the model's guess**; you confirm (1 key)
  or correct. Correcting a suggestion is ~5× faster than labeling blank.
- Retrain on accumulated human labels (+ weak labels for cheap classes), re-predict,
  re-surface. **2–3 rounds** is typically enough.

### Which files to surface each round

1. **Stratify by predicted class** so rare classes (rhodes, organ, beep) get enough examples.
2. **Uncertainty**: lowest top-1 probability, or small top1−top2 margin.
3. **Disagreement**: path ≠ PANNs ≠ model — these are the informative boundaries.
4. **Dedup by embedding similarity** before serving — the library has conflicted-copy
   duplicates (e.g. `... (conflicted 2).wav`); don't spend labels on near-identical files, and
   keep dupes out of the train/val split to avoid leakage.

### Label hygiene

- Each label writes `human_instrument` (ground truth); class set constrained to `labels.db`.
- Allow an **"unsure/skip"** so you never inject noisy labels.
- **Freeze a ~20% validation slice** of human labels (flag in DB), never trained on — this is
  what gives honest, non-circular accuracy. (DONE)

---

## Part B — Model training pipeline

New subpackage `sampletagger/ml/` with small, composable steps + CLI subcommands. Everything
is CPU and fast (LR trains in seconds; predicting 226k is a matmul).

### Components

1. **`export.py` — features**
   - Read embeddings (reuse `sampletagger.sim.SimIndex.load`'s streaming reader), labels, and
     side features from the DB.
   - Feature = **L2-normalized 2048-d embedding** (primary). Optional, add only if CV improves:
     `path_instrument` one-hot, `log(duration_s)`, `tonal` flag, `sample_type`.
   - Emit `human_instrument` (truth) + weak labels (path/PANNs→taxonomy) with a `source`/`weight`
     column. Cache to `.npy`/parquet keyed by embedding count so runs don't re-read the 2 GB DB.
   - **Class set read from `labels.db`** at runtime; drop labels not in the current set (warn).

2. **`train.py` — fit**
   - Start with **`LogisticRegression`** (multinomial, L2, `class_weight='balanced'`). Upgrade
     to a tiny torch MLP (2048→256→n, dropout) only if linear plateaus.
   - **Weak supervision**: combine human + weak labels with sample weights (human 1.0, weak ~0.2)
     — for the MLP, pretrain on weak then fine-tune on human.
   - **Calibrate** (`CalibratedClassifierCV` or temperature scaling) so `model_conf` is a real
     probability usable for thresholding.
   - **Validation**: stratified k-fold on **human labels only**; report per-class P/R/F1 +
     confusion matrix; keep the frozen 20% test set untouched.
   - Save `joblib` + a `model_version` string.

3. **`predict.py` — write-back**
   - Score all embeddings → argmax + max prob → write `model_instrument`, `model_conf`,
     `model_version` (new additive columns, same pattern as `panns_label`).
   - **Final `instrument` resolution policy** (the fusion the project always intended) (DONE):
     `human_instrument` > (`model_instrument` if `model_conf ≥ threshold`) > path weak > none,
     setting `source` to `human`/`model`/`path`/`none` accordingly.
   - Threshold: start global ~0.6; move to **per-class** thresholds (reliable classes lower,
     catch-alls higher) tuned on held-out for a target precision.

4. **`active.py` — next batch**
   - Return the next K files to label (uncertainty + disagreement + class-stratified + dedup).
     Surfaced through a webapp endpoint with the model guess pre-filled.

5. **`report.py` — eval**
   - Per-class P/R/F1, confusion matrix, calibration curve, and **coverage@precision** (how many
     files auto-label at a given precision). Track across rounds to know when to stop.

### Schema additions (migrations, additive)

```
model_instrument TEXT, model_conf REAL, model_version TEXT
is_val INTEGER DEFAULT 0          -- frozen validation slice of human labels
```

### Config (`config.json`, parameterized)

```
feature_set:      ["embedding"]            # optionally + ["path_onehot","duration","tonal"]
weak_label_map:   { "808":"bass", ... }    # path/PANNs class -> target taxonomy (identity default)
weak_weight:      0.2
conf_threshold:   0.6                       # or per-class dict
val_fraction:     0.2
model_path:       "models/head.joblib"
```

### Reuse

- `sampletagger.sim.SimIndex` — embedding matrix loader.
- `labels.db` — dynamic class set.
- webapp review page + `/api/label`, `/api/labels` — the labeling loop UI already exists.
- `sampletagger.panns._PANNS_MAP` / `map_scores` — reference for the weak-label PANNs→taxonomy map.

### Accuracy targets (from the separability analysis)

- Reliable (kick, vocal, tom, cymbal, clap, organ, strings, rhodes, guitar): **>85% F1**.
- Muddy (piano, bass, synth): **>70%**, lean on the confidence threshold.
- Catch-alls (drums, perc, fx, snare, hihat): accept lower; raise their thresholds or merge
  confusable pairs (e.g. snare↔clap) if precision matters more than granularity.

---

## Deferred decisions (parameterized, decide later)

- Final taxonomy (edit `labels.db`); whether to merge confusable pairs or keep them split.
- Whether to add side features beyond the embedding (let CV decide).
- Linear head vs MLP (start linear).
- Global vs per-class confidence thresholds (start global, refine).
