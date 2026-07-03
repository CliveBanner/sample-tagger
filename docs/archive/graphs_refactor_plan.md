# Dashboard graphs refactor (post raw-PANNs / human-label changes)

## Context

The dashboard charts (`/api/stats` → `index.js`) were built around the original data model:
three weak classifiers (`path_instrument`, `panns_instrument`, `audio_instrument`) each rendered
as a coverage + distribution card, plus sample-type/key/bpm charts. Recent changes broke that
framing:

- **PANNs now stores raw output** — `panns_label` (top-1 of 527 AudioSet classes), `panns_label_conf`,
  `panns_topk`. The mapped `panns_instrument` is now a legacy/secondary signal, yet the dashboard's
  "PANNs" card still keys off it (`inst_dist("panns_instrument")`, `cov_row[1]`).
- **Human labels** live in `human_instrument` (the labeling campaign target), but the "human"
  coverage KPI counts `source='human'` (`stats()` `cov_row[3]`) — a different, inconsistent number.
- **Model predictions** (`model_instrument`/`model_conf`) are incoming (already referenced in
  `review_queue`), with no graph for them.
- **Taxonomy is now parameterized** to 19 classes in `labels.db`, but `index.js` `COLORS` (and the
  duplicate `INSTR_COLORS` in `api.py` and `COLORS` in `review.js`) hardcode ~14 and miss/collide on
  the new classes — graphs fall back to random `autoColor`.

Goal: re-frame the graphs around the real signal hierarchy — **weak signals → final label
(human > model > weak)** — surface the campaign-critical human/model distributions, treat raw PANNs
honestly, and drive all colors from one taxonomy source. Backend changes are additive to
`/api/stats`; no route/shape removals (frontend reads new keys, tolerates missing ones).

Critical files: `sampletagger/web/api.py` (`stats`, `dist`/`inst_dist`, `coverage`, color source),
`sampletagger/web/static/index.js` (`tick`, `classifierCard`, `pie`, `bars`, `COLORS`).

## Changes

### 1. Re-frame the cards into "weak signals" vs "final label"
- Keep `path` and `audio` distribution cards (still valid).
- **PANNs card → two truths**: (a) the *mapped* `panns_instrument` stays as a weak-signal
  distribution, but relabel it clearly as "PANNs (mapped)"; (b) add a **"PANNs raw (AudioSet)"**
  view from `panns_label` — top-N + an "other" bucket (raw is 404 classes, "Music" ~67%, so a
  plain top-14 bar is misleading; show top ~12 with the long tail collapsed, and note coverage).
- Drop the stale `panns_skipped`/`panns_min_duration` coverage note — relabel-panns derives labels
  from stored embeddings, so the "too short → skipped" framing no longer applies.

### 2. Add the **final-label** section (campaign-critical)
- **Human labels by class**: `GROUP BY human_instrument` distribution + a count vs. a per-class
  target, so the 700-label progress is visible at a glance. (This subsumes the "labels by class"
  card from `docs/webui_usability_plan.md` — implement it here.)
- **Model predictions** (render only when `model_instrument` exists): class distribution + a
  **`model_conf` histogram** and a **coverage@threshold** readout (how many auto-label at the
  configured confidence cutoff). This is the dashboard view of the training pipeline's output.
- Fix the **human coverage** KPI to count `human_instrument IS NOT NULL` (not `source='human'`).

### 3. Backend (`api.py:stats`)
- Add to the response: `panns_raw_dist` (top-N + other from `panns_label`), `human_dist`
  (`GROUP BY human_instrument`), and — guarded by a `has_model` column check — `model_dist` and
  `model_conf_hist`. Reuse the existing `inst_dist` helper; add a small `top_n_with_other(col, n)`
  for the raw-PANNs tail.
- Correct `coverage["human"]` to use `human_instrument`.

### 4. One color source (kills 3 copies, fixes new classes)
- Serve the taxonomy→color map from the backend (extend `/api/labels` to return `{name,color}` or
  add `/api/colors`), derived from `api.py:INSTR_COLORS` with a deterministic fallback hue for any
  `labels.db` class lacking an explicit color. `index.js`, `review.js`, and `map.js` consume that
  instead of their own hardcoded `COLORS`. Removes the missing/colliding-color problem for
  organ/piano/rhodes/guitar/strings/beep and aligns every chart with the parameterized taxonomy.

## Verification

- Restart `sample-tagger-web`; load `/`.
- **Stale framing gone**: PANNs card shows both "mapped" and "raw (AudioSet)"; no "too short"
  skipped note; numbers match `sqlite3 samples.db` `GROUP BY` queries for `panns_instrument`,
  `panns_label`, `human_instrument`.
- **Human/model sections**: human-by-class card renders and matches
  `SELECT human_instrument,COUNT(*) ... GROUP BY 1`; model section is hidden when `model_instrument`
  is absent and appears once populated; human coverage KPI equals `COUNT(human_instrument)`.
- **Colors**: every class in `labels.db` (incl. the new melodic ones) gets a stable, distinct color
  across the dashboard, map legend, and review buttons; no random colors.
- `/api/stats` still returns all previously-present keys (old charts unaffected); `node --check`
  passes on `index.js`.

## Notes

- Overlaps `docs/webui_usability_plan.md` (labels-by-class card) and `docs/classification_plan.md`
  (model_instrument/model_conf, coverage@threshold) — implement the dashboard side here; keep the
  column/threshold definitions consistent with the classification plan.
- Repo is being refactored elsewhere — changes are confined to `api.py:stats` and `index.js` plus
  the shared color endpoint; coordinate timing.
