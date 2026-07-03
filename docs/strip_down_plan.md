# Strip-down plan — reduce to essentials (2026-07-03)

## Context

The codebase (~4,600 py lines + 5 web pages) carries three generations of approaches:
spectral heuristics (gen 1), cluster/sonic bulk-labeling tooling (gen 2), and the current
multi-label ML pipeline (gen 3). Gen 1 and most of gen 2 no longer earn their complexity:
the audio classifier is a weak signal the model outperforms, the cluster tooling did its
job (7.4k bulk labels are banked in `sample_labels` — data survives deletion of the
tooling), and several modules were one-time migrations. The essential product is:
**index → embed → label (review UI) → train/predict → browse (map + search)**.

## KEEP — the essential core

| Area | Modules | Why |
|---|---|---|
| Indexing | `cli.py` (discover, label), `stages.py`, `workers.py`, `paths.py` | new files must be discovered + embedded; path priors are the weak-label source |
| Embeddings | `embeddings.py`, `panns.py` (forward pass), sidecars | feature source for everything (CLAP will join, not replace, until proven) |
| Retrieval | `sim.py`, `projection.py`, map page | the product's core value |
| ML | `ml/export.py`, `train.py`, `predict.py`, `gold.py`, `pipeline.py`, `cli.py` | the classification pipeline + eval |
| Web | `server.py`, `state.py`, `routes.py`, `labeling.py`, `mapview.py`, `runs.py`, `gold.py`, slim `stats.py`; pages: dashboard, review, map, settings | the workflow UI |
| Data | `samples.db`, `labels.db` (taxonomy, ml_params, weak_map), `models/`, sidecars | |
| Docs | `taxonomy.md`, `clap_eval_plan.md`, this file | |

## DROP

### 1. Spectral audio classifier (gen 1)
- `workers.py`: the `_DO_AUDIO` branch of `label_one`, `_init_label_worker`'s audio flag
- `audio.py`: shrink to `load_audio` + `_true_duration` (delete `harmonic_ratio`,
  `count_onsets`, `detect_bpm`, `classify_loop`, `detect_key`,
  `classify_instrument_audio`, the Krumhansl tables)
- `cli.py` / `web/runs.py` / settings page / `config.json`: remove the `audio`
  classifier option and `label_audio` flag
- Keep the `audio_instrument` DB column (historic data, still displayed as a weak-signal
  row) — just nothing writes it anymore. Remove it from the review-queue disagreement
  CASE (`web/labeling.py`) and from `dist` cards in `stats.py`.

### 2. relabel-panns stage (broken no-op)
- `stages.py:stage_relabel_panns` + alias, `panns.py:load_head`, the `relabel-panns`
  subparser and its `--gpu/--batch` flags in `cli.py`. It reads DB blobs that were
  NULLed by the compact; the raw AudioSet labels it maintained are frozen in
  `panns_label/panns_topk` and can't be recomputed anyway.

### 3. Cluster & sonic tooling (gen 2) — EXCEPT the map's overview view
**Kept by request:** the map page stays fully intact as the library overview,
including its "sonic family" view mode. That means the read-only pieces survive:
`sonic_families`/`sonic_grains`/`sonic_members` + `sonic_for`/`sonic_family_labels`
(move from `web/clusters.py` into `web/mapview.py` or a small `web/sonic.py`),
their routes, the `cluster_l1`/sonic columns and tables, and all of `map.js`'s
view-by-family code. The sonic data is static in the DB, so it keeps working with
no producer.

Dropped:
- `ml/cluster.py` + the `cluster` subcommand; remove the cluster step from
  `ml/pipeline.py`
- `web/clusters.py`'s cluster review parts (`build_clusters`, `clusters_list`,
  `cluster_detail`), their routes, the `/clusters` page (`clusters.html`,
  `clusters.js`), nav links to it, `label_cluster` in `labeling.py`
- `scripts/sonic_label.py` (producer — regenerating sonic clusters needs a git
  checkout; acceptable since the data is static)
- Judgment call: cluster bulk-labeling produced 7.4k labels. If CLAP text search
  lands, its role is covered by search + propagate; if you miss it, it's in git.

### 4. One-time / superseded modules
- `migrate.py` + `sample-tagger-migrate` entry point (export/compact already done;
  `sidecar_paths` already lives in `embeddings.py`)
- `ml/report.py` + `report` subcommand (metrics table + train log cover it)
- `scripts/` entirely: `seed_taxonomy.py` (ran once), `cluster_apply.py`,
  `cluster_dump.py`, `llm_auto_classify.py`, `name_clusters.py`,
  `test_cluster_names.py`, `llm_classification_manual.md`
- `config.py`: drop fields for removed features (`label_audio`, `label_path`?,
  `redo`?) — check settings page consumers first; keep the analysis thresholds that
  `workers.py` still reads (`analyze_seconds`, `panns_min_duration`) and scan options.

### 5. Docs housekeeping
- `mkdir docs/archive`; move completed/superseded plans there: `cleanup_plan.md`,
  `boilerplate_reduction_plan.md`, `refactoring_plan.md`, `webapp_workflow_plan.md`
  (A+C done; B/D/E fold into future work), `classification_efficiency_plan.md`,
  `classification_plan.md`, `classification_optimization_plan.md` + runbook,
  `threshold_calibration_plan.md` (done), `graphs_refactor_plan.md`,
  `webui_usability_plan.md`, `map_batch_select_plan.md`
- Active set stays top-level: `taxonomy.md`, `clap_eval_plan.md`, `strip_down_plan.md`

## Phase 2 — GATE FAILED (2026-07-03): DO NOT EXECUTE

Pilot results: zero-shot 0.359 vs trained 0.518–0.586 — zero-shot cannot replace the
trained model, so labeling stays and everything below remains hypothetical. Kept for the
record; revisit only if a future embedding/prompt generation changes the zero-shot number
materially. The winning config was concat[PANNs|CLAP]+trained (0.586) → both models stay
(CLAP: search + features; PANNs: acoustic similarity + concat features).

## ~~Phase 2~~ (was: gated on the CLAP pilot)

If CLAP zero-shot ≥ the trained model on the gold set, the classification apparatus
itself becomes strippable:

- **Path priors**: `paths.py`, `path_instrument` writes in discover, the weak-training
  tier, the resolve fallback (`source='path'`), disagreement scoring. Resolve becomes
  human > CLAP-zero-shot.
- **The trained model**: `ml/train.py`, `predict.py` (replaced by a zero-shot predict
  writing the same `model_labels`/`model_instrument` columns from text anchors),
  `weak_map` (no weak labels → no mapping), most `ml_params`.
- **Webapp strip (labeling retired completely, per user):**
  - `/review` page (`review.html`, `review.js`), nav links to it
  - `web/labeling.py`: `review_queue`, `label_propagate`, the gold-mode plumbing;
    `web/gold.py` sample/freeze endpoints + the gold panel (campaign is closed —
    the val set stays frozen as-is)
  - the run/ML round tracker pieces of the dashboard that exist only for labeling
    rounds
  - **Smallest correction channel to consider keeping:** the map's existing label
    button + `label_api` (a dozen lines of already-written code). Zero-shot will
    make embarrassing mistakes; one click on the map to overrule a file is cheap
    insurance. If even that goes, the system is fully read-only on labels.
- **KEEP even then**: the frozen val set + `gold.status` (read-only) + metrics table —
  they measure zero-shot too; the eval never becomes optional. `sample_labels` stays
  as data (human > zero-shot in resolve). Caveat: with labeling retired, the val set
  can't grow — new taxonomy classes (new prompts) launch unmeasured until you
  temporarily resurrect labeling from git for a mini gold round.

If the trained model wins, path priors stay as its weak fuel and this phase is void.

## Order & verification

1. Commit current state first.
2. Strip in the order above (each numbered section = one commit).
3. After each: `python -m compileall sampletagger`, restart webapp, click through
   dashboard / review (label one file end-to-end incl. propagate) / map (select,
   play, label) / settings; `curl` the remaining GET routes for 200s.
4. After section 1: run `label --limit 3 --classifiers panns -j 1` to confirm the
   label stage still decodes/embeds.
5. Final: `sample-tagger-ml pipeline samples.db` → metrics row appears, macro F1
   unchanged (±noise) — proves the ML path untouched.
6. Update `pyproject.toml` entry points (remove `sample-tagger-migrate`),
   README, and the stats dashboard cards that referenced removed signals.

Expected reduction: roughly 1,800–2,000 python lines plus one whole web page —
about 40% of the codebase — with zero loss of banked data.
