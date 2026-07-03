# Classification optimization — exact step-by-step runbook

Companion to `docs/classification_optimization_plan.md`. Every step is either **[edit]** (code change), **[cmd]** (terminal), **[ui]** (web UI work), or **[check]** (verify before moving on). Work top to bottom; commit after each phase.

## Phase 0 — Plumbing (~1h, no labeling)

1. **[edit]** Fix `save_config` so it stops deleting unknown keys — `sampletagger/web/state.py:98`: load the raw JSON dict, update only the keys present in the POST, write back (or add `ml: dict = field(default_factory=dict)` to `Config` in `sampletagger/config.py` and make `load_config`/`asdict` carry it through).
2. **[edit]** Restore the ml section in `config.json`:
   ```json
   "ml": {
     "weak_weight": 0.007,
     "conf_threshold": 0.6,
     "weak_label_map": {"snare": "snare_clap", "clap": "snare_clap",
       "hihat": "hats_cymbals", "cymbal": "hats_cymbals",
       "fx": "sfx", "drums": "perc", "808": "bass"}
   }
   ```
3. **[cmd]** Tag the legacy no-provenance labels as bulk:
   ```
   sqlite3 samples.db "UPDATE samples SET label_source='bulk_legacy' WHERE human_instrument IS NOT NULL AND human_instrument!='' AND label_source IS NULL"
   ```
4. **[edit]** `sampletagger/ml/train.py:build_dataset` — make sure only `label_source='single'` rows count as gold (weight 1.0 + CV pool); everything else including `bulk_legacy` gets `bulk_weight`. (Currently `src == "single"` — already correct; just confirm `bulk_legacy` falls into the else branch.)
5. **[cmd]** `./venv/bin/sample-tagger-ml export samples.db && ./venv/bin/sample-tagger-ml train samples.db`
6. **[check]** Train log: "Training set: N samples" must be much larger than before (weak panns rows recovered), and no more 0.2 default weight. Save the printed val numbers as the "before" reference.

## Phase 1 — Frozen eval set (~400 labels, one evening)

7. **[edit]** New `scripts/sample_gold.py`: select ~23 files per class stratified by `model_instrument` (plus ~40 random from `source='none'`), excluding anything already `human_instrument`-labeled; write the paths into a review-queue-consumable place — simplest: `UPDATE samples SET rating=9` marker or a dedicated `gold_candidate` column, and add a `mode=gold` branch in `review_queue` (`sampletagger/web/labeling.py`) that serves exactly these.
8. **[cmd]** Run it: `./venv/bin/python scripts/sample_gold.py`
8a. **[edit]** Expose the gold mode in the UI — `sampletagger/web/static/review.html:9`, add inside `<select id="modesel">`:
    ```html
    <option value="gold">Gold set (eval)</option>
    ```
8b. **[edit]** Make the candidates file path CWD-independent — `sampletagger/web/labeling.py:205`: `open(os.path.join(state.HERE, "gold_candidates.txt"))` instead of the bare relative path (only resolves today if the server happened to start in the repo root).
8c. **[cmd]** Restart the webapp so the new backend + page are live:
    ```
    pkill -f sampletagger.web.server; setsid ./venv/bin/python -m sampletagger.web.server > /tmp/webapp.log 2>&1 &
    ```
9. **[ui]** Review page → select "Gold set (eval)" in the mode dropdown → label all ~441, one by one, listening to each. These arrive as `label_source='single'`.
10. **[cmd]** Freeze them (adapt `scripts/freeze_val.py` to set `is_val=1` on exactly this batch, e.g. `WHERE gold_candidate=1 AND human_instrument IS NOT NULL`).
11. **[edit]** `sampletagger/ml/train.py`: after the val report, append one JSON line to `models/metrics.jsonl`: `{"version": model_version, "ts": ..., "val_n": len(y_v), "macro_f1": ..., "per_class_f1": {...}}` (use `classification_report(..., output_dict=True)`).
12. **[cmd]** `sample-tagger-ml pipeline samples.db`
13. **[check]** `models/metrics.jsonl` has a line with `val_n ≈ 400+29` and a believable macro_f1. **This is the baseline number for everything below.**

## Phase 2 — Free experiments (no labels; after EACH: retrain, compare metrics.jsonl, keep or revert)

14. **[edit]** *Purity weighting* — in `build_dataset`: precompute per-cluster purity (`SELECT cluster_id, ...` majority-share among human-labeled members), then for cluster/bulk_legacy rows use `bulk_weight * purity**2` instead of flat `bulk_weight`.
15. **[cmd]** Retrain → **[check]** macro_f1 vs step 13; keep if better.
16. **[edit]** *Confident learning* — after fitting, predict on the bulk training rows; flag `pred != y AND conf >= 0.8`; write flags to a column (`label_suspect=1`) and down-weight flagged rows ×0.1 on refit. Optionally add `mode=suspect` to `review_queue` to relabel the worst ~100 by hand later.
17. **[cmd]** Retrain → **[check]** keep if better.
18. **[edit]** *Head experiment* — behind `config.json ml.head: "logreg"|"mlp"`, add `MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True)` as alternative in `run_train`; also sweep logreg `C` in {0.01, 0.1, 1, 10} quickly.
19. **[cmd]** One retrain per variant → **[check]** keep the winner.
20. **[edit]** *kNN label propagation* — new `sampletagger/ml/propagate.py`: load fp16 sidecar via `sampletagger.embeddings.load`, take the ~10k labeled rows as anchors, blocked matmul (226k × 10k), for each unlabeled file take k=15 nearest anchors; if ≥70% agree and mean sim ≥ ~0.55 emit pseudo-label; store to a new column (`prop_instrument`, `prop_conf`) and add as a training tier at weight ~0.05–0.1 in `build_dataset`.
21. **[cmd]** Retrain → **[check]** keep if better. (Self-training from `model_conf ≥ 0.9` is the fallback if propagation disappoints — same pattern, one evening less code.)

## Phase 3 — Active learning (~600 labels in ~5 rounds)

22. **[edit]** `review_queue` (`sampletagger/web/labeling.py`): add `mode=active_stratified` — round-robin over `model_instrument`, within class `ORDER BY model_margin ASC`, excluding human-labeled and `is_val=1` rows. One window-function query:
    `ROW_NUMBER() OVER (PARTITION BY model_instrument ORDER BY model_margin ASC)` then `ORDER BY row_number, model_margin`.
23. **[ui]** Label ~120 files in that mode.
24. **[cmd]** `sample-tagger-ml pipeline samples.db`
25. **[check]** metrics.jsonl — macro_f1 should tick up each round.
26. Repeat 23–25 up to 5 rounds; **stop early** when two consecutive rounds are flat, and spend leftover budget on the `suspect` queue from step 16 instead.

## Phase 4 — Per-class thresholds & final resolve (no labels)

27. **[edit]** `sampletagger/ml/predict.py`: after loading the model, compute per-class precision on the frozen val set at thresholds 0.4–0.95; per class pick the lowest threshold with precision ≥ `ml.target_precision` (default 0.9, classes with too little val support fall back to global `conf_threshold`); use that per-class threshold in the resolve loop instead of the single global one. Persist chosen thresholds into the joblib/metrics line.
28. **[cmd]** `sample-tagger-ml pipeline samples.db`
29. **[check]** The one-query before/after: `sqlite3 samples.db "SELECT source, COUNT(*) FROM samples GROUP BY source"` — target: `model` well above the current 61.5k, `none` well below 73k.
30. **[ui]** Map page: spot-check ~20 freshly model-labeled files (filter by instrument, listen). If a class sounds wrong, raise its threshold and re-run predict (0.5s).
31. **[cmd]** Commit; update `docs/classification_optimization_plan.md` status lines.

## Verification summary
- Every experiment = one `sample-tagger-ml pipeline` run (~30s) + one look at `models/metrics.jsonl`.
- Phase gates: step 6 (training set grew), step 13 (baseline exists), step 25 (F1 climbing), step 29 (coverage moved).
