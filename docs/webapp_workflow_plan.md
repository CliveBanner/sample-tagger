# Webapp workflow plan — from toolchain patchwork to reusable product (2026-07-02)

## Context

The classification effort currently spans the webapp, five standalone scripts, hand-edited
config sections, a candidates .txt file, and manual SQL — a workflow that only works on this
repo checkout and only if you know the runbook. Goal: **the webapp alone carries a user from
"empty library" to "fully labeled library"**, with every step reusable on any sample
collection. No loose files, no repo-relative assumptions, no steps that exist only in docs.

Guiding rules:
1. Workflow state lives in the DBs, never in loose files (`gold_candidates.txt` etc.).
2. Anything the runbook says to do by hand becomes a button or a form.
3. Nothing assumes the repo checkout: no `state.HERE` data paths, no `scripts/` invocations.
4. The dashboard tells you where you are in the workflow and what's next.

## Target workflow (what a new user sees)

```
1 Setup      → point at library + db, scan (discover/label buttons exist today)
2 Taxonomy   → define classes + weak-signal mapping in UI
3 Bulk pass  → clusters/map bulk labeling (exists today)
4 Gold set   → guided eval campaign: sample → label → freeze (in-app)
5 Train      → pipeline button + metrics history (button exists, history doesn't)
6 Refine     → active-learning rounds with per-round progress
7 Apply      → final resolve, coverage report, optional in-file tagging (future)
```

## Phase A — Kill the loose-file/manual-SQL state

1. **Gold campaign into the DB.** New columns on `samples`: `gold_candidate INTEGER DEFAULT 0`
   (add to `SAMPLE_COLUMNS` in `sampletagger/db.py` — migration is automatic).
   - Port `scripts/sample_gold.py` → `sampletagger/ml/gold.py:sample_gold(db, per_class, extra_none)`
     writing `gold_candidate=1` instead of a txt file.
   - Port `scripts/freeze_val.py` → `gold.py:freeze(db)`: `SET is_val=1 WHERE gold_candidate=1
     AND label_source='single'` (propagate-contamination guard built in).
   - `review_queue` gold mode reads the column, not the file (`web/labeling.py:203-213`).
   - Delete `gold_candidates.txt` handling; delete both scripts after porting.
2. **Metrics into the DB.** Replace the planned `models/metrics.jsonl` with a `metrics` table
   in `samples.db` (`version, ts, val_n, macro_f1, per_class_f1 JSON, coverage JSON, notes`).
   `ml/train.py` inserts a row after every fit; API `GET /api/ml/metrics` returns history.
3. **Weak-label map + ml params into `labels.db`.** The `config.json "ml"` section is fragile
   (save_config ate it once already). Move to `labels.db`: table `ml_params(key, value)` and
   `weak_map(old_label, new_label)`. `ml/export.py:load_ml_cfg` reads from there;
   seed defaults on first run in `web/state.py:init`.

## Phase B — Taxonomy & weak signals editable in the UI

4. **Taxonomy page** (extend the existing ⚙ Labels modal into `/taxonomy`):
   - add/rename/merge classes (merge = `UPDATE samples SET human_instrument=? WHERE
     human_instrument=?` + relabel warning),
   - per-class color stored in `labels.db` (replaces hardcoded `INSTR_COLORS` in
     `web/state.py:21` — that dict is this repo's taxonomy baked into "reusable" code),
   - weak-map editor: two-column table old→new, backed by `weak_map`.
   - Port `scripts/seed_taxonomy.py` → "seed from path patterns" button (one-time helper).
5. **Path-rule editor (optional, later).** `paths.py:_INSTRUMENT_PATTERNS` is also baked-in
   taxonomy; expose as editable regex rules in `labels.db` with the current list as defaults.

## Phase C — Guided gold campaign in the UI

6. **Gold panel on the review page** (or a `/gold` card on the dashboard):
   - "Build gold set" form: per-class count (default 25), include-none slice → calls
     `POST /api/gold/sample` (wraps `gold.sample_gold`).
   - Progress bar: `labeled/total`, per-class support counts, live while labeling.
   - "Freeze eval set" button → `POST /api/gold/freeze`, disabled until 100% labeled;
     shows how many were skipped as non-single.
   - After freeze the panel becomes a val-set summary (n, per-class support).

## Phase D — Training & rounds as first-class UI

7. **Metrics history on the dashboard**: macro-F1 over model versions (from the `metrics`
   table), coverage per source (`human/model/path/none`) as the headline chart —
   replaces reading train logs.
8. **Round tracker**: an active-learning card — "Round N: 87/120 labeled since last train" —
   computed from `COUNT(label_source='single' AND ts > last_train_ts)`; "Train now" button =
   existing `/api/run/ml`. Nudge when a round is complete.
9. **Suspect-labels queue** (Phase-2 confident learning, when implemented) appears as one more
   dropdown mode — same pattern as `class_*`, reads a `label_suspect` column.

## Phase E — Portability (true reusability)

10. **One `--data-dir` (default: alongside the db)** replacing `state.HERE` for
    `labels.db`, `run.log`, `ml.log`, `models/`, projection sidecars — currently all pinned
    to the repo root (`web/state.py:9-19`). `server.py --db /anywhere/lib.db` must just work.
11. **Subprocess spawns already use `-m sampletagger...`** — keep it that way; the venv-python
    guess in `state.py:17` falls back to `sys.executable`, fine.
12. **Sonic clustering**: port `scripts/sonic_label.py` into `sampletagger/ml/sonic.py` with a
    CLI subcommand + "rebuild sonic clusters" button (the API already serves its tables;
    the producer is the last script-only dependency).

## Order & effort

| Phase | Effort | Unblocks |
|---|---|---|
| A | ~1 day | current gold campaign stops being file-based; metrics history exists |
| B | ~1 day | taxonomy changes (e.g. adding `drums`) fully in-app incl. weak map |
| C | ~½ day | the 400-label campaign is guided, progress visible |
| D | ~1 day | training rounds measurable from the dashboard |
| E | ~½ day | works on any library/db path — actual reusability |

A and C directly serve the running campaign — do them first; B/D/E can follow between
labeling rounds.

## Verification

- Fresh-start test: new empty dir, `sample-tagger-web --db /tmp/fresh/lib.db`, walk the UI
  through scan → taxonomy → bulk → gold → train without touching a terminal or the repo.
- Existing-data test: current `samples.db` — gold panel shows the in-flight campaign after
  the column migration (one-time import of `gold_candidates.txt` → `gold_candidate=1`).
- `grep -rn "state.HERE" sampletagger/web` returns only data-dir plumbing, no data paths.
