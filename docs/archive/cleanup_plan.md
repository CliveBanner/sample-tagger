# Cleanup & consolidation plan

Post-refactor cleanup of the `sampletagger` package and surrounding root cruft. Four
independent phases plus one optional follow-up. Do them in any order; the suggested
order is lowest-risk first. Each phase is self-contained and has a verify step.

## Key facts that shape this plan

- **Some root stubs are load-bearing.** The web layer spawns and detects them:
  - `sample_tagger.py` — spawned at `api.py:155`; process-detection greps `/proc`
    for the literal string `"sample_tagger.py"` (`api.py:109`, `api.py:638`).
  - `project.py` — spawned at `api.py:592`.

  The rest are referenced nowhere and are safe to delete:
  `similar.py`, `restore.py`, `rewrite_css.py`, `rewrite_index.py`, `fix_css.py`,
  `split_script_1.py`..`split_script_5.py`, `test.py`.

- **`Config` is used, but the core ignores it.** The `Config` dataclass backs the
  `/api/config` settings page (`settings.js`). But the analysis modules import
  hardcoded tunables from `constants.py` and ignore `config.json` entirely — so
  changing e.g. `panns_min_duration` in the web UI silently does nothing for
  labeling. This is the real drift bug, and it dictates the direction of Phase 4:
  make `config.json` authoritative, do **not** delete it.

- **Git tracks no large binaries.** The ~3 GB is untracked local artifacts, so
  cleanup is filesystem-only — no history rewrite needed.

---

## Phase 1 — README (fixes broken build) (DONE)

`pyproject.toml:11` sets `readme = "README.md"` but no `README.md` exists, so
`pip install .` / `python -m build` fails.

1. Create `README.md` at repo root: title, one-paragraph description, install
   (`pip install -e .`), and pipeline usage
   (`sample-tagger <path> --stage discover|label`, `--stage relabel-panns`,
   `sample-tagger-web`).
2. **Verify:** `pip install -e .` (or `python -m build`) succeeds.

## Phase 2 — Delete pure cruft

1. Delete only the unreferenced files:
   `similar.py`, `restore.py`, `rewrite_css.py`, `rewrite_index.py`, `fix_css.py`,
   `split_script_1.py`..`split_script_5.py`, `test.py`.
2. **Do NOT delete** `sample_tagger.py` or `project.py` (see Phase 5).
3. Optionally clear stale artifacts: `*.log`, `test.db`. Leave `samples.db`,
   `samples.emb.npy`, `models/` — those are data.
4. Add to `.gitignore`: `*.npy` and `models/` (or `*.npz`) so the 885 MB / 333 MB
   artifacts can never be committed.
5. **Verify:** `sample-tagger-web` still starts; run/map buttons still work.

## Phase 3 — Fix `stages.py` artifacts (DONE)

1. Dedent the bodies of `run_discover` (`stages.py:104`) and `run_label`
   (`stages.py:174`) from 12 spaces to 4.
2. Remove the redundant `import json` / `import torch` inside `stage_relabel_panns`
   (`stages.py:37-38`) — already imported at module top (`stages.py:3,11`).
3. **Verify:** `python -c "import sampletagger.stages"`; a
   `--stage discover --dry-run -n` run behaves identically.

## Phase 4 — Config single-source-of-truth (the real fix)

Goal: `config.json` (and the settings UI) actually drives analysis; eliminate the
`config.py`↔`constants.py` duplication.

1. In `constants.py`, keep genuinely-fixed constants literal (`SR`, `DIM`,
   `AUDIO_EXTS`). For tunables that also live in `Config`, derive them from the
   loaded config:

   ```python
   from .config import cfg
   ANALYZE_SECONDS = cfg.analyze_seconds
   LOOP_MIN_SEC = cfg.loop_min_sec
   LOOP_BAR_TOLERANCE = cfg.loop_bar_tolerance
   HARMONIC_RATIO_TONAL = cfg.harmonic_ratio_tonal
   BPM_MIN, BPM_MAX = cfg.bpm_min, cfg.bpm_max
   PANNS_MIN_DURATION = cfg.panns_min_duration
   ```

   Non-breaking: every `from .constants import ...` site (`audio.py:2`,
   `workers.py:4`, `analyze.py:4`) keeps working but now reflects `config.json`.
   Workers re-import at process start, so the Pool picks it up.
2. Keep the dependency one-way: `constants → config`. `config.py` must not import
   `constants.py`.
3. Apply the `fields()`-loop simplification in `load_config` per
   `boilerplate_reduction_plan.md` Part 1 (already half-done at
   `config.py:37-41`; confirm `fld.type(val)` coercion).
4. **Verify:** change `panns_min_duration` in the settings UI → `config.json`
   updates AND a fresh `--stage label` run honors it. Confirm `/api/config` and
   `/api/stats` (`api.py:686`) return the same shape.

## Phase 5 (optional, defer) — decouple web from root stubs (DONE)

Only if you want to eventually delete `sample_tagger.py`/`project.py`.

1. Replace the subprocess targets at `api.py:155` and `api.py:592` with
   `[py, "-m", "sampletagger.cli", ...]` / `"-m", "sampletagger.projection"`.
2. Update the `/proc` cmdline greps (`api.py:109`, `api.py:638`, and the marker in
   `_tagger_pid`) to match the new `-m sampletagger.cli` string. Spawn and
   detection must agree on the marker — treat as one atomic change.
3. **Verify:** start a scan from the web UI; status/stop still detect the process.

---

## Out of scope

The `api.py` (1056-line) split — covered by `boilerplate_reduction_plan.md` and
`webui_usability_plan.md`.
