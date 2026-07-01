# sample-tagger refactoring plan (2026-07-02)

## Context

The project recently went from a monolithic `sample_tagger.py`/`webapp.py` to a proper `sampletagger` package (core + `web/` + `ml/`, static HTML/JS split out of Python strings). That refactor left behind: dead code from the old pipeline, four copies of the embedding-loading logic, a 1,352-line `web/api.py`, a CLI with a hand-rolled subcommand hack, and — most urgently — **the entire refactor is uncommitted** (git has exactly one commit, "Initial commit prior to refactoring"). Several `docs/` plans are also stale: they describe work that is already done in the code (review-queue guards, `is_val`, `label_source` provenance, config unification, SQL cluster aggregation), which actively misleads anyone reading them.

Scope: code-health refactoring only. No feature work, no test/lint scaffolding.

---

## P0 — Git hygiene (do first, everything else builds on it)

1. Extend `.gitignore`: `run.log`, `ml.log`, `*.db`, `*.db-shm`, `*.db-wal`, `samples.db.*`, `samples.emb.*`, `venv/`, `__pycache__/`, `*.egg-info/`, `models/`.
2. Remove stray root artifacts: `__pycache__/sample_tagger.cpython-*.pyc` (compiled remains of the deleted monolith).
3. Commit the current state as the baseline (`git add -A && git commit`) — this captures the deletions of `sample_tagger.py`, `webapp.py`, `project.py`, `similar.py`, `simlib.py` plus the new package. Safe: `web/api.py` already spawns `-m sampletagger.cli` / `-m sampletagger.projection`, so nothing references the deleted root scripts.
4. Commit after each phase below so steps are individually revertable.

## P1 — Delete dead code from the old pipeline (verified unreferenced)

- **`sampletagger/analyze.py`** — delete whole file. `analyze_file()` is only called by dead `workers.process_one`, and it's actually broken: it uses `BPM_MIN`, `BPM_MAX` (line 118) and `HARMONIC_RATIO_TONAL` (line 95) without importing them → would `NameError` if ever run. The `Analysis` dataclass goes with it (only consumers are the dead functions below).
- **`sampletagger/workers.py`** — remove `process_one`, `_init_worker`, and globals `_WRITE_TAGS`, `_PANNS_ONLY` (only the dead path used them). Keep `discover_one`, `label_one`, `_init_label_worker`.
- **`sampletagger/db.py`** — remove `db_upsert` (never called) and the `from .analyze import Analysis` import.
- **`sampletagger/tags.py`** — delete. `write_tags` is only reachable via dead `process_one`. In-file tagging is still a future goal, but when it returns it should read final labels from `samples.db` rows, not live `Analysis` objects — rebuilding it then is cleaner than keeping this version on life support. Also drop `mutagen` from `pyproject.toml` deps when this goes.
- **`sampletagger/panns.py`** — remove `classify_instrument_panns` (never called). Keep `load_head` but make `stages.stage_relabel_panns` actually use it: stages.py imports it (line 9) and then re-implements the same AudioTagging/`getattr(model,'module',...)` dance inline (lines 42–46).
- **`sampletagger/stages.py`** — drop the `run_relabel_panns = stage_relabel_panns` alias (line 100); rename the function to `run_relabel_panns` to match `run_discover`/`run_label` and fix the import in `cli.py`.
- **Small bug while there:** `sim.py:127` prints literal `\n` — `print(f"query: {matched}\\n(...)")` has escaped backslashes in the Python source. Fix to real newlines.

## P2 — One embedding loader instead of four

Near-identical "read float16 sidecar, else read fp32 BLOBs from `embeddings` table, L2-normalize" logic lives in:
- `sampletagger/sim.py` `SimIndex.load()` (lines 19–61)
- `sampletagger/projection.py` `load_embeddings()` (lines 10–39)
- `sampletagger/migrate.py` `cmd_export()` (DB→matrix half, lines 50–67)
- `sampletagger/ml/export.py` (its own DB read)

Create **`sampletagger/embeddings.py`**:
- `sidecar_paths(db)` — move `emb_sidecar()` out of `migrate.py` (both `sim.py` and `projection.py` currently import it from there, which is backwards: query code depending on a migration tool).
- `load(db, *, dtype=np.float16, mmap=True) -> (paths, mat)` — sidecar fast path (mmap), DB-BLOB fallback with the DIM/length guards and L2 normalization that each copy currently repeats.

Rewire `sim.py`, `projection.py`, `migrate.py`, `ml/export.py` to use it. Behavior notes to preserve: `sim.py` wants float16 mmap; `projection.py` wants float32 in memory and has the torn-pair guard (`mat.shape[0] != len(paths)`) — keep that guard in the shared loader.

## P3 — CLI restructure (`sampletagger/cli.py`)

Replace the `if sys.argv[1] == "sim"` hack with real argparse subparsers: `discover`, `label`, `relabel-panns`, `sim`. Per-stage flags move to their subparser (`--classifiers`/`--redo` → label; `--gpu`/`--batch` → relabel-panns; `--trust-db`/`--no-cache` → discover); shared flags (`--db`, `-j`, `--limit`, `--dry-run`) via a parent parser. This replaces `--stage X` with `sample-tagger X` — update the two spawn sites in `web/api.py` (`run_start`, line 174–175) accordingly.

## P4 — Config lifecycle cleanup

Current shape: `config.py` builds a module-level `cfg = load_config()` at import time, and `constants.py` copies its values into module constants (`ANALYZE_SECONDS = cfg.analyze_seconds`, …). Two sources of truth, values frozen at import, config path hardwired to repo ROOT. It works today only because every run is a fresh subprocess that re-imports.

- Keep `constants.py` for true constants only (`SR`, `DIM`, `AUDIO_EXTS`); delete the copied config values.
- Consumers (`audio.py`, `workers.py`, `stages.py`) import `cfg` from `sampletagger.config` and read attributes at call time (`cfg.analyze_seconds` instead of `ANALYZE_SECONDS`). Mechanical rename, no behavior change, removes the snapshot layer.
- `web/api.py` already routes through `load_core_config(CONFIG_FILE)` — leave it; both resolve to the same repo-root `config.json`.

## P5 — Split `web/api.py` (1,352 lines) into domain modules

Keep `server.py` as-is (it's clean). Split `api.py` into a flat set of modules under `sampletagger/web/`:

| Module | Contents (current api.py regions) |
|---|---|
| `runs.py` | scan + ML subprocess management: `run_start/stop/status`, `ml_run_*`, `_tagger_pid`, log tailing |
| `labeling.py` | `label_api`, `label_propagate`, `label_cluster`, `label_map`, `label_type_api`, `rate_api`, labels-taxonomy add/delete, `review_queue` |
| `clusters.py` | `build_clusters`, `clusters_list`, `cluster_detail`, sonic_* endpoints |
| `mapview.py` | `build_map`, `map_api`, `point_api`, reprojection, `similar_api`, `propagate_candidates` |
| `stats.py` | `stats`, `recent_errors`, dashboard aggregation |
| `state.py` | shared paths (`HERE`, `DB`, `LABELS_DB`, `PYTHON`), the `ro()`/`q()` DB helpers, and the caches (`_SIM`, `_MAP`, `_CLUSTERS`) **behind a `threading.Lock`** — the server is `ThreadingHTTPServer`, so two requests can currently race a `_MAP`/`_SIM` rebuild |
| `routes.py` | `GET_ROUTES`/`POST_ROUTES` + a param-extraction helper so routes stop being nested `(_qs(req).get("x") or ["d"])[0]` lambdas — e.g. `qparams(req, path="", k=24)` returning typed values from defaults |

Two concrete fixes to fold in while moving code:
- **`ml_run_start` (line 218–229):** replace the `/bin/sh -c '"…" export && … train && … predict'` string with a `pipeline` subcommand in `sampletagger/ml/cli.py` that runs export→train→predict in-process, spawned as `[PYTHON, "-m", "sampletagger.ml.cli", "pipeline", DB]`. Removes shell quoting fragility and gives the status endpoint a single honest PID.
- **`migrate_db()` (line 1103):** delete; call `sampletagger.db.db_connect()` instead — `db.py` already auto-migrates from `SAMPLE_COLUMNS`, so the web layer's parallel ALTER-TABLE logic is drift waiting to happen.

Mechanical rule for the split: move functions verbatim, fix imports, keep endpoint URLs and JSON shapes identical so the static JS needs no changes.

## P6 — Housekeeping

- Move `name_clusters.py` and `test_cluster_names.py` from repo root into `scripts/` (they're experiments, same category as `cluster_dump.py`).
- Refresh `docs/`: mark the already-implemented items (cleanup_plan phases 1/3/5, boilerplate parts 1/2/4, efficiency A1/B1, `is_val` + provenance) as DONE or delete those sections, so the plans reflect reality again.
- `sampletagger/panns.py` `get_panns()` stores the model in `workers._PANNS`/`workers._PANNS_LABELS` — move that cache into module-level globals in `panns.py` itself and delete the two globals from `workers.py` (removes a circular-ish cross-module dependency; `workers` currently imports `panns` while `panns` reaches back into `workers`).

---

## Suggested order & commit points

P0 (baseline commit) → P1 (dead code) → P2 (embeddings module) → P3 (CLI) → P4 (config) → P5 (api split, biggest — can be done one module at a time) → P6. Each phase is independent enough to commit and verify separately; P5 depends on P3 only for the two spawn commands.

## Verification (no test suite, so smoke-check each phase)

- After every phase: `python -m compileall sampletagger` and `./venv/bin/python -c "import sampletagger.cli, sampletagger.web.server, sampletagger.ml.cli"`.
- P1/P2/P3: `./venv/bin/python -m sampletagger.cli discover ~/pcloud/DAW/Samples --db samples.db --limit 50 --dry-run`, same for `label --limit 5`, and `sample-tagger sim <some filename>` — compare output to a pre-refactor run.
- P2: quick parity check — load embeddings via old `SimIndex.load()` (pre-change git stash) vs new `embeddings.load()`, assert same row count and a matching sample row.
- P5: restart the webapp (`setsid ./venv/bin/python -m sampletagger.web.server > /tmp/webapp.log 2>&1 &`), then click through dashboard, map (select a point, play audio), review queue, clusters, settings save, and start/stop a `--limit` scan from the UI. `curl` each `/api/*` GET route and diff JSON keys against the pre-split server.
- P4: change `analyze_seconds` in `config.json`, run one `label --limit 1` and confirm the value is honored, proving the snapshot layer is gone.
