# Reduce boilerplate in the sampletagger package

## Context

The modularization refactor produced a clean package, but several mechanical patterns repeat
across modules — the worst being **column lists restated 4×** in the DB layer and **two
independent copies** of config loading and DB migration (one in the `sampletagger` core, one in
`web/api.py`). These duplications are boilerplate *and* a drift risk: add a column or a config
key in one place and the other silently disagrees. This plan removes the repetition with
single-source-of-truth definitions and a few small helpers. **No behavior change** — same DB,
same routes, same JSON shapes; purely internal.

Critical files: `sampletagger/config.py`, `sampletagger/db.py`, `sampletagger/web/api.py`
(and minor: `sampletagger/sim.py`). The web frontend, stages, workers, and analysis logic are
left alone.

## Part 1 — `config.py`: loop over dataclass fields

The seven `cfg.x = float(c.get("x", cfg.x))` lines just restate the dataclass fields with their
types. Replace with a `fields()` loop that coerces via each field's declared type:

```python
from dataclasses import dataclass, fields
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(path=None):
    cfg = Config()
    try:
        with open(path or os.path.join(ROOT, "config.json")) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return cfg
    for fld in fields(cfg):
        if fld.name in data:
            setattr(cfg, fld.name, fld.type(data[fld.name]))   # fld.type is float/int
    return cfg
```

Adding a tunable becomes one dataclass line. (~32 → ~16 lines.)

## Part 2 — `db.py`: one source of truth for the `samples` columns

Today the column set lives in `SCHEMA`, in `MIGRATIONS`, and ~4× inside `db_upsert` (insert
list, placeholders, `ON CONFLICT SET`, value tuple). Collapse to a single ordered dict and
generate the rest:

```python
SAMPLE_COLUMNS = {
    "path": "TEXT PRIMARY KEY", "mtime": "REAL", "size": "INTEGER", "duration_s": "REAL",
    "sample_type": "TEXT", "instrument": "TEXT", "tonal": "TEXT", "bpm": "INTEGER", "key": "TEXT",
    "source": "TEXT", "status": "TEXT", "error": "TEXT", "tagged": "INTEGER DEFAULT 0", "ts": "REAL",
    "path_instrument": "TEXT", "panns_instrument": "TEXT", "panns_conf": "REAL",
    "audio_instrument": "TEXT", "panns_label": "TEXT", "panns_label_conf": "REAL", "panns_topk": "TEXT",
}
```

- **Schema**: `CREATE TABLE ... (", ".join(f"{c} {t}" ...))` from the dict.
- **Auto-migrate**: in `db_connect`, diff `SAMPLE_COLUMNS` against `PRAGMA table_info(samples)`
  and `ALTER TABLE ADD COLUMN` whatever's missing. **Deletes the `MIGRATIONS` list and the
  fragile `sql.split()[-2]` parser.** Adding a column = one dict entry.
- **Generic upsert** kills the 4× repetition and is reused for the `embeddings` table:

```python
def upsert(con, table, row, key="path"):
    cols = list(row)
    sets = ",".join(f"{c}=excluded.{c}" for c in cols if c != key)
    con.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?'*len(cols))}) "
                f"ON CONFLICT({key}) DO UPDATE SET {sets}", list(row.values()))

def db_upsert(con, a, tagged):
    row = {c: getattr(a, c) for c in SAMPLE_COLUMNS if hasattr(a, c)}
    row["tagged"], row["ts"] = int(tagged), time.time()
    upsert(con, "samples", row)
    if a.emb is not None:
        upsert(con, "embeddings", {"path": a.path, "dim": len(a.emb)//4, "vec": a.emb})
```

- **`db_label_update`**: drive the repeated COALESCE/redo blocks from a small table
  `[("path_instrument", None), ("audio_instrument", None), ("panns_instrument", "panns_conf")]`
  with an `overwrite = name in redo_set` flag, instead of three near-identical `if` branches.

## Part 3 — `web/api.py`: response + DB helpers, dispatch tables

Three repeating patterns:

1. **JSON responses** — `req._send(200, json.dumps(X), "application/json")` appears on nearly
   every route. Add `_json(req, obj, code=200)` and call it everywhere.
2. **Read-only DB access** — `con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, ...)` +
   `try/finally con.close()` repeats ~12×. Add a context manager `ro(db=DB)` (`@contextmanager`,
   yields the connection, closes in `finally`) and use `with ro() as con:`. Keep the existing
   `q(con, sql, params)` helper (`api.py:410`) for fetches.
3. **Route dispatch** — replace the long `if/elif` chains in `handle_get`/`handle_post` with
   `GET_ROUTES = {"/api/stats": ...}` / `POST_ROUTES = {...}` dicts mapping route → handler that
   returns a plain object; the dispatcher serializes via `_json`. Routes needing query/body args
   wrap a tiny lambda that pulls them from `req`. Keeps every URL and JSON shape identical.

## Part 4 — De-duplicate config and migration (highest value)

`web/api.py` carries its own `DEFAULT_CONFIG` + `load_config`/`save_config` and its own
`migrate_db()` (`api.py:545`, `ALTER TABLE ... ADD COLUMN`). These shadow `sampletagger/config.py`
and `sampletagger/db.py`.

- **Migration**: delete `api.py:migrate_db`; have the web layer call `sampletagger.db.db_connect`
  (which now auto-migrates from `SAMPLE_COLUMNS`), so columns are defined in exactly one place.
- **Config**: the web layer needs more keys than the analysis `Config` dataclass (library_path,
  workers, label flags, proj_*). Make `config.py` the single owner: extend the `Config` dataclass
  with those fields (or expose a `DEFAULTS` dict there) and have `api.py` import it instead of
  redefining `DEFAULT_CONFIG`. `save_config` stays in the web layer (it's HTTP-specific) but
  validates against the shared field set. Removes the second config schema entirely.

## Out of scope (leave as-is)

- `sim.fetch_meta`'s SELECT lists a display subset of columns — fine to leave; optionally note it
  references `SAMPLE_COLUMNS` for the names. Not worth coupling.
- Frontend `static/*`, `stages.py`, `workers.py`, `analyze.py`, `audio.py`, `paths.py`,
  `panns.py` — no significant duplication; don't touch.

## Verification

- `venv/bin/python -m py_compile` the four changed files; `python -c "import sampletagger.db,
  sampletagger.config, sampletagger.web.api"`.
- **Schema/migaration parity**: open a throwaway copy of `samples.db` with the new `db_connect`
  and confirm `PRAGMA table_info(samples)` matches the current 21 columns (no spurious adds, no
  drops). Confirm a fresh empty DB builds identically.
- **Upsert parity**: round-trip an `Analysis` through `db_upsert` into a temp DB and read it back;
  values match the pre-refactor path.
- **Web parity** (against the running app): restart `sample-tagger-web`; `curl` `/api/stats`,
  `/api/point?i=0` (still includes `panns_label`/`panns_topk`), `/api/similar`, `/api/map`,
  `/api/labels`; POST `/api/label` and confirm it writes `human_instrument`. Load `/`, `/map`,
  `/review`, `/settings`.
- **Config parity**: `GET`/`POST /api/config` round-trips the same keys; `config.json` on disk
  unchanged in shape.

## Coordination note

The package is being actively refactored by someone else. These changes touch `db.py`,
`config.py`, `web/api.py` — coordinate timing (apply on top of their latest), or hand these
snippets to the same implementer to fold in.
