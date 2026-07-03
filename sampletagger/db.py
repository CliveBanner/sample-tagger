import sqlite3
import time
SAMPLE_COLUMNS = {
    "path": "TEXT PRIMARY KEY", "mtime": "REAL", "size": "INTEGER", "duration_s": "REAL",
    "sample_type": "TEXT", "instrument": "TEXT", "tonal": "TEXT", "bpm": "INTEGER", "key": "TEXT",
    "source": "TEXT", "status": "TEXT", "error": "TEXT", "tagged": "INTEGER DEFAULT 0", "ts": "REAL",
    "path_instrument": "TEXT", "panns_instrument": "TEXT", "panns_conf": "REAL",
    "audio_instrument": "TEXT", "panns_label": "TEXT", "panns_label_conf": "REAL", "panns_topk": "TEXT",
    "human_sample_type": "TEXT", "human_instrument": "TEXT",
    "model_instrument": "TEXT", "model_conf": "REAL", "model_version": "TEXT", "model_margin": "REAL",
    "model_margin_label": "TEXT",
    "is_val": "INTEGER DEFAULT 0",
    "gold_candidate": "INTEGER DEFAULT 0",
    "rating": "INTEGER DEFAULT 0",
    "label_source": "TEXT",
    "cluster_id": "INTEGER", "cluster_d": "REAL",
}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS samples ({", ".join(f"{c} {t}" for c, t in SAMPLE_COLUMNS.items())});
CREATE INDEX IF NOT EXISTS idx_instr ON samples(instrument);
CREATE INDEX IF NOT EXISTS idx_type  ON samples(sample_type);
CREATE TABLE IF NOT EXISTS embeddings (
  path TEXT PRIMARY KEY, dim INTEGER, vec BLOB
);
CREATE TABLE IF NOT EXISTS metrics (
  version TEXT PRIMARY KEY, ts REAL, val_n INTEGER, macro_f1 REAL,
  per_class_f1 TEXT, coverage TEXT, notes TEXT
);
-- Multi-label truth: human label sets (rank 1 = dominant; drives the
-- human_instrument projection column) and the model's per-class output.
CREATE TABLE IF NOT EXISTS sample_labels (
  path TEXT, label TEXT, rank INTEGER DEFAULT 1, ts REAL,
  PRIMARY KEY (path, label)
);
CREATE TABLE IF NOT EXISTS model_labels (
  path TEXT, label TEXT, conf REAL,
  PRIMARY KEY (path, label)
);
CREATE INDEX IF NOT EXISTS idx_slbl ON sample_labels(label);
"""

def db_connect(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # apply any missing columns to existing databases
    existing = {r[1] for r in con.execute("PRAGMA table_info(samples)")}
    for col, ctype in SAMPLE_COLUMNS.items():
        if col not in existing:
            con.execute(f"ALTER TABLE samples ADD COLUMN {col} {ctype}")
    con.execute("PRAGMA journal_mode=WAL")
    return con

def db_known_set(con):
    """All indexed paths → (mtime, size, status)."""
    return {r[0]: (r[1], r[2], r[3]) for r in
            con.execute("SELECT path, mtime, size, status FROM samples")}

def upsert(con, table, row, key="path"):
    cols = list(row)
    sets = ",".join(f"{c}=excluded.{c}" for c in cols if c != key)
    con.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?'*len(cols))}) "
                f"ON CONFLICT({key}) DO UPDATE SET {sets}", list(row.values()))


def db_discover_upsert(con, path, mtime, size, path_instr):
    """Insert new file or refresh mtime/size. Never overwrites existing labels."""
    con.execute("""
        INSERT INTO samples (path, mtime, size, status, ts, path_instrument)
        VALUES (?, ?, ?, 'new', ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          mtime=excluded.mtime, size=excluded.size, ts=excluded.ts,
          path_instrument=COALESCE(samples.path_instrument, excluded.path_instrument),
          status=CASE WHEN samples.status='missing' THEN 'new' ELSE samples.status END
    """, (path, mtime, size, time.time(), path_instr))

def db_label_update(con, path, result, redo_set):
    """Write classifier results into their dedicated columns. Never touches
    instrument/source (those are reserved for human + trained classifier)."""
    ts = time.time()
    fields = [
        ("path", "path_instrument", None),
        ("panns", "panns_instrument", "panns_conf")
    ]
    for prefix, inst_col, conf_col in fields:
        if inst_col in result:
            overwrite = prefix in redo_set or (prefix == "path" and result.get("_missing"))
            if overwrite:
                if conf_col:
                    con.execute(f"UPDATE samples SET {inst_col}=?, {conf_col}=?, ts=? WHERE path=?",
                                (result[inst_col], result.get(conf_col), ts, path))
                else:
                    con.execute(f"UPDATE samples SET {inst_col}=?, ts=? WHERE path=?",
                                (result[inst_col], ts, path))
            else:
                if conf_col:
                    con.execute(f"UPDATE samples SET {inst_col}=COALESCE({inst_col},?), {conf_col}=COALESCE({conf_col},?), ts=? WHERE path=?",
                                (result[inst_col], result.get(conf_col), ts, path))
                else:
                    con.execute(f"UPDATE samples SET {inst_col}=COALESCE({inst_col},?), ts=? WHERE path=?",
                                (result[inst_col], ts, path))
    if result.get("emb"):
        con.execute("INSERT INTO embeddings(path,dim,vec) VALUES (?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
                    (path, len(result["emb"]) // 4, result["emb"]))
    # update status/error from decode attempt
    if "status" in result:
        con.execute("UPDATE samples SET status=?, error=? WHERE path=?",
                    (result["status"], result.get("error", ""), path))

