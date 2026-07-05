import os
import json
import sqlite3
import numpy as np
import time
from .. import embeddings
ML_PARAM_DEFAULTS = {"weak_weight": 0.007, "bulk_weight": 0.5, "conf_threshold": 0.6, "target_precision": 0.9, "feature_model": "panns"}
WEAK_MAP_DEFAULTS = {"snare": "snare_clap", "clap": "snare_clap",
                     "hihat": "hats_cymbals", "cymbal": "hats_cymbals",
                     "fx": "sfx", "drums": "perc", "808": "bass"}


def ensure_ml_tables(labels_db, db_dir=None):
    """Create/seed ml_params + weak_map in labels.db. On first run, values are
    imported from a legacy config.json "ml" section if one exists, else defaults."""
    legacy = {}
    if db_dir:
        try:
            with open(os.path.join(db_dir, "config.json")) as f:
                legacy = json.load(f).get("ml", {})
        except (OSError, ValueError):
            pass
    con = sqlite3.connect(labels_db, timeout=10)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS ml_params (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS weak_map (old_label TEXT PRIMARY KEY, new_label TEXT)")
        
        params = {**ML_PARAM_DEFAULTS,
                  **{k: v for k, v in legacy.items() if k != "weak_label_map"}}
        con.executemany("INSERT OR IGNORE INTO ml_params(key,value) VALUES(?,?)",
                        [(k, json.dumps(v)) for k, v in params.items()])
                        
        if con.execute("SELECT COUNT(*) FROM weak_map").fetchone()[0] == 0:
            wmap = legacy.get("weak_label_map") or WEAK_MAP_DEFAULTS
            con.executemany("INSERT INTO weak_map(old_label,new_label) VALUES(?,?)",
                            list(wmap.items()))
        con.commit()
    finally:
        con.close()


def load_ml_cfg(db_dir):
    """ML parameters + weak-label map from labels.db (next to samples.db).
    Values are JSON-encoded in ml_params; weak_map rows become cfg["weak_label_map"]."""
    labels_db = os.path.join(db_dir, "labels.db")
    ensure_ml_tables(labels_db, db_dir)
    con = sqlite3.connect(f"file:{labels_db}?mode=ro", uri=True, timeout=10)
    try:
        cfg = {}
        for k, v in con.execute("SELECT key, value FROM ml_params"):
            try:
                cfg[k] = json.loads(v)
            except ValueError:
                cfg[k] = v
        cfg["weak_label_map"] = dict(con.execute("SELECT old_label, new_label FROM weak_map"))
        return cfg
    finally:
        con.close()

def get_class_set(db_dir):
    labels_db = os.path.join(db_dir, "labels.db")
    if not os.path.exists(labels_db):
        return []
    con = sqlite3.connect(f"file:{labels_db}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute("SELECT name FROM labels ORDER BY name").fetchall()]
    finally:
        con.close()

def get_latest_features(db_dir, feature_model=None):
    """Latest feature cache FOR THE CONFIGURED MODEL — caches are keyed by
    (model, embedding count) so switching feature_model can't reuse stale
    features. Legacy unprefixed files count as panns."""
    import glob
    if feature_model is None:
        feature_model = load_ml_cfg(db_dir).get("feature_model", "panns")
    files = glob.glob(os.path.join(db_dir, "models", f"features_{feature_model}_*.npz"))
    if feature_model == "panns":
        files += [f for f in glob.glob(os.path.join(db_dir, "models", "features_*.npz"))
                  if f.split("features_")[-1].split(".")[0].isdigit()]
    if not files:
        return None
    return sorted(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))[-1]

def load_labels(db, paths):
    """Read label columns fresh from the DB, aligned to `paths`.

    The feature cache (.npz) is keyed by embedding *count*, which doesn't change
    when labels do — so the label arrays inside a cached npz go stale. Embeddings
    are the expensive part and never change, so train/predict cache those but read
    labels live via this function. Returns dict of object/int arrays."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        meta = {r[0]: r[1:] for r in con.execute(
            "SELECT path, human_instrument, path_instrument, panns_instrument, "
            "label_source, is_val FROM samples")}
    finally:
        con.close()
    human, weak_path, weak_panns, label_source, is_val = [], [], [], [], []
    for p in paths:
        h, wp, pn, ls, iv = meta.get(p, (None, None, None, None, 0))
        human.append(h or ""); weak_path.append(wp or ""); weak_panns.append(pn or "")
        label_source.append(ls or ""); is_val.append(iv or 0)
    return {
        "human": np.array(human, dtype=object),
        "weak_path": np.array(weak_path, dtype=object),
        "weak_panns": np.array(weak_panns, dtype=object),
        "label_source": np.array(label_source, dtype=object),
        "is_val": np.array(is_val, dtype=np.int8),
    }

def load_label_sets(db):
    """Human label SETS (multi-label truth) from sample_labels: path → [labels],
    ordered by rank (rank 1 = dominant)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        sets = {}
        for p, l in con.execute("SELECT path, label FROM sample_labels ORDER BY rank"):
            sets.setdefault(p, []).append(l)
        return sets
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()


def run_export(args):
    """
    Read embeddings and labels from the DB.
    Feature = L2-normalized 2048-d embedding.
    Cache to .npz keyed by embedding count.
    """
    db_dir = os.path.dirname(os.path.abspath(args.db))
    os.makedirs(os.path.join(db_dir, "models"), exist_ok=True)
    
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        if n == 0:
            print("No embeddings found.")
            return

        cfg = load_ml_cfg(db_dir)
        feature_model = cfg.get("feature_model", "panns")
        cache_path = os.path.join(db_dir, "models", f"features_{feature_model}_{n}.npz")
        
        if os.path.exists(cache_path) and not args.force:
            print(f"Loaded features from cache: {cache_path}")
            return
            
        print(f"Exporting {n} embeddings + labels...")

        # We need human_instrument, path_instrument, panns_instrument, label_source, is_val
        meta = {}
        for row in con.execute("SELECT path, human_instrument, path_instrument, panns_instrument, label_source, is_val FROM samples"):
            meta[row[0]] = {
                "human": row[1],
                "path": row[2],
                "panns": row[3],
                "label_source": row[4],
                "is_val": row[5] or 0
            }

        print(f"Exporting features from sidecar model: {feature_model}")
        
        t0 = time.time()
        emb_paths, emb_mat = embeddings.load(args.db, dtype=np.float32, mmap=False, model=feature_model)
        
        path_to_row = {p: i for i, p in enumerate(emb_paths)}
        paths, human, weak_path, weak_panns, label_source, is_val = [], [], [], [], [], []
        rows_mat = []
        for p in emb_paths:
            m = meta.get(p, {})
            paths.append(p)
            rows_mat.append(emb_mat[path_to_row[p]])
            human.append(m.get("human") or "")
            weak_path.append(m.get("path") or "")
            weak_panns.append(m.get("panns") or "")
            label_source.append(m.get("label_source") or "")
            is_val.append(m.get("is_val", 0))
        mat = np.array(rows_mat, dtype=np.float32)
        row_i = len(paths)
        
        # Convert strings to object arrays
        paths = np.array(paths, dtype=object)
        human = np.array(human, dtype=object)
        weak_path = np.array(weak_path, dtype=object)
        weak_panns = np.array(weak_panns, dtype=object)
        label_source = np.array(label_source, dtype=object)
        is_val = np.array(is_val, dtype=np.int8)
        
        np.savez_compressed(cache_path, 
                            X=mat, 
                            paths=paths, 
                            human=human, 
                            weak_path=weak_path, 
                            weak_panns=weak_panns,
                            label_source=label_source,
                            is_val=is_val)
        
        print(f"Exported {row_i} rows in {time.time()-t0:.1f}s to {cache_path}")
        
    finally:
        con.close()
