import os
import json
import sqlite3
import numpy as np
import time
from ..config import load_config
from ..constants import DIM
from ..migrate import emb_sidecar as _emb_sidecar

def load_ml_cfg(db_dir):
    """Read the [ml] section of config.json next to the DB. A missing file or bad
    JSON falls back to defaults ({}); any other error propagates rather than being
    silently swallowed."""
    try:
        with open(os.path.join(db_dir, "config.json")) as f:
            return json.load(f).get("ml", {})
    except (OSError, ValueError):
        return {}

def get_class_set(db_dir):
    labels_db = os.path.join(db_dir, "labels.db")
    if not os.path.exists(labels_db):
        return []
    con = sqlite3.connect(f"file:{labels_db}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute("SELECT name FROM labels ORDER BY name").fetchall()]
    finally:
        con.close()

def get_latest_features(db_dir):
    import glob
    files = glob.glob(os.path.join(db_dir, "models", "features_*.npz"))
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

        cache_path = os.path.join(db_dir, "models", f"features_{n}.npz")
        
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

        mat_file, paths_file = _emb_sidecar(args.db)
        t0 = time.time()
        if os.path.isfile(mat_file) and os.path.isfile(paths_file):
            # Fast path: load from sidecar (already L2-normalised float16)
            print(f"  loading from sidecar {mat_file}")
            with open(paths_file) as f:
                emb_paths = [line.rstrip("\n") for line in f if line.strip()]
            emb_mat = np.load(mat_file).astype(np.float32)
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
        else:
            # Slow path: read BLOBs from DB
            mat = np.empty((n, DIM), dtype=np.float32)
            paths, human, weak_path, weak_panns, label_source, is_val = [], [], [], [], [], []
            row_i = 0
            for p, v in con.execute("SELECT path, vec FROM embeddings WHERE vec IS NOT NULL"):
                a = np.frombuffer(v, dtype=np.float32)
                if a.shape[0] == DIM:
                    mat[row_i] = a
                    paths.append(p)
                    m = meta.get(p, {})
                    human.append(m.get("human") or "")
                    weak_path.append(m.get("path") or "")
                    weak_panns.append(m.get("panns") or "")
                    label_source.append(m.get("label_source") or "")
                    is_val.append(m.get("is_val", 0))
                    row_i += 1
            mat = mat[:row_i]
            # L2 normalize
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat /= norms
        
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
