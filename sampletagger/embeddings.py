import os
import sqlite3
import numpy as np
from .constants import DIM

def sidecar_paths(db_path, model="panns"):
    """Return (mat_npy_path, paths_txt_path) for the external embedding files."""
    base = db_path[:-3] if db_path.endswith(".db") else db_path
    if model == "clap":
        return base + ".clap.npy", base + ".clap.paths"
    return base + ".emb.npy", base + ".emb.paths"

def load(db, *, dtype=np.float16, mmap=True, model="panns"):
    mat_file, paths_file = sidecar_paths(db, model=model)
    if os.path.isfile(mat_file) and os.path.isfile(paths_file):
        with open(paths_file) as f:
            paths = [line.rstrip("\n") for line in f if line.strip()]
        if mmap:
            mat = np.load(mat_file, mmap_mode="r").astype(dtype, copy=False)
        else:
            mat = np.load(mat_file).astype(dtype, copy=False)
        
        # torn-pair guard
        if mat.shape[0] != len(paths):
            n = min(mat.shape[0], len(paths))
            mat, paths = mat[:n], paths[:n]
            
        return paths, mat

    # Slow path: load BLOB embeddings directly from the DB
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT path, vec FROM embeddings WHERE vec IS NOT NULL").fetchall()
    finally:
        con.close()
        
    paths, vecs = [], []
    for p, v in rows:
        if v is None or len(v) != DIM * 4:
            continue
        a = np.frombuffer(v, dtype=np.float32)
        if a.shape[0] == DIM:
            paths.append(p)
            vecs.append(a)
            
    mat = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, DIM), np.float32)
    if mat.shape[0] > 0:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms
        
    return paths, mat.astype(dtype, copy=False)
