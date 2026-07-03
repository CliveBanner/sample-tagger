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
    if model == "concat":
        return load_concat(db, dtype=dtype)
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


def load_concat(db, *, dtype=np.float32):
    """[PANNs | CLAP] feature concat (the pilot's winning config). Only paths
    present in BOTH sidecars are returned; each part is L2-normalized already,
    so each contributes comparable scale to the concatenated vector."""
    p_paths, p_mat = load(db, dtype=np.float32, mmap=True, model="panns")
    c_paths, c_mat = load(db, dtype=np.float32, mmap=True, model="clap")
    c_idx = {p: i for i, p in enumerate(c_paths)}
    keep = [(i, c_idx[p]) for i, p in enumerate(p_paths) if p in c_idx]
    if not keep:
        return [], np.zeros((0, p_mat.shape[1] + (c_mat.shape[1] if c_mat.ndim > 1 else 0)), dtype)
    pi = np.fromiter((a for a, _ in keep), dtype=np.int64, count=len(keep))
    ci = np.fromiter((b for _, b in keep), dtype=np.int64, count=len(keep))
    mat = np.concatenate([np.asarray(p_mat)[pi], np.asarray(c_mat)[ci]], axis=1).astype(dtype, copy=False)
    return [p_paths[i] for i in pi], mat
