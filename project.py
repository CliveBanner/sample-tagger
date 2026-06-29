#!/usr/bin/env python3
"""
project.py — reduce the 2048-d PANNs embeddings to 2D for the explore map.

Reads the embeddings table, computes a 2D layout (UMAP if installed, else PCA),
normalizes to [0,1], and writes a `projection(path, x, y)` table that the web
dashboard's map reads. Re-run any time to refresh as more embeddings land.

  ./venv/bin/python project.py                 # umap if available, else pca
  ./venv/bin/python project.py --method pca
  ./venv/bin/python project.py --sample 50000  # project a random subset (faster)
"""
import argparse
import os
import sqlite3
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "samples.db")
DIM = 2048


def load_embeddings(db, sample=0):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT path, vec FROM embeddings").fetchall()
    finally:
        con.close()
    paths, vecs = [], []
    for p, v in rows:
        a = np.frombuffer(v, dtype=np.float32)
        if a.shape[0] == DIM:
            paths.append(p); vecs.append(a)
    mat = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, DIM), np.float32)
    if sample and mat.shape[0] > sample:
        rng = np.random.default_rng(0)
        sel = rng.choice(mat.shape[0], sample, replace=False)
        mat = mat[sel]; paths = [paths[i] for i in sel]
    return paths, mat


def project(mat, method):
    # L2-normalize first so cosine geometry drives the layout
    norms = np.linalg.norm(mat, axis=1, keepdims=True); norms[norms == 0] = 1
    X = mat / norms
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.12, metric="cosine",
                            random_state=42, verbose=True)
        return reducer.fit_transform(X)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=0).fit_transform(X)


def normalize01(xy):
    lo = xy.min(0); hi = xy.max(0); span = np.where(hi - lo == 0, 1, hi - lo)
    return (xy - lo) / span


def write(proj_db, paths, xy):
    # Sidecar DB (separate file) so we never contend with the scan writing
    # samples.db. WAL keeps the dashboard's reads non-blocking during rewrites.
    con = sqlite3.connect(proj_db, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS projection (path TEXT PRIMARY KEY, x REAL, y REAL)")
    con.execute("DELETE FROM projection")
    con.executemany("INSERT OR REPLACE INTO projection(path,x,y) VALUES (?,?,?)",
                    [(p, float(x), float(y)) for p, (x, y) in zip(paths, xy)])
    con.commit(); con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--method", choices=["umap", "pca", "auto"], default="auto")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    method = args.method
    if method == "auto":
        try:
            import umap  # noqa: F401
            method = "umap"
        except Exception:
            method = "pca"

    t = time.time()
    paths, mat = load_embeddings(args.db, args.sample)
    print(f"loaded {len(paths)} embeddings; projecting with {method} ...", flush=True)
    if not len(paths):
        print("no embeddings yet — run the scan with --embed first."); return
    xy = normalize01(project(mat, method))
    proj_db = args.db + ".proj"
    write(proj_db, paths, xy)
    print(f"wrote projection for {len(paths)} points in {(time.time()-t)/60:.1f} min "
          f"(method={method}) -> {os.path.basename(proj_db)}")


if __name__ == "__main__":
    main()
