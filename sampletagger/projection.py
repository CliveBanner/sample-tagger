import argparse
import os
import sqlite3
import time
import numpy as np
from .constants import DIM
from .config import cfg
from .migrate import emb_sidecar as _emb_sidecar

def load_embeddings(db, sample=0):
    # Embeddings live in the float16 sidecar (samples.emb.npy); the DB `vec`
    # column is emptied to save space. Prefer the sidecar, fall back to DB blobs.
    mat_file, paths_file = _emb_sidecar(db)
    if os.path.isfile(mat_file) and os.path.isfile(paths_file):
        with open(paths_file) as f:
            paths = [line.rstrip("\n") for line in f if line.strip()]
        mat = np.load(mat_file).astype(np.float32)   # (n, DIM), L2-normalised
        if mat.shape[0] != len(paths):               # guard against a torn pair
            n = min(mat.shape[0], len(paths))
            mat, paths = mat[:n], paths[:n]
    else:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT path, vec FROM embeddings WHERE vec IS NOT NULL").fetchall()
        finally:
            con.close()
        paths, vecs = [], []
        for p, v in rows:
            if v is None or len(v) != DIM * 4:       # skip missing / malformed blobs
                continue
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
    norms = np.linalg.norm(mat, axis=1, keepdims=True); norms[norms == 0] = 1
    X = mat / norms
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_neighbors=cfg.proj_n_neighbors, min_dist=cfg.proj_min_dist,
                            metric="cosine", random_state=42, verbose=True)
        return reducer.fit_transform(X)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=0).fit_transform(X)

def normalize01(xy):
    lo = xy.min(0); hi = xy.max(0); span = np.where(hi - lo == 0, 1, hi - lo)
    return (xy - lo) / span

def write(proj_db, paths, xy):
    con = sqlite3.connect(proj_db, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS projection (path TEXT PRIMARY KEY, x REAL, y REAL)")
    con.execute("DELETE FROM projection")
    con.executemany("INSERT OR REPLACE INTO projection(path,x,y) VALUES (?,?,?)",
                    [(p, float(x), float(y)) for p, (x, y) in zip(paths, xy)])
    con.commit(); con.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.getcwd(), "samples.db"))
    ap.add_argument("--method", choices=["umap", "pca", "auto"], default="auto")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    method = args.method
    if method == "auto":
        try:
            import umap  # noqa: F401
            method = "umap"
        except ImportError:
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
