import os
import time
import sqlite3
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .export import get_latest_features, load_ml_cfg


def run_cluster(args):
    """Over-cluster the embeddings so similar samples can be reviewed and
    labeled in bulk. Writes cluster_id + cluster_d (distance to centroid,
    smaller = closer to the cluster core) back to samples.db."""
    db_dir = os.path.dirname(os.path.abspath(args.db))
    fp = get_latest_features(db_dir)
    if not fp:
        print("No features exported. Run `sample-tagger-ml export` first.")
        return

    cfg = load_ml_cfg(db_dir)
    target = int(getattr(args, "size", None) or cfg.get("cluster_size", 40))

    print(f"Loading {fp}...")
    d = np.load(fp, allow_pickle=True)
    X = d["X"]
    paths = d["paths"]
    n = len(X)
    K = max(2, n // target)
    print(f"{n} samples -> K={K} clusters (~{target}/cluster)")

    pca_dim = min(50, X.shape[1])
    t0 = time.time()
    Xr = PCA(n_components=pca_dim, svd_solver="randomized", random_state=0).fit_transform(X)
    print(f"PCA -> {pca_dim}d in {time.time()-t0:.1f}s")

    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=3, random_state=0)
    lab = km.fit_predict(Xr)
    print(f"KMeans in {time.time()-t0:.1f}s")

    # distance from each point to its assigned centroid (in PCA space)
    dist = np.linalg.norm(Xr - km.cluster_centers_[lab], axis=1)

    con = sqlite3.connect(args.db, timeout=60)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        t0 = time.time()
        con.executemany(
            "UPDATE samples SET cluster_id=?, cluster_d=? WHERE path=?",
            [(int(lab[i]), float(dist[i]), str(paths[i])) for i in range(n)])
        con.execute("CREATE INDEX IF NOT EXISTS idx_cluster ON samples(cluster_id)")
        con.commit()
    finally:
        con.close()
    print(f"Wrote {n} cluster ids in {time.time()-t0:.1f}s ({K} clusters)")
