#!/usr/bin/env python3
"""
Sonic clustering + free-form descriptor labels — taxonomy-free, audio-only.

Groups samples by timbral similarity (PANNs embedding space) in two levels —
coarse sonic *families* -> fine *grains* — then describes each cluster purely by
interpretable DSP features measured from the waveform (brightness, tonality,
attack, length). No filenames, no instrument taxonomy: two clusters that *sound*
alike get similar descriptors regardless of what produced them.

Pipeline:
  1. load embeddings (models/features_*.npz), PCA-reduce
  2. coarse KMeans (--families); fine KMeans within each family (~--fine-size each)
  3. write cluster_l1 (coarse), cluster_id (fine), cluster_d to samples.db
  4. sample members per fine cluster, extract DSP features in parallel
  5. bin each axis into adjectives by dataset-relative terciles -> composed label
  6. store sonic_clusters(level,id,parent,size,label,<axes>) + print a summary

Usage:
  python3 scripts/sonic_label.py [--db samples.db] [--families 20]
      [--fine-size 200] [--per-cluster 15] [--workers N]
"""
import argparse
import glob
import os
import random
import sqlite3
import time
from multiprocessing import Pool

import numpy as np


# ---- DSP feature extraction (runs in worker processes) --------------------

def extract_features(path):
    """Interpretable acoustic descriptors from the waveform. None on failure or
    on near-silent input (which would otherwise read as flat/"noisy")."""
    try:
        import librosa
        y, sr = librosa.load(path, sr=22050, mono=True, duration=5.0)
        if y.size < 512 or float(np.max(np.abs(y))) < 1e-3:
            return None                                  # empty / silent → skip
        # trim leading/trailing silence so descriptors reflect the actual sound
        # (and attack is measured from the real onset); don't trim a transient away.
        yt, _ = librosa.effects.trim(y, top_db=40)
        if yt.size >= 512:
            y = yt
        cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        attack = float(np.argmax(rms) * 512 / sr)        # seconds from onset to peak energy
        if not np.all(np.isfinite([cent, flat, bw, attack])):
            return None
        return (path, cent, flat, bw, attack)
    except Exception:
        return None


# ---- clustering -----------------------------------------------------------

def get_features_npz(db_dir):
    files = glob.glob(os.path.join(db_dir, "models", "features_*.npz"))
    if not files:
        return None
    return sorted(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))[-1]


def two_level_cluster(X, families, fine_size):
    """Coarse KMeans into `families`, then fine KMeans (~fine_size each) within
    each family. Returns (l1, l2, cluster_d, n_fine). l2 ids are globally unique."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    n = len(X)
    pca_dim = min(50, X.shape[1])
    t0 = time.time()
    print(f"PCA {X.shape} -> {pca_dim}d ...", flush=True)
    Xr = PCA(n_components=pca_dim, svd_solver="randomized",
             random_state=0).fit_transform(X)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print(f"coarse KMeans: {families} families ...", flush=True)
    l1 = MiniBatchKMeans(n_clusters=families, batch_size=4096, n_init=3,
                         random_state=0).fit_predict(Xr)

    l2 = np.full(n, -1, dtype=np.int64)
    cd = np.zeros(n, dtype=np.float32)
    next_id = 0
    for fam in range(families):
        idx = np.where(l1 == fam)[0]
        if len(idx) == 0:
            continue
        kf = max(1, len(idx) // fine_size)
        if kf > 1:
            km = MiniBatchKMeans(n_clusters=kf, batch_size=4096, n_init=3,
                                 random_state=0)
            slab = km.fit_predict(Xr[idx])
            centers = km.cluster_centers_
        else:
            slab = np.zeros(len(idx), dtype=int)
            centers = Xr[idx].mean(0, keepdims=True)
        for s in range(int(slab.max()) + 1):
            m = idx[slab == s]
            l2[m] = next_id
            cd[m] = np.linalg.norm(Xr[m] - centers[s], axis=1)
            next_id += 1
        print(f"  family {fam:3d}: {len(idx):6d} samples -> {kf} grains", flush=True)
    return l1, l2, cd, next_id


# ---- adjective binning ----------------------------------------------------

def tercile_binner(values):
    """Return f(v) -> 0/1/2 by dataset-relative terciles (auto-calibrated)."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) < 3:
        q1 = q2 = (float(np.median(arr)) if len(arr) else 0.0)
    else:
        q1, q2 = np.quantile(arr, [1 / 3, 2 / 3])
    return lambda v: 0 if v <= q1 else (1 if v <= q2 else 2)


BRIGHT = ["dark", "warm", "bright"]       # spectral centroid
TONAL = ["tonal", "mixed", "noisy"]        # spectral flatness (low=tonal)
LENGTH = ["short", "medium", "sustained"]  # duration


def compose_label(dur_b, cent_b, flat_b, attack_b):
    parts = [LENGTH[dur_b], BRIGHT[cent_b], TONAL[flat_b]]
    if attack_b == 0:
        parts.append("punchy")
    elif attack_b == 2:
        parts.append("soft-attack")
    return " ".join(parts)


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="samples.db")
    ap.add_argument("--families", type=int, default=20, help="coarse sonic families (level 1)")
    ap.add_argument("--fine-size", type=int, default=200, help="target samples per fine grain (level 2)")
    ap.add_argument("--per-cluster", type=int, default=15, help="members sampled per grain for DSP features")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    db_dir = os.path.dirname(os.path.abspath(args.db))
    npz = get_features_npz(db_dir)
    if not npz:
        print("No features npz. Run `sample-tagger-ml export` first.")
        return
    print(f"loading {npz} ...", flush=True)
    data = np.load(npz, allow_pickle=True)
    X = data["X"]
    paths = [str(p) for p in data["paths"]]
    n = len(paths)

    l1, l2, cd, n_fine = two_level_cluster(X, args.families, args.fine_size)
    print(f"\n{n} samples -> {args.families} families / {n_fine} grains", flush=True)

    # write assignments back to samples.db
    con = sqlite3.connect(args.db, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    cols = {r[1] for r in con.execute("PRAGMA table_info(samples)")}
    if "cluster_l1" not in cols:
        con.execute("ALTER TABLE samples ADD COLUMN cluster_l1 INTEGER")
    con.executemany(
        "UPDATE samples SET cluster_l1=?, cluster_id=?, cluster_d=? WHERE path=?",
        [(int(l1[i]), int(l2[i]), float(cd[i]), paths[i]) for i in range(n)])
    con.execute("CREATE INDEX IF NOT EXISTS idx_cluster ON samples(cluster_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cluster_l1 ON samples(cluster_l1)")
    con.commit()
    print("wrote cluster_l1 / cluster_id / cluster_d to samples.db", flush=True)

    # sample members per fine grain for DSP feature extraction
    members = {}
    for i in range(n):
        members.setdefault(int(l2[i]), []).append(i)
    sample_paths = set()
    for fid, idxs in members.items():
        pick = idxs if len(idxs) <= args.per_cluster else random.sample(idxs, args.per_cluster)
        for i in pick:
            sample_paths.add(paths[i])
    sample_paths = list(sample_paths)
    print(f"\nextracting DSP features for {len(sample_paths)} sampled files "
          f"on {args.workers} workers ...", flush=True)

    feats = {}
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(extract_features, sample_paths, chunksize=16):
            done += 1
            if r:
                feats[r[0]] = r[1:]   # (cent, flat, bw, attack)
            if done % 2000 == 0:
                print(f"  {done}/{len(sample_paths)}  {done/(time.time()-t0):.0f}/s", flush=True)
    print(f"  extracted {len(feats)} in {time.time()-t0:.0f}s", flush=True)

    # durations come free from the DB
    dur = {p: (d or 0.0) for p, d in
           con.execute("SELECT path, duration_s FROM samples WHERE cluster_id IS NOT NULL")}

    # aggregate per fine grain (median of sampled members)
    def agg(idxs):
        cs, fs, bs, ats, ds = [], [], [], [], []
        for i in idxs:
            p = paths[i]
            if p in feats:
                c, f, b, a = feats[p]
                cs.append(c); fs.append(f); bs.append(b); ats.append(a)
            ds.append(dur.get(p, 0.0))
        if not cs:
            return None
        return (float(np.median(cs)), float(np.median(fs)), float(np.median(bs)),
                float(np.median(ats)), float(np.median(ds)) if ds else 0.0)

    fine = {fid: agg(idxs) for fid, idxs in members.items()}
    fine = {k: v for k, v in fine.items() if v is not None}

    # dataset-relative binners (computed across fine grains)
    cent_b = tercile_binner([v[0] for v in fine.values()])
    flat_b = tercile_binner([v[1] for v in fine.values()])
    attack_b = tercile_binner([v[3] for v in fine.values()])
    dur_b = tercile_binner([v[4] for v in fine.values()])

    def label_for(v):
        c, f, b, a, d = v
        return compose_label(dur_b(d), cent_b(c), flat_b(f), attack_b(a))

    # store sonic_clusters table
    con.execute("DROP TABLE IF EXISTS sonic_clusters")
    con.execute("""CREATE TABLE sonic_clusters (
        id INTEGER, level INTEGER, parent INTEGER, size INTEGER, label TEXT,
        centroid REAL, flatness REAL, bandwidth REAL, attack REAL, duration REAL,
        PRIMARY KEY (level, id))""")

    fine_size_count = {fid: len(idxs) for fid, idxs in members.items()}
    fine_parent = {int(l2[i]): int(l1[i]) for i in range(n)}
    rows = []
    for fid, v in fine.items():
        rows.append((fid, 2, fine_parent[fid], fine_size_count[fid], label_for(v),
                     v[0], v[1], v[2], v[3], v[4]))

    # coarse family = median over its fine grains' medians (same bin scale)
    fam_members = {}
    for fid, v in fine.items():
        fam_members.setdefault(fine_parent[fid], []).append(v)
    fam_rows = []
    for fam, vs in fam_members.items():
        med = tuple(float(np.median([v[k] for v in vs])) for k in range(5))
        size = sum(fine_size_count[fid] for fid in members if fine_parent[fid] == fam)
        fam_rows.append((fam, 1, -1, size, label_for(med),
                         med[0], med[1], med[2], med[3], med[4]))

    con.executemany("INSERT INTO sonic_clusters VALUES (?,?,?,?,?,?,?,?,?,?)",
                    fam_rows + rows)
    con.commit()
    con.close()

    # summary
    print(f"\n=== sonic families (coarse, level 1) ===")
    for r in sorted(fam_rows, key=lambda r: -r[3]):
        fam, _, _, size, label = r[0], r[1], r[2], r[3], r[4]
        n_grains = sum(1 for fid in fam_members if fid == fam)
        n_grains = len(fam_members[fam])
        print(f"  family {fam:3d}  n={size:6d}  {n_grains:3d} grains  →  {label}")
    print(f"\nwrote sonic_clusters ({len(fam_rows)} families, {len(rows)} grains) to {args.db}")


if __name__ == "__main__":
    main()
