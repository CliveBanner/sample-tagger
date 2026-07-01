"""
Migrate embeddings out of samples.db into a flat float16 sidecar.

Usage:
  sample-tagger-migrate export samples.db          # dump BLOBs → sidecar (safe, non-destructive)
  sample-tagger-migrate export samples.db --force  # overwrite existing sidecar
  sample-tagger-migrate export samples.db --dry-run
  sample-tagger-migrate compact samples.db         # NULL vec column + VACUUM (run after export)

After export, sim.py and ml/export.py automatically prefer the sidecar over the DB BLOBs.
After compact, samples.db shrinks from ~2 GB to ~250 MB.
"""

import argparse
import os
import sqlite3
import time

import numpy as np

from .constants import DIM


from .embeddings import sidecar_paths

def cmd_export(db_path, dry_run=False, force=False):
    mat_file, paths_file = sidecar_paths(db_path)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        n = con.execute("SELECT COUNT(*) FROM embeddings WHERE vec IS NOT NULL").fetchone()[0]
    finally:
        con.close()

    size_mb = n * DIM * 2 // 1024 // 1024  # float16 bytes
    print(f"{n} embeddings → {mat_file}  (~{size_mb} MB float16)")

    if dry_run:
        print("dry-run: no files written")
        return

    if os.path.isfile(mat_file) and not force:
        print("sidecar already exists; use --force to overwrite")
        return

    t0 = time.time()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        mat = np.empty((n, DIM), dtype=np.float32)
        paths = []
        i = 0
        for p, v in con.execute("SELECT path, vec FROM embeddings WHERE vec IS NOT NULL"):
            a = np.frombuffer(v, dtype=np.float32)
            if a.shape[0] == DIM:
                mat[i] = a
                paths.append(p)
                i += 1
    finally:
        con.close()

    mat = mat[:i]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms
    np.save(mat_file, mat.astype(np.float16))
    with open(paths_file, "w") as f:
        f.write("\n".join(paths))

    elapsed = time.time() - t0
    actual_mb = os.path.getsize(mat_file) // 1024 // 1024
    print(f"wrote {i} embeddings in {elapsed:.1f}s  ({actual_mb} MB on disk)")
    print(f"paths: {paths_file}")
    print("next: run 'compact' to NULL the DB BLOBs and reclaim disk space")


def cmd_compact(db_path):
    mat_file, paths_file = sidecar_paths(db_path)
    if not os.path.isfile(mat_file) or not os.path.isfile(paths_file):
        print("ERROR: sidecar not found — run 'export' first")
        return

    with open(paths_file) as f:
        n_ext = sum(1 for line in f if line.strip())

    con = sqlite3.connect(db_path, timeout=120)
    try:
        n_db = con.execute("SELECT COUNT(*) FROM embeddings WHERE vec IS NOT NULL").fetchone()[0]
        print(f"sidecar: {n_ext} paths  |  DB non-null vecs: {n_db}")
        if abs(n_ext - n_db) > 200:
            print("ERROR: counts differ by more than 200 — aborting to be safe")
            return

        print("nulling vec column…")
        t0 = time.time()
        con.execute("UPDATE embeddings SET vec = NULL")
        con.commit()
        print(f"  done in {time.time()-t0:.1f}s")

        print("vacuuming (may take a few minutes on a 2 GB file)…")
        t0 = time.time()
        con.execute("VACUUM")
        print(f"  done in {time.time()-t0:.1f}s")
    finally:
        con.close()

    sz = os.path.getsize(db_path) // 1024 // 1024
    print(f"samples.db is now {sz} MB")


def main():
    ap = argparse.ArgumentParser(
        description="Migrate PANNs embeddings out of samples.db into a flat float16 sidecar"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="Dump BLOB embeddings → sidecar files (non-destructive)")
    p_exp.add_argument("db", help="Path to samples.db")
    p_exp.add_argument("--dry-run", action="store_true", help="Show what would happen, write nothing")
    p_exp.add_argument("--force", action="store_true", help="Overwrite existing sidecar")

    p_cmp = sub.add_parser("compact", help="NULL vec column and VACUUM (run after export)")
    p_cmp.add_argument("db", help="Path to samples.db")

    args = ap.parse_args()
    if args.cmd == "export":
        cmd_export(args.db, dry_run=args.dry_run, force=getattr(args, "force", False))
    elif args.cmd == "compact":
        cmd_compact(args.db)


if __name__ == "__main__":
    main()
