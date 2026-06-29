#!/usr/bin/env python3
"""
simlib.py — audio-similarity search over the PANNs embeddings in samples.db.

Embeddings are 2048-d CNN14 vectors (one per file). Similarity is cosine, done
as a single normalized matrix-vector product — fast enough for the whole library
(~226k x 2048) in plain numpy, no ANN index needed.

Used by webapp.py (/api/similar) and similar.py (CLI).
"""

import json
import os
import sqlite3
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "samples.db")
DIM = 2048


class SimIndex:
    """Lazily-loaded in-memory cosine index over the embeddings table."""

    def __init__(self, db=DEFAULT_DB):
        self.db = db
        self.paths = []
        self.idx = {}
        self.mat = np.zeros((0, DIM), dtype=np.float16)   # L2-normalized rows, float16 halves RAM
        self.loaded_at = 0.0

    def load(self):
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        try:
            n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if n == 0:
                self.mat = np.zeros((0, DIM), dtype=np.float16)
                self.paths = []
                self.idx = {}
                self.loaded_at = time.time()
                return 0
            # Pre-allocate: avoids fetchall blob list + vecs list + vstack triple-copy.
            # Streaming row-by-row keeps peak overhead to ~1x the final matrix size.
            mat = np.empty((n, DIM), dtype=np.float32)
            paths = []
            row_i = 0
            for p, v in con.execute("SELECT path, vec FROM embeddings"):
                a = np.frombuffer(v, dtype=np.float32)
                if a.shape[0] == DIM:
                    mat[row_i] = a
                    paths.append(p)
                    row_i += 1
        finally:
            con.close()
        mat = mat[:row_i]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms
        self.mat = mat.astype(np.float16)  # halves RAM; dot product upcasts to float32 at runtime
        self.paths = paths
        self.idx = {p: i for i, p in enumerate(paths)}
        self.loaded_at = time.time()
        return row_i

    def ensure(self, max_age=0):
        """Load if never loaded, or if older than max_age seconds (0 = once)."""
        if self.loaded_at == 0 or (max_age and time.time() - self.loaded_at > max_age):
            self.load()

    def resolve(self, query):
        """Map a query (exact path or case-insensitive substring) to a row index."""
        if query in self.idx:
            return self.idx[query]
        ql = query.lower()
        for i, p in enumerate(self.paths):
            if ql in p.lower():
                return i
        return None

    def neighbors(self, query, k=20):
        """Return (matched_path, [(path, score), ...]) or (None, []) if not found."""
        self.ensure()
        i = self.resolve(query)
        if i is None or self.mat.shape[0] == 0:
            return None, []
        sims = self.mat @ self.mat[i]
        order = np.argpartition(-sims, min(k + 1, len(sims) - 1))[:k + 1]
        order = order[np.argsort(-sims[order])]
        out = [(self.paths[j], float(sims[j])) for j in order if j != i][:k]
        return self.paths[i], out


def fetch_meta(db, paths):
    """Pull display metadata for a list of paths from the samples table."""
    if not paths:
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        qs = ",".join("?" * len(paths))
        rows = con.execute(
            f"SELECT path,instrument,sample_type,bpm,key,duration_s,source,"
            f"path_instrument,panns_instrument,panns_conf,audio_instrument,"
            f"panns_label,panns_label_conf,panns_topk "
            f"FROM samples WHERE path IN ({qs})", paths).fetchall()
    finally:
        con.close()

    def _topk(s):
        try:
            return json.loads(s) if s else None
        except (ValueError, TypeError):
            return None

    return {r[0]: dict(instrument=r[1], sample_type=r[2], bpm=r[3],
                       key=r[4], duration_s=r[5], source=r[6],
                       path_instrument=r[7], panns_instrument=r[8],
                       panns_conf=round(r[9], 3) if r[9] else None,
                       audio_instrument=r[10],
                       panns_label=r[11],
                       panns_label_conf=round(r[12], 3) if r[12] else None,
                       panns_topk=_topk(r[13])) for r in rows}
