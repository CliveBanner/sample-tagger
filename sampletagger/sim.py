import os
import sqlite3
import time
import json
import numpy as np
from .constants import DIM


from .migrate import emb_sidecar as _emb_sidecar

class SimIndex:
    def __init__(self, db):
        self.db = db
        self.paths = []
        self.idx = {}
        self.mat = np.zeros((0, DIM), dtype=np.float16)
        self.loaded_at = 0.0

    def load(self):
        mat_file, paths_file = _emb_sidecar(self.db)
        if os.path.isfile(mat_file) and os.path.isfile(paths_file):
            # Fast path: memory-mapped float16 sidecar (already L2-normalised)
            with open(paths_file) as f:
                paths = [line.rstrip("\n") for line in f if line.strip()]
            mat = np.load(mat_file, mmap_mode="r")   # float16, zero-copy
            self.mat = mat
            self.paths = paths
            self.idx = {p: i for i, p in enumerate(paths)}
            self.loaded_at = time.time()
            return len(paths)

        # Slow path: load BLOB embeddings directly from the DB
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        try:
            n = con.execute("SELECT COUNT(*) FROM embeddings WHERE vec IS NOT NULL").fetchone()[0]
            if n == 0:
                self.mat = np.zeros((0, DIM), dtype=np.float16)
                self.paths = []
                self.idx = {}
                self.loaded_at = time.time()
                return 0
            mat = np.empty((n, DIM), dtype=np.float32)
            paths = []
            row_i = 0
            for p, v in con.execute("SELECT path, vec FROM embeddings WHERE vec IS NOT NULL"):
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
        self.mat = mat.astype(np.float16)
        self.paths = paths
        self.idx = {p: i for i, p in enumerate(paths)}
        self.loaded_at = time.time()
        return row_i

    def ensure(self, max_age=0):
        if self.loaded_at == 0 or (max_age and time.time() - self.loaded_at > max_age):
            self.load()

    def resolve(self, query):
        if query in self.idx:
            return self.idx[query]
        ql = query.lower()
        for i, p in enumerate(self.paths):
            if ql in p.lower():
                return i
        return None

    def neighbors(self, query, k=20):
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
    if not paths:
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        qs = ",".join("?" * len(paths))
        rows = con.execute(
            f"SELECT path,instrument,sample_type,bpm,key,duration_s,source,"
            f"path_instrument,panns_instrument,panns_conf,audio_instrument,"
            f"panns_label,panns_label_conf,panns_topk,rating,model_instrument,model_conf "
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
                       panns_topk=_topk(r[13]),
                       rating=r[14] or 0,
                       model_instrument=r[15],
                       model_conf=round(r[16], 3) if r[16] else None) for r in rows}

def sim_cmd(args):
    ix = SimIndex(args.db)
    n = ix.load()
    matched, hits = ix.neighbors(args.query, args.k)
    if matched is None:
        print(f"no sample matching {args.query!r} (index has {n} embeddings)")
        return
    meta = fetch_meta(args.db, [p for p, _ in hits])
    print(f"query: {matched}\\n({n} embeddings indexed)\\n")
    print(f"{'sim':>5s}  {'instr':7s} {'type':7s} {'bpm':>4s} {'key':>4s}  file")
    for p, score in hits:
        m = meta.get(p, {})
        print(f"{score:5.3f}  {str(m.get('instrument','')):7s} "
              f"{str(m.get('sample_type','')):7s} {str(m.get('bpm') or ''):>4s} "
              f"{str(m.get('key') or ''):>4s}  {os.path.basename(p)}")
