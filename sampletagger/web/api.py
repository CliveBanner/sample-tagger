#!/usr/bin/env python3
"""
webapp.py — tiny live dashboard for sample_tagger's samples.db.

No third-party deps (stdlib http.server + sqlite3). Reads the same DB the
scan writes to (WAL mode -> safe concurrent reads). Serves:
  /            HTML dashboard (auto-refreshes)
  /api/stats   JSON snapshot of progress + statistics

Run:  ./venv/bin/python webapp.py            (defaults: 0.0.0.0:8765, ./samples.db)
      ./venv/bin/python webapp.py --port 9000 --db /path/samples.db
"""

import argparse
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .. import sim as simlib

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(HERE, "samples.db")
LABELS_DB = os.path.join(HERE, "labels.db")
RUNLOG = os.path.join(HERE, "run.log")
ML_LOG = os.path.join(HERE, "ml.log")
CONFIG_FILE = os.path.join(HERE, "config.json")
START = time.time()

# Use venv Python if available so the scan subprocess gets all deps
PYTHON = os.path.join(HERE, "venv", "bin", "python")
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable

from ..config import Config, load_config as load_core_config
from dataclasses import asdict

def load_config():
    return asdict(load_core_config(CONFIG_FILE))

def save_config(data):
    cfg = load_core_config(CONFIG_FILE)
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, type(getattr(cfg, k))(v))
    
    out = asdict(cfg)
    with open(CONFIG_FILE, "w") as f:
        json.dump(out, f, indent=2)
    return out

# Canonical default taxonomy + display palette. `labels.db` is the live source of
# truth (editable via the UI); this dict is the bootstrap seed (its keys seed an
# empty labels.db, see migrate_db) AND the map/legend color palette. Keeping both
# derived from one dict means the seed list and the colors can never drift apart.
# Any label added later that isn't here still gets a deterministic fallback color
# in get_colors(), so coverage is automatic.
INSTR_COLORS = {
    # percussion (embedding-separability-optimized taxonomy)
    "kick": "#f92672", "snare_clap": "#fd971f", "hats_cymbals": "#e6db74",
    "tom": "#e6a23c", "perc": "#a6e22e",
    # low end
    "bass": "#66d9ef",
    # keys
    "piano_keys": "#5c6bc0", "organ": "#42a5f5",
    # tuned percussion
    "mallet": "#c0ca33",
    # melodic acoustic
    "guitar": "#26a69a", "strings": "#29b6f6", "brass": "#ffb300", "winds": "#9ccc65",
    # synth
    "synth": "#9a6cff", "pad": "#7986cb",
    # voice
    "vocal": "#ff5fa2",
    # texture / non-pitched
    "sfx": "#75715e",
}

AUDIO_CT = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
            ".aif": "audio/aiff", ".aiff": "audio/aiff", ".ogg": "audio/ogg"}

_SIM = None            # lazily-built SimIndex
_SIM_LAST_USE = 0.0    # timestamp of last /api/similar call
_SIM_TTL = 600         # evict SimIndex after 10 min idle (saves ~1-2 GB)
_MAP = None            # cached map payload (dict)
_CLUSTERS = None       # cached cluster summaries (list)
_CLUSTERS_MTIME = 0.0  # samples.db mtime when _CLUSTERS was built
_RUN_PROC = None       # Popen handle for a scan started via /api/run/start
_ML_PROC = None        # Popen handle for the ML pipeline
_ML_STATE = "idle"     # idle | running | done | error


from contextlib import contextmanager

@contextmanager
def ro(db=None):
    db_path = db or DB
    if not os.path.exists(db_path):
        yield None
        return
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        yield con
    finally:
        con.close()

def _json(req, obj, code=200):
    req._send(code, json.dumps(obj), "application/json")

# ---- run management -------------------------------------------------------

def _tagger_pid():
    """Return PID of any live sample_tagger.py process found in /proc."""
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    if b"sampletagger.cli" in f.read():
                        return int(pid)
            except OSError:
                continue
    except OSError:
        pass
    return None


def run_status():
    global _RUN_PROC
    running, pid = False, None
    if _RUN_PROC is not None:
        if _RUN_PROC.poll() is None:
            running, pid = True, _RUN_PROC.pid
        else:
            _RUN_PROC = None
    if not running:
        pid = _tagger_pid()
        running = pid is not None
    progress = {}
    try:
        with open(RUNLOG) as f:
            for line in f:
                # label stage: "225372 files to process on 3 workers."
                m = re.search(r"(\d+) files to process on \d+ workers", line)
                if m:
                    progress = {"total": int(m.group(1)), "done": 0,
                                "rate": 0, "eta_min": None}
                # label stage progress: "  200/225372  1.2/s  eta  456.7m  err=0"
                m = re.search(r"(\d+)/(\d+)\s+([\d.]+)/s\s+eta\s+([\d.]+)m", line)
                if m:
                    progress = {"done": int(m.group(1)), "total": int(m.group(2)),
                                "rate": float(m.group(3)), "eta_min": float(m.group(4))}
    except OSError:
        pass
    return {"running": running, "pid": pid, "progress": progress}


def run_start(stage):
    global _RUN_PROC
    if scan_running():
        return {"ok": False, "msg": "a scan is already running"}
    cfg = load_config()
    gpu_py = cfg.get("gpu_python", "").strip()
    py = gpu_py if (stage == "label" and gpu_py and os.path.isfile(gpu_py)) else PYTHON
    cmd = [py, "-m", "sampletagger.cli", "--db", DB, "-j", str(cfg.get("workers", 5))]
    if cfg.get("limit"): cmd += ["--limit", str(int(cfg["limit"]))]
    cmd.append(stage)

    if stage == "discover":
        cmd.append(cfg["library_path"])
        if cfg.get("trust_db"): cmd.append("--trust-db")
        if cfg.get("no_cache"): cmd.append("--no-cache")

    elif stage == "label":
        classifiers = []
        if cfg.get("label_path"):  classifiers.append("path")
        if cfg.get("label_audio"): classifiers.append("audio")
        if cfg.get("label_panns"): classifiers.append("panns")
        if not classifiers:
            return {"ok": False, "msg": "no classifiers selected"}
        cmd += ["--classifiers", ",".join(classifiers)]
        redo = cfg.get("redo", "").strip()
        if redo:
            cmd += ["--redo", redo]
    logf = open(RUNLOG, "a")
    _RUN_PROC = subprocess.Popen(cmd, stdout=logf, stderr=logf)
    return {"ok": True, "pid": _RUN_PROC.pid, "stage": stage}


def run_stop():
    global _RUN_PROC
    pid = None
    if _RUN_PROC is not None and _RUN_PROC.poll() is None:
        pid = _RUN_PROC.pid
        _RUN_PROC.terminate()
        _RUN_PROC = None
    else:
        pid = _tagger_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    return {"ok": True, "pid": pid}




def ml_run_start():
    global _ML_PROC, _ML_STATE
    if _ML_PROC is not None and _ML_PROC.poll() is None:
        return {"ok": False, "msg": "ML pipeline already running"}
    ml_bin = os.path.join(HERE, "venv", "bin", "sample-tagger-ml")
    if not os.path.isfile(ml_bin):
        return {"ok": False, "msg": "sample-tagger-ml not found; is the venv installed?"}
    cmd = f'"{ml_bin}" export "{DB}" && "{ml_bin}" train "{DB}" && "{ml_bin}" predict "{DB}"'
    logf = open(ML_LOG, "w")
    _ML_PROC = subprocess.Popen(["/bin/sh", "-c", cmd], stdout=logf, stderr=logf)
    _ML_STATE = "running"
    return {"ok": True, "pid": _ML_PROC.pid}


def ml_run_stop():
    global _ML_PROC, _ML_STATE
    if _ML_PROC is not None and _ML_PROC.poll() is None:
        pid = _ML_PROC.pid
        _ML_PROC.terminate()
        _ML_PROC = None
        _ML_STATE = "idle"
        return {"ok": True, "pid": pid}
    return {"ok": True, "pid": None}


def ml_run_status():
    global _ML_PROC, _ML_STATE
    running = False
    pid = None
    if _ML_PROC is not None:
        if _ML_PROC.poll() is None:
            running, pid = True, _ML_PROC.pid
            _ML_STATE = "running"
        else:
            _ML_STATE = "done" if _ML_PROC.returncode == 0 else "error"
            _ML_PROC = None
    head_path = os.path.join(HERE, "models", "head.joblib")
    last_trained = None
    try:
        last_trained = os.path.getmtime(head_path)
    except OSError:
        pass
    log_tail = []
    try:
        with open(ML_LOG) as f:
            log_tail = [l.rstrip() for l in f.readlines()[-20:] if l.strip()]
    except OSError:
        pass
    return {"running": running, "pid": pid, "state": _ML_STATE,
            "log_tail": log_tail, "last_trained": last_trained}


def valid_sample(path):
    """Only serve files that are actually in the catalog (no path traversal)."""
    with ro() as con:
        if not con: return False
        return con.execute("SELECT 1 FROM samples WHERE path=? LIMIT 1",
                           (path,)).fetchone() is not None


# Derived from the canonical palette above so the bootstrap seed can't drift
# from the colors. labels.db is the live source of truth once seeded.
INSTRUMENTS = list(INSTR_COLORS)


SAMPLE_TYPES = ("oneshot", "loop")

def label_type_api(path, sample_type):
    if not path:
        return {"ok": False, "msg": "no path"}
    if sample_type not in SAMPLE_TYPES and sample_type != "":
        return {"ok": False, "msg": f"unknown type: {sample_type}"}
    con = sqlite3.connect(DB, timeout=10)
    try:
        con.execute(
            "UPDATE samples SET human_sample_type=?, ts=? WHERE path=?",
            (sample_type or None, time.time(), path))
        con.commit()
    finally:
        con.close()
    return {"ok": True, "path": path, "sample_type": sample_type}


def label_api(path, instrument):
    """Save a human instrument label for one sample."""
    global _MAP, _CLUSTERS
    if not path:
        return {"ok": False, "msg": "no path"}
    if instrument and instrument not in get_labels():
        return {"ok": False, "msg": f"unknown instrument: {instrument}"}
    con = sqlite3.connect(DB, timeout=10)
    try:
        con.execute(
            """UPDATE samples SET 
                 human_instrument=?, 
                 label_source=?, 
                 ts=?,
                 instrument = COALESCE(?, model_instrument, panns_instrument, path_instrument),
                 source = CASE 
                    WHEN ? IS NOT NULL AND ? != '' THEN 'human'
                    WHEN model_instrument IS NOT NULL THEN 'model'
                    WHEN panns_instrument IS NOT NULL THEN 'panns'
                    WHEN path_instrument IS NOT NULL THEN 'path'
                    ELSE 'none' END
               WHERE path=?""",
            (instrument or None, ("single" if instrument else None), time.time(), 
             instrument or None, instrument or None, instrument or None, path))
        con.commit()
    finally:
        con.close()
    _MAP = None
    _CLUSTERS = None
    return {"ok": True, "path": path, "instrument": instrument}


def propagate_candidates(path, k=24):
    """Nearest embedding neighbors of a just-labeled sample, as propagation
    candidates. Already human-labeled samples are excluded."""
    if not path:
        return {"items": []}
    ix = get_sim()
    ix.ensure(max_age=0)
    matched, hits = ix.neighbors(path, k)
    if matched is None or not hits:
        return {"items": []}
    score = {p: s for p, s in hits}
    cand_paths = list(score)
    with ro() as con:
        if not con:
            return {"items": []}
        qs = ",".join("?" * len(cand_paths))
        rows = con.execute(
            f"SELECT path, model_instrument, model_conf, path_instrument, panns_instrument, "
            f"human_instrument, duration_s, sample_type, rating "
            f"FROM samples WHERE path IN ({qs})", cand_paths).fetchall()
    items = []
    for r in rows:
        if r[5]:                      # already has a human label — don't propagate over it
            continue
        items.append({"path": r[0], "name": os.path.basename(r[0]),
                      "score": round(score.get(r[0], 0), 3),
                      "model_instrument": r[1],
                      "model_conf": round(r[2], 3) if r[2] else None,
                      "path_instrument": r[3], "panns_instrument": r[4],
                      "duration_s": r[6], "sample_type": r[7], "rating": r[8] or 0})
    items.sort(key=lambda d: -d["score"])
    return {"items": items}


def _bulk_label(paths, instrument, source, only_unlabeled=True):
    if not paths:
        return 0
    con = sqlite3.connect(DB, timeout=20)
    n = 0
    try:
        ts = time.time()
        if only_unlabeled:
            cur = con.executemany(
                """UPDATE samples SET 
                     human_instrument=?, 
                     label_source=?, 
                     ts=?,
                     instrument=?,
                     source='human'
                   WHERE path=? AND (human_instrument IS NULL OR human_instrument='')""",
                [(instrument, source, ts, instrument, p) for p in paths])
        else:
            cur = con.executemany(
                """UPDATE samples SET 
                     human_instrument=?, 
                     label_source=?, 
                     ts=?,
                     instrument = COALESCE(?, model_instrument, panns_instrument, path_instrument),
                     source = CASE 
                        WHEN ? IS NOT NULL AND ? != '' THEN 'human'
                        WHEN model_instrument IS NOT NULL THEN 'model'
                        WHEN panns_instrument IS NOT NULL THEN 'panns'
                        WHEN path_instrument IS NOT NULL THEN 'path'
                        ELSE 'none' END
                   WHERE path=?""",
                [(instrument, source, ts, instrument, instrument, instrument, p) for p in paths])
        n = cur.rowcount
        con.commit()
    finally:
        con.close()
    return max(n, 0)

def label_propagate(paths, instrument):
    """Apply one instrument label to a hand-picked set of neighbors at once."""
    global _MAP, _CLUSTERS
    if not paths or not instrument:
        return {"ok": False, "msg": "missing paths or instrument"}
    if instrument not in get_labels():
        return {"ok": False, "msg": f"unknown instrument: {instrument}"}
        
    n = _bulk_label(paths, instrument, "propagate", only_unlabeled=True)
    _MAP = None
    _CLUSTERS = None
    return {"ok": True, "n": n, "instrument": instrument}


def build_clusters():
    """Summarize every cluster: unlabeled size, dominant model label + agreement,
    and the medoid (sample closest to the centroid). Cached; invalidated on any
    label write."""
    global _CLUSTERS, _CLUSTERS_MTIME
    db_mtime = _db_mtime()
    if _CLUSTERS is not None and _CLUSTERS_MTIME == db_mtime:
        return _CLUSTERS
    _CLUSTERS_MTIME = db_mtime

    with ro() as con:
        if not con:
            _CLUSTERS = []
            return _CLUSTERS
        try:
            base_rows = con.execute("""
                SELECT s.cluster_id, 
                       COUNT(*),
                       SUM(CASE WHEN s.human_instrument IS NULL OR s.human_instrument='' THEN 1 ELSE 0 END),
                       MIN(s.cluster_d),
                       s.path,
                       c.name
                FROM samples s
                LEFT JOIN clusters c ON s.cluster_id = c.id
                WHERE s.cluster_id IS NOT NULL 
                GROUP BY s.cluster_id
            """).fetchall()
            
            label_rows = con.execute("""
                SELECT cluster_id, model_instrument, COUNT(*) as c
                FROM samples 
                WHERE cluster_id IS NOT NULL 
                  AND (human_instrument IS NULL OR human_instrument='') 
                  AND model_instrument IS NOT NULL
                GROUP BY cluster_id, model_instrument
            """).fetchall()
        except sqlite3.OperationalError:
            _CLUSTERS = []
            return _CLUSTERS

    from collections import defaultdict
    dom_label = {}
    dom_count = defaultdict(int)
    for cid, model_i, c in label_rows:
        if c > dom_count[cid]:
            dom_count[cid] = c
            dom_label[cid] = model_i

    out = []
    for cid, n_tot, n_unlab, _min_d, path, cname in base_rows:
        if not n_unlab:
            continue
        dl = dom_label.get(cid)
        agr = dom_count[cid] / n_unlab if n_unlab > 0 else 0.0
        out.append({
            "id": int(cid), "n": n_unlab, "n_total": n_tot,
            "label": dl, "agreement": round(agr, 2),
            "medoid": path,
            "name": cname or ""
        })
    _CLUSTERS = out
    return _CLUSTERS


def clusters_list(mode="value", limit=300):
    items = [c for c in build_clusters() if c["n"] >= 2]
    if mode == "messy":
        items = [c for c in items if c["agreement"] < 0.6]
        items.sort(key=lambda c: -c["n"])
    else:                                           # "value": big & pure first
        items = [c for c in items if c["agreement"] >= 0.5]
        items.sort(key=lambda c: -(c["n"] * c["agreement"]))
    total = len(items)
    items = items[:limit]
    return {"items": [{**c, "medoid_name": os.path.basename(c["medoid"]) if c["medoid"] else ""}
                      for c in items], "total": total}


def cluster_detail(cid):
    with ro() as con:
        if not con:
            return {"members": []}
        rows = con.execute(
            "SELECT path, model_instrument, model_conf, path_instrument, panns_instrument, "
            "human_instrument, duration_s, sample_type, rating, cluster_d "
            "FROM samples WHERE cluster_id=? ORDER BY cluster_d ASC", (cid,)).fetchall()
    members = [{"path": r[0], "name": os.path.basename(r[0]), "model_instrument": r[1],
                "model_conf": round(r[2], 3) if r[2] else None, "path_instrument": r[3],
                "panns_instrument": r[4], "human_instrument": r[5], "duration_s": r[6],
                "sample_type": r[7], "rating": r[8] or 0,
                "cluster_d": round(r[9], 4) if r[9] is not None else None} for r in rows]
    from collections import Counter
    c = Counter(m["model_instrument"] for m in members
                if not m["human_instrument"] and m["model_instrument"])
    return {"id": cid, "members": members, "label": c.most_common(1)[0][0] if c else None}


def label_cluster(cid, instrument, exclude=None):
    """Apply one label to all unlabeled members of a cluster except excluded paths."""
    global _MAP, _CLUSTERS
    if instrument not in get_labels():
        return {"ok": False, "msg": f"unknown instrument: {instrument}"}
    exclude = set(exclude or [])
    with ro() as con:
        if not con:
            return {"ok": False, "msg": "no db"}
        rows = con.execute(
            "SELECT path FROM samples WHERE cluster_id=? "
            "AND (human_instrument IS NULL OR human_instrument='')", (cid,)).fetchall()
    targets = [p for (p,) in rows if p not in exclude]
    n = _bulk_label(targets, instrument, "cluster", only_unlabeled=True)
    _MAP = None
    _CLUSTERS = None
    return {"ok": True, "n": n, "instrument": instrument}


def rate_api(path, rating):
    """Save a 1-5 star rating for one sample (0 clears it)."""
    if not path:
        return {"ok": False, "msg": "no path"}
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        return {"ok": False, "msg": "bad rating"}
    if rating < 0 or rating > 5:
        return {"ok": False, "msg": "rating out of range"}
    con = sqlite3.connect(DB, timeout=10)
    try:
        con.execute("UPDATE samples SET rating=?, ts=? WHERE path=?",
                    (rating, time.time(), path))
        con.commit()
    finally:
        con.close()
    return {"ok": True, "path": path, "rating": rating}


def get_sim():
    global _SIM, _SIM_LAST_USE
    _SIM_LAST_USE = time.time()
    if _SIM is None:
        _SIM = simlib.SimIndex(DB)
    return _SIM


def _maybe_evict_sim():
    global _SIM, _SIM_LAST_USE
    if _SIM is not None and time.time() - _SIM_LAST_USE > _SIM_TTL:
        _SIM = None


def similar_api(query, k=24):
    ix = get_sim()
    ix.ensure(max_age=0)              # load once; eviction + manual reload handle freshness
    matched, hits = ix.neighbors(query, k)
    if matched is None:
        return {"query": query, "matched": None, "hits": [], "n": len(ix.paths)}
    meta = simlib.fetch_meta(DB, [p for p, _ in hits])
    return {"query": query, "matched": matched, "matched_name": os.path.basename(matched),
            "n": len(ix.paths),
            "hits": [dict(path=p, name=os.path.basename(p), score=round(s, 3),
                          **meta.get(p, {})) for p, s in hits]}


def _db_mtime():
    """Newest mtime across samples.db and its WAL — changes on any write, so
    caches can detect out-of-process edits (CLI predict/cluster, manual SQL)."""
    m = 0.0
    for suffix in ("", "-wal"):
        try:
            m = max(m, os.path.getmtime(DB + suffix))
        except OSError:
            pass
    return m


def build_map():
    """Join projection + samples into compact parallel arrays for the canvas.

    Cache is keyed by the projection sidecar mtime AND samples.db mtime, so the
    map rebuilds both when the layout is re-projected and when labels change —
    including writes made by CLI tools outside this process.
    """
    global _MAP
    proj_db = DB + ".proj"
    try:
        sidecar_mtime = os.path.getmtime(proj_db)
    except OSError:
        sidecar_mtime = 0
    db_mtime = _db_mtime()
    if (_MAP is not None
            and _MAP.get("_sidecar_mtime") == sidecar_mtime
            and _MAP.get("_db_mtime") == db_mtime):
        return _MAP
    fam_labels = []        # sonic family descriptors, indexed by family id
    with ro() as con:
        if not con:
            rows = []
        else:
            try:
                cols = ("s.instrument, s.human_instrument, s.model_instrument, "
                        "s.path_instrument, s.panns_instrument, s.audio_instrument, "
                        "s.sample_type, s.duration_s, s.label_source, s.cluster_l1")
                if sidecar_mtime:              # sidecar exists — use it
                    con.execute("ATTACH DATABASE ? AS pj", (f"file:{proj_db}?mode=ro",))
                    rows = con.execute(
                        f"SELECT p.path, p.x, p.y, {cols} "
                        "FROM pj.projection p JOIN samples s ON s.path=p.path").fetchall()
                else:                          # legacy: projection table inside samples.db
                    rows = con.execute(
                        f"SELECT p.path, p.x, p.y, {cols} "
                        "FROM projection p JOIN samples s ON s.path=p.path").fetchall()
            except sqlite3.OperationalError:
                rows = []
            fam_labels = sonic_family_labels(con)
    colors_dict = get_colors()
    cats = sorted(colors_dict.keys())
    cidx = {c: i for i, c in enumerate(cats)}
    none_idx = len(cats)                       # bucket for "no label in this field"
    tcode = {"oneshot": 0, "loop": 1}
    # provenance codes for label_source (how a human_instrument was applied)
    LS_CODE = {"single": 0, "cluster": 1, "map": 2, "propagate": 3, "llm": 4}
    LS_NONE = 5
    # instrument-label fields index into `cats`; the sonic "family" field indexes
    # into fam_labels (a separate category space), with its own none bucket.
    FIELD_COLS = ("instrument", "human", "model", "path", "panns", "audio")
    fam_none = len(fam_labels)
    fields = {k: [] for k in FIELD_COLS}
    fields["family"] = []
    xs, ys, ts, ds, ls, paths = [], [], [], [], [], []
    for (path, x, y, instr, human_i, model_i, path_i, panns_i, audio_i,
         st, dur, lsrc, cl1) in rows:
        xs.append(round(x, 4)); ys.append(round(y, 4))
        fields["instrument"].append(cidx.get(instr, none_idx))
        fields["human"].append(cidx.get(human_i, none_idx))
        fields["model"].append(cidx.get(model_i, none_idx))
        fields["path"].append(cidx.get(path_i, none_idx))
        fields["panns"].append(cidx.get(panns_i, none_idx))
        fields["audio"].append(cidx.get(audio_i, none_idx))
        fields["family"].append(cl1 if (cl1 is not None and 0 <= cl1 < fam_none) else fam_none)
        ts.append(tcode.get(st, 2))
        ds.append(round(dur, 2) if dur else 0)
        ls.append(LS_CODE.get(lsrc, LS_NONE))
        paths.append(path)
    fam_colors = [f"hsl({int(360*i/max(1,fam_none))},55%,55%)" for i in range(fam_none)]
    _MAP = {"_sidecar_mtime": sidecar_mtime if paths else 0, "_db_mtime": db_mtime,
            "paths": paths, "x": xs, "y": ys, "t": ts, "d": ds,
            "fields": fields, "ls": ls,
            "cats": cats, "colors": [colors_dict[c] for c in cats],
            "famCats": fam_labels, "famColors": fam_colors,
            "n": len(paths)}
    return _MAP


def map_api():
    m = build_map()
    return {
        "x": m["x"], "y": m["y"], "t": m["t"], "d": m["d"],
        "fields": m["fields"], "ls": m["ls"],
        "cats": m["cats"], "colors": m["colors"],
        "famCats": m["famCats"], "famColors": m["famColors"],
        "n": m["n"], "sidecar_mtime": m.get("_sidecar_mtime", 0),
    }


def label_map(data):
    """Batch label from map selection. Requires sidecar_mtime token to ensure
    indices haven't shifted out from under the client."""
    global _MAP, _CLUSTERS
    indices = data.get("indices", [])
    instrument = data.get("instrument")
    sidecar_mtime = data.get("sidecar_mtime")
    mode = data.get("mode", "all")
    
    if not indices or not instrument:
        return {"ok": False, "msg": "missing indices or instrument"}
    if instrument not in get_labels():
        return {"ok": False, "msg": f"unknown instrument: {instrument}"}
        
    m = build_map()
    if not m or m.get("_sidecar_mtime") != sidecar_mtime:
        return {"ok": False, "stale": True}
        
    paths = [m["paths"][i] for i in indices if 0 <= i < m["n"]]
    if not paths:
        return {"ok": False, "msg": "no valid indices"}
        
    n = _bulk_label(paths, instrument, "map", only_unlabeled=(mode == "unlabeled"))
    _MAP = None
    _CLUSTERS = None
    return {"ok": True, "n": n, "instrument": instrument}


def sonic_family_labels(con):
    """Sonic family descriptors indexed by family id; [] if not computed yet."""
    try:
        rows = con.execute("SELECT id, label FROM sonic_clusters WHERE level=1").fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        return []
    labels = [f"family {i}" for i in range(max(r[0] for r in rows) + 1)]
    for i, lab in rows:
        labels[i] = lab or f"family {i}"
    return labels


def _sonic_row(r):
    return dict(id=r[0], n=r[1], label=r[2], centroid=round(r[3], 1),
                flatness=round(r[4], 4), attack=round(r[5], 3), duration=round(r[6], 2))


def sonic_families():
    """Coarse sonic families (descriptor + size), largest first."""
    with ro() as con:
        if not con:
            return {"families": []}
        try:
            rows = con.execute(
                "SELECT id, size, label, centroid, flatness, attack, duration "
                "FROM sonic_clusters WHERE level=1 ORDER BY size DESC").fetchall()
        except sqlite3.OperationalError:
            return {"families": [], "pending": True}
    return {"families": [_sonic_row(r) for r in rows]}


def sonic_grains(family):
    """Fine grains within a family (descriptor + size + medoid), largest first."""
    with ro() as con:
        if not con:
            return {"grains": []}
        try:
            rows = con.execute(
                "SELECT id, size, label, centroid, flatness, attack, duration "
                "FROM sonic_clusters WHERE level=2 AND parent=? ORDER BY size DESC",
                (family,)).fetchall()
        except sqlite3.OperationalError:
            return {"grains": []}
        grains = []
        for r in rows:
            g = _sonic_row(r)
            md = con.execute("SELECT path FROM samples WHERE cluster_id=? "
                             "ORDER BY cluster_d ASC LIMIT 1", (r[0],)).fetchone()
            g["medoid"] = md[0] if md else None
            g["medoid_name"] = os.path.basename(md[0]) if md else None
            grains.append(g)
    return {"family": family, "grains": grains}


def sonic_members(grain, limit=60):
    """Member samples of a grain, closest to the grain core first (for auditioning)."""
    with ro() as con:
        if not con:
            return {"members": []}
        rows = con.execute(
            "SELECT path, duration_s, sample_type, cluster_d FROM samples "
            "WHERE cluster_id=? ORDER BY cluster_d ASC LIMIT ?", (grain, limit)).fetchall()
    return {"grain": grain, "members": [
        dict(path=r[0], name=os.path.basename(r[0]), duration_s=r[1],
             sample_type=r[2], cluster_d=round(r[3], 4) if r[3] is not None else None)
        for r in rows]}


def sonic_for(con, path):
    """Audio-only sonic descriptors (grain + coarse family) for a sample, from the
    sonic_clusters table. Returns None if sonic labels haven't been computed yet."""
    try:
        row = con.execute("SELECT cluster_id, cluster_l1 FROM samples WHERE path=?",
                          (path,)).fetchone()
        if not row or row[0] is None:
            return None
        grain = con.execute("SELECT label FROM sonic_clusters WHERE level=2 AND id=?",
                            (row[0],)).fetchone()
        fam = con.execute("SELECT label FROM sonic_clusters WHERE level=1 AND id=?",
                          (row[1],)).fetchone() if row[1] is not None else None
        if not grain:
            return None
        return {"grain": grain[0], "family": fam[0] if fam else None}
    except sqlite3.OperationalError:
        return None


def point_api(i):
    m = build_map()
    if i < 0 or i >= len(m["paths"]):
        return {}
    path = m["paths"][i]
    meta = simlib.fetch_meta(DB, [path]).get(path, {})
    with ro() as con:
        sonic = sonic_for(con, path) if con else None
    return dict(path=path, name=os.path.basename(path), sonic=sonic, **meta)


# --- on-demand map regeneration (runs project.py as a subprocess) ------------
_REPROJ = {"running": False, "ok": None, "msg": "idle", "ts": 0}


def _do_reproject():
    global _MAP
    try:
        r = subprocess.run([sys.executable, "-m", "sampletagger.projection", "--db", DB],
                           capture_output=True, text=True, timeout=3600)
        ok = r.returncode == 0
        out = (r.stdout if ok else r.stderr).strip().splitlines()
        _REPROJ["ok"] = ok
        _REPROJ["msg"] = out[-1][:200] if out else ("done" if ok else "failed")
    except Exception as e:
        _REPROJ["ok"] = False
        _REPROJ["msg"] = str(e)[:200]
    finally:
        _REPROJ["running"] = False
        _REPROJ["ts"] = time.time()
        _MAP = None                  # force the map to rebuild from the new table


def reproject_start():
    if _REPROJ["running"]:
        return dict(_REPROJ)
    _REPROJ.update(running=True, ok=None, msg="projecting…")
    threading.Thread(target=_do_reproject, daemon=True).start()
    return dict(_REPROJ)


def total_target():
    """Total file count, from run.log's scan line; falls back to 0 (unknown)."""
    try:
        with open(RUNLOG) as f:
            for line in f:
                m = re.search(r"(\d+) audio files", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return 0


def scan_running():
    """Best-effort: is a sample_tagger.py process alive?"""
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    if b"sampletagger.cli" in f.read():
                        return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def q(con, sql, params=()):
    return con.execute(sql, params).fetchall()


def stats():
    _maybe_evict_sim()
    with ro() as con:
        if not con:
            return {"ready": False, "msg": "samples.db not found yet"}
        con.row_factory = sqlite3.Row
        now = time.time()
        rs = run_status()
        prog = rs.get("progress", {})
        processed = q(con, "SELECT COUNT(*) FROM samples WHERE status != 'missing'")[0][0]
        ok = q(con, "SELECT COUNT(*) FROM samples WHERE status='ok'")[0][0]
        errors = q(con, "SELECT COUNT(*) FROM samples WHERE status='error'")[0][0]
        tagged = q(con, "SELECT COUNT(*) FROM samples WHERE tagged=1")[0][0]
        last_ts = q(con, "SELECT MAX(ts) FROM samples")[0][0] or 0

        total = prog.get("total") or processed
        rate = prog.get("rate") or 0
        eta_min = prog.get("eta_min")  # from log; None when idle

        def dist(col, where=""):
            return [dict(label=r[0] if r[0] is not None else "—", n=r[1])
                    for r in q(con, f"SELECT {col}, COUNT(*) FROM samples "
                                    f"WHERE status='ok' {where} GROUP BY {col} "
                                    f"ORDER BY COUNT(*) DESC")]

        active = q(con, "SELECT COUNT(*) FROM samples WHERE status != 'missing'")[0][0]
        cov_row = q(con, """
            SELECT COUNT(path_instrument), COUNT(panns_instrument),
                   COUNT(audio_instrument),
                   COUNT(human_instrument),
                   SUM(CASE WHEN path_instrument IS NOT NULL
                             OR panns_instrument IS NOT NULL
                             OR audio_instrument IS NOT NULL THEN 1 ELSE 0 END)
            FROM samples WHERE status != 'missing'""")[0]
        # load_config returns a dict now, so we can get keys directly
        panns_min_dur = load_config().get("panns_min_duration", 1.0)
        panns_skipped = q(con, "SELECT COUNT(*) FROM samples "
                               "WHERE status != 'missing' AND duration_s < ?",
                          (panns_min_dur,))[0][0]
        coverage = {
            "active": active,
            "path":  {"n": cov_row[0], "total": active},
            "panns": {"n": cov_row[1], "total": active},
            "audio": {"n": cov_row[2], "total": active},
            "human": {"n": cov_row[3], "total": active},
            "any":   {"n": cov_row[4] or 0, "total": active},
        }

        def inst_dist(col):
            rows = con.execute(
                f"SELECT {col}, COUNT(*) n FROM samples "
                f"WHERE status != 'missing' AND {col} IS NOT NULL "
                f"GROUP BY {col} ORDER BY n DESC LIMIT 14").fetchall()
            return [{"label": r[0], "n": r[1]} for r in rows]

        def top_n_with_other(col, n):
            rows = con.execute(
                f"SELECT {col}, COUNT(*) n FROM samples "
                f"WHERE status != 'missing' AND {col} IS NOT NULL "
                f"GROUP BY {col} ORDER BY n DESC").fetchall()
            if not rows: return []
            top = [{"label": r[0], "n": r[1]} for r in rows[:n]]
            other = sum(r[1] for r in rows[n:])
            if other > 0:
                top.append({"label": "other", "n": other, "color": "#75715e"})
            return top

        path_dist  = inst_dist("path_instrument")
        panns_dist = inst_dist("panns_instrument")
        panns_raw_dist = top_n_with_other("panns_label", 12)
        audio_dist = inst_dist("audio_instrument")
        human_dist = inst_dist("human_instrument")
        sample_type = inst_dist("sample_type")
        keys = inst_dist("key")

        bpm_rows = q(con, "SELECT (bpm/10)*10 AS bucket, COUNT(*) FROM samples "
                          "WHERE bpm IS NOT NULL GROUP BY bucket ORDER BY bucket")
        bpm_hist = [dict(label=f"{int(b)}–{int(b)+9}", n=n) for b, n in bpm_rows]
        
        has_model = False
        try:
            con.execute("SELECT model_instrument FROM samples LIMIT 1")
            has_model = True
        except sqlite3.OperationalError:
            pass
            
        model_dist = []
        model_conf_hist = []
        model_cov = 0
        if has_model:
            model_dist = inst_dist("model_instrument")
            # Histogram for confidence (0.0 to 1.0 in buckets of 0.1)
            conf_rows = q(con, "SELECT CAST(model_conf * 10 AS INTEGER) AS bucket, COUNT(*) FROM samples "
                               "WHERE model_conf IS NOT NULL GROUP BY bucket ORDER BY bucket")
            model_conf_hist = [dict(label=f"{(b/10):.1f}+", n=n) for b, n in conf_rows]
            model_cov = q(con, "SELECT COUNT(*) FROM samples WHERE model_conf IS NOT NULL")[0][0]
            coverage["model"] = {"n": model_cov, "total": active}

        try:
            with open(RUNLOG) as f:
                log_tail = [l.rstrip() for l in f.readlines()[-12:] if l.strip()]
        except OSError:
            log_tail = []

        # sonic family distribution (audio-only clusters); empty until computed
        try:
            sonic_dist = [{"label": r[1], "n": r[0]} for r in con.execute(
                "SELECT size, label FROM sonic_clusters WHERE level=1 ORDER BY size DESC")]
        except sqlite3.OperationalError:
            sonic_dist = []
        return {
            "ready": True,
            "scan_running": scan_running(),
            "active": active, "processed": processed, "errors": errors,
            "rate": round(rate, 2), "eta_min": round(eta_min, 1) if eta_min is not None else None,
            "pct": round(100 * prog.get("done", 0) / total, 1) if total else None,
            "label_done": prog.get("done", 0), "label_total": total,
            "coverage": coverage,
            "path_dist": path_dist, "panns_dist": panns_dist, "panns_raw_dist": panns_raw_dist,
            "audio_dist": audio_dist, "human_dist": human_dist, "sample_type": sample_type,
            "model_dist": model_dist, "model_conf_hist": model_conf_hist,
            "keys": keys, "bpm_hist": bpm_hist, "log_tail": log_tail,
            "sonic_dist": sonic_dist,
        }


def review_queue(mode="unified", limit=80):
    if not os.path.exists(DB):
        return {"items": [], "total": 0}
        
    where = "status != 'missing' AND source != 'human' AND (human_instrument IS NULL OR human_instrument='')"
    
    with ro() as con:
        if not con:
            return {"items": [], "total": 0}
        total = con.execute(f"SELECT COUNT(*) FROM samples WHERE {where}").fetchone()[0]
        
        query = f"""
        SELECT path, path_instrument, panns_instrument, panns_conf,
               audio_instrument, duration_s, sample_type, human_sample_type, human_instrument,
               model_instrument, model_conf, rating, cluster_id, cluster_l1
        FROM (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY COALESCE(model_instrument, 'unknown') 
            ORDER BY COALESCE(model_margin, model_conf, 1.0) - (
              CASE WHEN 
                    (path_instrument IS NOT NULL AND panns_instrument IS NOT NULL AND path_instrument != panns_instrument)
                 OR (path_instrument IS NOT NULL AND audio_instrument IS NOT NULL AND path_instrument != audio_instrument)
                 OR (panns_instrument IS NOT NULL AND audio_instrument IS NOT NULL AND panns_instrument != audio_instrument)
                 OR (model_instrument IS NOT NULL AND path_instrument IS NOT NULL AND model_instrument != path_instrument)
              THEN 2.0 ELSE 0.0 END
            ) ASC
          ) as rn
          FROM samples
          WHERE {where}
        )
        WHERE rn <= 100
        ORDER BY rn ASC
        """
        rows = con.execute(query).fetchall()

    ix = get_sim()
    ix.ensure(max_age=0)
    
    selected_rows = []
    selected_embs = []
    SIM_THRESH = 0.95
    
    for r in rows:
        p = r[0]
        if p in ix.idx:
            emb = ix.mat[ix.idx[p]]
            is_dup = False
            for prev_emb in selected_embs:
                if np.dot(prev_emb, emb) > SIM_THRESH:
                    is_dup = True
                    break
            if is_dup:
                continue
            selected_embs.append(emb)
        selected_rows.append(r)
        if len(selected_rows) >= limit:
            break

    items = [{"path": r[0], "path_instrument": r[1],
              "panns_instrument": r[2],
              "panns_conf": round(r[3], 3) if r[3] else None,
              "audio_instrument": r[4],
              "duration_s": r[5], "sample_type": r[6],
              "human_sample_type": r[7], "human_instrument": r[8],
              "model_instrument": r[9],
              "model_conf": round(r[10], 3) if r[10] else None,
              "rating": r[11] or 0}
             for r in selected_rows]

    # attach audio-only sonic descriptors (grain + coarse family)
    grain_ids = {r[12] for r in selected_rows if r[12] is not None}
    grain_lbl, fam_lbl = {}, []
    if grain_ids:
        with ro() as con:
            if con:
                fam_lbl = sonic_family_labels(con)
                try:
                    qs = ",".join("?" * len(grain_ids))
                    for gid, lab in con.execute(
                            f"SELECT id, label FROM sonic_clusters WHERE level=2 "
                            f"AND id IN ({qs})", list(grain_ids)):
                        grain_lbl[gid] = lab
                except sqlite3.OperationalError:
                    pass
    for it, r in zip(items, selected_rows):
        g = grain_lbl.get(r[12])
        f = fam_lbl[r[13]] if (r[13] is not None and 0 <= r[13] < len(fam_lbl)) else None
        it["sonic"] = {"grain": g, "family": f} if (g or f) else None
    return {"items": items, "total": total}


def recent_errors(limit=15):
    with ro() as con:
        if not con: return []
        rows = con.execute(
            "SELECT path, error FROM samples WHERE status='error' "
            "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(path=p, error=e) for p, e in rows]


from ..db import db_connect

def migrate_db():
    # Primary schema auto-migrated via db_connect
    con = db_connect(DB)
    con.close()
    
    # labels live in their own file — no contention with the scanner
    lcon = sqlite3.connect(LABELS_DB, timeout=10)
    try:
        lcon.execute("CREATE TABLE IF NOT EXISTS labels (name TEXT PRIMARY KEY, created_at REAL)")
        if lcon.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0:
            for inst in INSTRUMENTS:
                lcon.execute("INSERT OR IGNORE INTO labels(name,created_at) VALUES(?,?)",
                             (inst, time.time()))
        lcon.commit()
    finally:
        lcon.close()


def get_labels():
    with ro(LABELS_DB) as con:
        if not con: return []
        return [r[0] for r in con.execute("SELECT name FROM labels ORDER BY name").fetchall()]

def get_colors():
    labels = get_labels()
    out = {}
    for lbl in labels:
        if lbl in INSTR_COLORS:
            out[lbl] = INSTR_COLORS[lbl]
        else:
            h = (sum(ord(c) for c in lbl) * 137) % 360
            out[lbl] = f"hsl({h}, 60%, 55%)"
    return out


def add_label(name):
    name = name.strip().lower()
    if not name or len(name) > 40:
        return {"ok": False, "msg": "invalid name"}
    con = sqlite3.connect(LABELS_DB, timeout=10)
    try:
        con.execute("INSERT OR IGNORE INTO labels(name,created_at) VALUES(?,?)",
                    (name, time.time()))
        con.commit()
    finally:
        con.close()
    return {"ok": True, "name": name}


def delete_label(name):
    global _MAP, _CLUSTERS
    con = sqlite3.connect(LABELS_DB, timeout=10)
    try:
        con.execute("DELETE FROM labels WHERE name=?", (name,))
        con.commit()
    finally:
        con.close()
    # Keep applied labels in sync with the taxonomy: a removed class must not
    # linger in samples.human_instrument (it would be an un-trainable, un-colored
    # orphan). Clear it so the catalog stays consistent.
    cleared = 0
    scon = sqlite3.connect(DB, timeout=20)
    try:
        cur = scon.execute(
            "UPDATE samples SET human_instrument=NULL, label_source=NULL, ts=? "
            "WHERE human_instrument=?", (time.time(), name))
        cleared = cur.rowcount
        scon.commit()
    finally:
        scon.close()
    _MAP = None
    _CLUSTERS = None
    return {"ok": True, "cleared": max(cleared, 0)}

def init(db_path):
    global DB
    DB = db_path
    migrate_db()

def serve_audio(req, path):
    if not path or not valid_sample(path) or not os.path.isfile(path):
        req._send(404, "not found", "text/plain"); return
    ct = AUDIO_CT.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
    size = os.path.getsize(path)
    start, end, status = 0, size - 1, 200
    rng = req.headers.get("Range")
    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if m:
            if m.group(1):
                start = int(m.group(1))
            if m.group(2):
                end = int(m.group(2))
            end = min(end, size - 1); start = min(start, end); status = 206
    length = end - start + 1
    req.send_response(status)
    req.send_header("Content-Type", ct)
    req.send_header("Accept-Ranges", "bytes")
    req.send_header("Content-Length", str(length))
    if status == 206:
        req.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    req.end_headers()
    try:
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                req.wfile.write(chunk)
                remaining -= len(chunk)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
def serve_audio_normalized(req, path):
    if not path or not valid_sample(path) or not os.path.isfile(path):
        req._send(404, "not found", "text/plain"); return
    import subprocess
    # Pass 1: measure peak (fast, much faster than real-time)
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path,
         "-af", "volumedetect", "-vn", "-sn", "-dn", "-f", "null", "/dev/null"],
        capture_output=True, text=True)
    peak_db = 0.0
    for line in probe.stderr.splitlines():
        m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", line)
        if m:
            peak_db = float(m.group(1)); break
    gain_db = min(-1.0 - peak_db, 30.0)  # target -1 dBFS, cap boost at 30 dB
    # Pass 2: serve with exact gain correction
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", path,
           "-af", f"volume={gain_db:.2f}dB",
           "-ar", "44100", "-ac", "2",
           "-c:a", "libmp3lame", "-q:a", "4",
           "-f", "mp3", "pipe:1"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        req.serve_audio(path); return
    req.send_response(200)
    req.send_header("Content-Type", "audio/mpeg")
    req.send_header("Cache-Control", "no-store")
    req.end_headers()
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            req.wfile.write(chunk)
    except (BrokenPipeError, ConnectionResetError, OSError):
        proc.kill()
    finally:
        proc.wait()
def log_tail():
    try:
        with open(RUNLOG) as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-12:] if l.strip()]
    except OSError:
        return []

def _qs(req):
    import urllib.parse
    return urllib.parse.parse_qs(urllib.parse.urlparse(req.path).query)

GET_ROUTES = {
    "/api/labels": get_labels,
    "/api/colors": get_colors,
    "/api/review/queue": lambda req: review_queue((_qs(req).get("mode") or ["disagree"])[0]),
    "/api/config": load_config,
    "/api/run/status": run_status,
    "/api/run/ml/status": ml_run_status,
    "/api/stats": stats,
    "/api/log": log_tail,
    "/api/errors": recent_errors,
    "/api/similar": lambda req: similar_api((_qs(req).get("path") or _qs(req).get("q") or [""])[0], int((_qs(req).get("k") or ["24"])[0])),
    "/api/propagate": lambda req: propagate_candidates((_qs(req).get("path") or [""])[0], int((_qs(req).get("k") or ["24"])[0])),
    "/api/clusters": lambda req: clusters_list((_qs(req).get("mode") or ["value"])[0], int((_qs(req).get("limit") or ["300"])[0])),
    "/api/sonic/families": sonic_families,
    "/api/sonic/grains": lambda req: sonic_grains(int((_qs(req).get("family") or ["-1"])[0])),
    "/api/sonic/members": lambda req: sonic_members(int((_qs(req).get("grain") or ["-1"])[0])),
    "/api/cluster": lambda req: cluster_detail(int((_qs(req).get("id") or ["-1"])[0])),
    "/api/map": map_api,
    "/api/point": lambda req: point_api(int((_qs(req).get("i") or ["-1"])[0])),
    "/api/reproject": reproject_start,
    "/api/reproject_status": lambda req: _REPROJ,
}

POST_ROUTES = {
    "/api/config": lambda data: save_config(data),
    "/api/run/start": lambda data: {"ok": False, "msg": "use /api/run/discover or /api/run/label"},
    "/api/run/stop": lambda data: run_stop(),
    "/api/run/discover": lambda data: run_start("discover"),
    "/api/run/label": lambda data: run_start("label"),
    "/api/run/ml": lambda data: ml_run_start(),
    "/api/run/ml/stop": lambda data: ml_run_stop(),
    "/api/label": lambda data: label_api(data.get("path",""), data.get("instrument","")),
    "/api/label_type": lambda data: label_type_api(data.get("path",""), data.get("sample_type","")),
    "/api/rate": lambda data: rate_api(data.get("path",""), data.get("rating", 0)),
    "/api/label_propagate": lambda data: label_propagate(data.get("paths", []), data.get("instrument","")),
    "/api/label_cluster": lambda data: label_cluster(int(data.get("cluster_id", -1)), data.get("instrument",""), data.get("exclude", [])),
    "/api/labels/add": lambda data: add_label(data.get("name","")),
    "/api/labels/delete": lambda data: delete_label(data.get("name","")),
    "/api/label_map": lambda data: label_map(data),
}

def _read_body(req):
    try:
        length = int(req.headers.get("Content-Length", 0))
        if length > 0:
            return req.rfile.read(length).decode("utf-8")
    except (ValueError, OSError):   # bad Content-Length / decode / socket read
        pass
    return ""

def handle_post(req, route):
    body = _read_body(req)
    if route in POST_ROUTES:
        try:
            data = json.loads(body) if body else {}
            _json(req, POST_ROUTES[route](data))
        except Exception as e:
            req._send(400, str(e), "text/plain")
    else:
        req._send(404, "not found", "text/plain")

def handle_get(req, route):
    if route == "/api/audio":
        qs = _qs(req)
        path = (qs.get("path") or [""])[0]
        if (qs.get("norm") or [""])[0] == "1":
            serve_audio_normalized(req, path)
        else:
            serve_audio(req, path)
        return

    if route in GET_ROUTES:
        try:
            handler = GET_ROUTES[route]
            # checking __code__ fails on builtins or lambdas sometimes, simpler to just inspect
            # or we can pass req to all lambdas, and non-req handlers are just direct functions
            if route in ["/api/review/queue", "/api/similar", "/api/propagate", "/api/clusters", "/api/cluster", "/api/point", "/api/reproject_status", "/api/sonic/grains", "/api/sonic/members"]:
                _json(req, handler(req))
            else:
                _json(req, handler())
        except Exception as e:
            _json(req, {"ready": False, "msg": str(e)}, code=500)
    else:
        req._send(404, "not found", "text/plain")
