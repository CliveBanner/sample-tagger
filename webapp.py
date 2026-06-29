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

import simlib

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "samples.db")
LABELS_DB = os.path.join(HERE, "labels.db")
RUNLOG = os.path.join(HERE, "run.log")
CONFIG_FILE = os.path.join(HERE, "config.json")
START = time.time()

# Use venv Python if available so the scan subprocess gets all deps
PYTHON = os.path.join(HERE, "venv", "bin", "python")
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable

DEFAULT_CONFIG = {
    "library_path": "/home/phlp/pcloud/DAW/Samples",
    "workers": 5,
    # discover stage
    "trust_db": True,
    "no_cache": False,
    # label stage
    "label_path": False,
    "label_audio": False,
    "label_panns": True,
    "gpu_python": "",    # path to GPU-capable python (e.g. venv_gpu/bin/python); empty = use default
    "redo": "",          # comma-separated classifiers to overwrite, or "all"
    "limit": 0,
    # analysis tunables
    "analyze_seconds": 30.0,
    "loop_min_sec": 0.8,
    "loop_bar_tolerance": 0.12,
    "harmonic_ratio_tonal": 0.35,
    "bpm_min": 60,
    "bpm_max": 200,
    "panns_min_duration": 1.0,
    # projection
    "proj_method": "auto",
    "proj_n_neighbors": 25,
    "proj_min_dist": 0.12,
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_FILE) as f:
            cfg.update(json.load(f))
    except (OSError, json.JSONDecodeError):
        pass
    return cfg


def save_config(data):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({k: v for k, v in data.items() if k in DEFAULT_CONFIG})
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg

# instrument -> color, shared by the map legend and points
INSTR_COLORS = {
    "kick": "#f92672", "808": "#fd1f6a", "snare": "#fd971f", "tom": "#e6a23c",
    "hihat": "#e6db74", "clap": "#ae81ff", "cymbal": "#dcd45a", "perc": "#a6e22e",
    "drums": "#7ec92e", "bass": "#66d9ef", "synth": "#9a6cff", "tonal": "#4fc3e8",
    "vocal": "#ff5fa2", "fx": "#75715e",
}

AUDIO_CT = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
            ".aif": "audio/aiff", ".aiff": "audio/aiff", ".ogg": "audio/ogg"}

_SIM = None            # lazily-built SimIndex
_SIM_LAST_USE = 0.0    # timestamp of last /api/similar call
_SIM_TTL = 600         # evict SimIndex after 10 min idle (saves ~1-2 GB)
_MAP = None            # cached map payload (dict)
_RUN_PROC = None       # Popen handle for a scan started via /api/run/start


# ---- run management -------------------------------------------------------

def _tagger_pid():
    """Return PID of any live sample_tagger.py process found in /proc."""
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    if b"sample_tagger.py" in f.read():
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
    cmd = [py, os.path.join(HERE, "sample_tagger.py"), cfg["library_path"]]
    cmd += ["--db", DB, "--stage", stage, "-j", str(cfg.get("workers", 5))]

    if stage == "discover":
        if cfg.get("trust_db"): cmd += ["--trust-db"]
        if cfg.get("no_cache"): cmd += ["--no-cache"]

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

    if cfg.get("limit"): cmd += ["--limit", str(int(cfg["limit"]))]
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


def valid_sample(path):
    """Only serve files that are actually in the catalog (no path traversal)."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return con.execute("SELECT 1 FROM samples WHERE path=? LIMIT 1",
                           (path,)).fetchone() is not None
    finally:
        con.close()


INSTRUMENTS = [
    "kick", "snare", "hihat", "tom", "cymbal", "clap", "perc", "drums",
    "bass", "synth", "tonal", "vocal", "fx",
]


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
    global _MAP
    if not path:
        return {"ok": False, "msg": "no path"}
    if instrument and instrument not in get_labels():
        return {"ok": False, "msg": f"unknown instrument: {instrument}"}
    con = sqlite3.connect(DB, timeout=10)
    try:
        con.execute(
            "UPDATE samples SET human_instrument=?, ts=? WHERE path=?",
            (instrument or None, time.time(), path))
        con.commit()
    finally:
        con.close()
    _MAP = None
    return {"ok": True, "path": path, "instrument": instrument}


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


def build_map():
    """Join projection + samples into compact parallel arrays for the canvas.

    Cache is keyed by the sidecar file's mtime — the map only rebuilds when
    project.py actually writes a new projection, so all clients share one copy.
    """
    global _MAP
    proj_db = DB + ".proj"
    try:
        sidecar_mtime = os.path.getmtime(proj_db)
    except OSError:
        sidecar_mtime = 0
    if (_MAP is not None
            and sidecar_mtime > 0
            and _MAP.get("_sidecar_mtime") == sidecar_mtime):
        return _MAP
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = []
    try:
        if sidecar_mtime:              # sidecar exists — use it
            con.execute("ATTACH DATABASE ? AS pj", (f"file:{proj_db}?mode=ro",))
            rows = con.execute(
                "SELECT p.path, p.x, p.y, s.instrument, s.sample_type, s.duration_s, s.source "
                "FROM pj.projection p JOIN samples s ON s.path=p.path").fetchall()
        else:                          # legacy: projection table inside samples.db
            rows = con.execute(
                "SELECT p.path, p.x, p.y, s.instrument, s.sample_type, s.duration_s, s.source "
                "FROM projection p JOIN samples s ON s.path=p.path").fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()
    cats = sorted(INSTR_COLORS.keys())
    cidx = {c: i for i, c in enumerate(cats)}
    tcode = {"oneshot": 0, "loop": 1}
    scode = {"path": 0, "panns": 1, "audio": 2, "human": 4}
    xs, ys, cs, ts, ds, ss, paths = [], [], [], [], [], [], []
    for path, x, y, instr, st, dur, src in rows:
        xs.append(round(x, 4)); ys.append(round(y, 4))
        cs.append(cidx.get(instr, len(cats)))
        ts.append(tcode.get(st, 2))
        ds.append(round(dur, 2) if dur else 0)
        ss.append(scode.get(src, 3))
        paths.append(path)
    _MAP = {"_sidecar_mtime": sidecar_mtime if paths else 0,
            "paths": paths, "x": xs, "y": ys, "c": cs, "t": ts, "d": ds, "s": ss,
            "cats": cats, "colors": [INSTR_COLORS[c] for c in cats],
            "n": len(paths)}
    return _MAP


def map_api():
    m = build_map()
    return {k: m[k] for k in ("x", "y", "c", "t", "d", "s", "cats", "colors", "n")}


def point_api(i):
    m = build_map()
    if i < 0 or i >= len(m["paths"]):
        return {}
    path = m["paths"][i]
    meta = simlib.fetch_meta(DB, [path]).get(path, {})
    return dict(path=path, name=os.path.basename(path), **meta)


# --- on-demand map regeneration (runs project.py as a subprocess) ------------
_REPROJ = {"running": False, "ok": None, "msg": "idle", "ts": 0}


def _do_reproject():
    global _MAP
    proj = os.path.join(HERE, "project.py")
    try:
        r = subprocess.run([sys.executable, proj, "--db", DB],
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
                    if b"sample_tagger.py" in f.read():
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
    if not os.path.exists(DB):
        return {"ready": False, "msg": "samples.db not found yet"}
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    try:
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
                   SUM(CASE WHEN source='human' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN path_instrument IS NOT NULL
                             OR panns_instrument IS NOT NULL
                             OR audio_instrument IS NOT NULL THEN 1 ELSE 0 END)
            FROM samples WHERE status != 'missing'""")[0]
        panns_min_dur = load_config().get("panns_min_duration", 1.0)
        panns_skipped = q(con, "SELECT COUNT(*) FROM samples "
                               "WHERE status != 'missing' AND duration_s < ?",
                          (panns_min_dur,))[0][0]
        coverage = {
            "active": active,
            "path":  {"n": cov_row[0], "total": active},
            "panns": {"n": cov_row[1], "total": active, "skipped": panns_skipped},
            "audio": {"n": cov_row[2], "total": active},
            "human": {"n": cov_row[3] or 0, "total": active},
            "any":   {"n": cov_row[4] or 0, "total": active},
        }

        def inst_dist(col):
            rows = con.execute(
                f"SELECT {col}, COUNT(*) n FROM samples "
                f"WHERE status != 'missing' AND {col} IS NOT NULL "
                f"GROUP BY {col} ORDER BY n DESC LIMIT 14").fetchall()
            return [{"label": r[0], "n": r[1]} for r in rows]

        path_dist  = inst_dist("path_instrument")
        panns_dist = inst_dist("panns_instrument")
        audio_dist = inst_dist("audio_instrument")
        sample_type = inst_dist("sample_type")
        keys = inst_dist("key")

        bpm_rows = q(con, "SELECT (bpm/10)*10 AS bucket, COUNT(*) FROM samples "
                          "WHERE bpm IS NOT NULL GROUP BY bucket ORDER BY bucket")
        bpm_hist = [dict(label=f"{int(b)}–{int(b)+9}", n=n) for b, n in bpm_rows]

        try:
            with open(RUNLOG) as f:
                log_tail = [l.rstrip() for l in f.readlines()[-12:] if l.strip()]
        except OSError:
            log_tail = []
        return {
            "ready": True,
            "scan_running": scan_running(),
            "active": active, "processed": processed, "errors": errors,
            "rate": round(rate, 2), "eta_min": round(eta_min, 1) if eta_min is not None else None,
            "pct": round(100 * prog.get("done", 0) / total, 1) if total else None,
            "label_done": prog.get("done", 0), "label_total": total,
            "coverage": coverage,
            "path_dist": path_dist, "panns_dist": panns_dist,
            "audio_dist": audio_dist, "sample_type": sample_type,
            "keys": keys, "bpm_hist": bpm_hist, "log_tail": log_tail,
        }
    finally:
        con.close()


def review_queue(mode="disagree", limit=80):
    if not os.path.exists(DB):
        return {"items": [], "total": 0}
    if mode == "disagree":
        where = """status != 'missing' AND source != 'human'
            AND (
              (path_instrument  IS NOT NULL AND panns_instrument IS NOT NULL AND path_instrument  != panns_instrument)
           OR (path_instrument  IS NOT NULL AND audio_instrument IS NOT NULL AND path_instrument  != audio_instrument)
           OR (panns_instrument IS NOT NULL AND audio_instrument IS NOT NULL AND panns_instrument != audio_instrument)
            )"""
    else:
        where = """status != 'missing' AND source != 'human'
            AND (path_instrument IS NOT NULL OR panns_instrument IS NOT NULL OR audio_instrument IS NOT NULL)"""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    try:
        total = con.execute(f"SELECT COUNT(*) FROM samples WHERE {where}").fetchone()[0]
        rows  = con.execute(
            f"SELECT path, path_instrument, panns_instrument, panns_conf, "
            f"audio_instrument, duration_s, sample_type, human_sample_type, human_instrument "
            f"FROM samples WHERE {where} ORDER BY RANDOM() LIMIT ?", (limit,)).fetchall()
    finally:
        con.close()
    items = [{"path": r[0], "path_instrument": r[1],
              "panns_instrument": r[2],
              "panns_conf": round(r[3], 3) if r[3] else None,
              "audio_instrument": r[4],
              "duration_s": r[5], "sample_type": r[6],
              "human_sample_type": r[7], "human_instrument": r[8]}
             for r in rows]
    return {"items": items, "total": total}


def recent_errors(limit=15):
    if not os.path.exists(DB):
        return []
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    try:
        rows = con.execute(
            "SELECT path, error FROM samples WHERE status='error' "
            "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(path=p, error=e) for p, e in rows]
    finally:
        con.close()


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Sample Tagger</title>
<style>
:root{--bg:#272822;--fg:#f8f8f2;--dim:#75715e;--card:#1e1f1c;--accent:#a6e22e;
--blue:#66d9ef;--pink:#f92672;--orange:#fd971f;--purple:#ae81ff;--yellow:#e6db74}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:16px 20px;border-bottom:1px solid #3e3d32;display:flex;
align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:18px;margin:0;color:var(--accent)}
.badge{padding:2px 10px;border-radius:10px;font-size:12px;font-weight:bold}
.run{background:var(--accent);color:#000}.done{background:var(--blue);color:#000}
.muted{color:var(--dim)}
main{padding:20px;display:grid;gap:18px;
grid-template-columns:repeat(auto-fit,minmax(320px,1fr));max-width:1400px}
.card{background:var(--card);border:1px solid #3e3d32;border-radius:10px;padding:16px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;
color:var(--blue);margin:0 0 12px}
.kpis{display:flex;gap:24px;flex-wrap:wrap}
.kpi .v{font-size:26px;color:var(--fg)}.kpi .l{font-size:11px;color:var(--dim)}
.prog{height:14px;background:#3e3d32;border-radius:7px;overflow:hidden;margin:10px 0}
.prog>div{height:100%;background:linear-gradient(90deg,var(--accent),var(--blue))}
.bar{display:flex;align-items:center;gap:8px;margin:3px 0}
.bar .name{width:88px;color:var(--fg);white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.bar .track{flex:1;background:#3e3d32;border-radius:4px;height:16px;position:relative}
.bar .fill{display:block;height:100%;border-radius:4px;background:var(--purple)}
.bar .num{width:64px;text-align:right;color:var(--dim);font-size:12px}
.span2{grid-column:1/-1}
table{width:100%;border-collapse:collapse;font-size:12px}
td{padding:3px 6px;border-bottom:1px solid #2d2e28;vertical-align:top}
td.err{color:var(--pink)}td.pth{color:var(--dim);word-break:break-all}
footer{padding:12px 20px;color:var(--dim);font-size:12px}
</style></head><body>
<header><h1>🎛 Sample Tagger</h1><span id=status></span>
<span class=muted id=clock></span>
<button onclick="tick()" style="margin-left:auto" title="Refresh stats now">↻ Refresh</button>
<a href="/review" style="color:#e6db74;text-decoration:none;font-weight:bold">✎ Review</a>
<a href="/map" style="color:#66d9ef;text-decoration:none;font-weight:bold">🗺 Map ▸</a>
<a href="/settings" style="color:#ae81ff;text-decoration:none;font-weight:bold">⚙ Settings</a></header>
<main id=app></main>
<div style="padding:0 20px 20px;max-width:1400px">
<div class=card style="grid-column:1/-1">
<h2>Scan log</h2>
<pre id=logbox style="margin:0;font-size:12px;color:#a6e22e;white-space:pre-wrap;word-break:break-all">(loading…)</pre>
</div></div>
<footer>auto-refreshes every 4s · reads samples.db live</footer>
<script>
const COLORS={kick:'#f92672',snare:'#fd971f',hihat:'#e6db74',clap:'#ae81ff',
tom:'#fd971f',cymbal:'#e6db74',perc:'#a6e22e','808':'#f92672',bass:'#66d9ef',
synth:'#ae81ff',tonal:'#66d9ef',vocal:'#f92672',fx:'#75715e',drums:'#a6e22e',
loop:'#66d9ef',oneshot:'#a6e22e',percussive:'#fd971f'};
const DIM='#3a3a35';

function autoColor(i, n){ return `hsl(${Math.round(i*360/n)},60%,55%)`; }
function pie(items, sz=140){
  const total=items.reduce((s,i)=>s+i.n,0);
  if(!total)return '<span class=muted>no data yet</span>';
  const cx=sz/2,cy=sz/2,r=sz/2-2;
  let a=-Math.PI/2,paths='',leg='';
  const n=items.filter(i=>i.n).length;
  let ci=0;
  for(const it of items){
    if(!it.n)continue;
    const c=it.color||(COLORS[it.label])||autoColor(ci++,n);
    const sw=it.n/total*2*Math.PI;
    const pct=Math.round(it.n/total*100);
    if(sw>=2*Math.PI-0.001){
      paths+=`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${c}"/>`;
    } else {
      const x1=cx+r*Math.cos(a),y1=cy+r*Math.sin(a);
      a+=sw;
      const x2=cx+r*Math.cos(a),y2=cy+r*Math.sin(a);
      paths+=`<path d="M${cx},${cy}L${x1.toFixed(1)},${y1.toFixed(1)}A${r},${r},0,${sw>Math.PI?1:0},1,${x2.toFixed(1)},${y2.toFixed(1)}Z" fill="${c}"><title>${it.label}: ${it.n.toLocaleString()} (${pct}%)</title></path>`;
    }
    leg+=`<div style="display:flex;align-items:center;gap:5px;margin:2px 0">
      <span style="display:inline-block;width:9px;height:9px;background:${c};border-radius:2px;flex-shrink:0"></span>
      <span style="color:var(--fg)">${it.label}</span>
      <span class=muted style="margin-left:auto;padding-left:8px">${pct}%</span>
    </div>`;
  }
  return `<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
    <svg width="${sz}" height="${sz}" style="flex-shrink:0">${paths}</svg>
    <div style="font-size:12px;flex:1;min-width:80px">${leg}</div>
  </div>`;
}

function classifierCard(title, cov, dist, hitColor, missLabel){
  const n=cov?cov.n:0, tot=cov?cov.total:0, skipped=cov?cov.skipped||0:0;
  const pct=tot?Math.round(n/tot*100):0;
  const covItems=[
    {label:'labeled', n, color:hitColor},
    ...(skipped?[{label:'too short', n:skipped, color:'#fd971f'}]:[]),
    {label:missLabel, n:tot-n-skipped, color:DIM}
  ];
  const header=`<div style="display:flex;align-items:baseline;gap:10px;
    padding-bottom:10px;margin-bottom:14px;border-bottom:1px solid #3e3d32">
    <span style="font-size:28px;font-weight:bold;color:${hitColor}">${pct}%</span>
    <span class=muted style="font-size:12px">${n.toLocaleString()} / ${tot.toLocaleString()}</span>
  </div>`;
  const body=`<div style="display:flex;gap:16px;flex-wrap:wrap">
    <div style="flex:0 0 auto">${pie(covItems,100)}</div>
    <div style="flex:1;min-width:120px;border-left:1px solid #3e3d32;padding-left:16px">${pie(dist,120)}</div>
  </div>`;
  return card(title, header+body);
}

function bars(items){
  if(!items||!items.length)return '<span class=muted>no data yet</span>';
  const max=Math.max(...items.map(i=>i.n));
  return items.map(i=>{const c=COLORS[i.label]||'#ae81ff';
    return `<div class=bar><span class=name title="${i.label}">${i.label}</span>
    <span class=track><span class=fill style="width:${100*i.n/max}%;background:${c}"></span></span>
    <span class=num>${i.n.toLocaleString()}</span></div>`}).join('');
}
function card(title,inner,span){return `<div class="card${span?' span2':''}">
  <h2>${title}</h2>${inner}</div>`}

let _lastTick=0;
function updateClock(){
  if(!_lastTick)return;
  const age=Math.round((Date.now()-_lastTick)/1000);
  document.getElementById('clock').textContent='updated '+(age===0?'just now':age+'s ago');
}
setInterval(updateClock,1000);

async function tick(){
  let s; try{s=await (await fetch('/api/stats')).json()}catch(e){return}
  _lastTick=Date.now(); updateClock();
  const st=document.getElementById('status');
  if(!s.ready){st.innerHTML='<span class="badge done">waiting for db</span>';
    document.getElementById('app').innerHTML=card('Status',s.msg||'');return}
  st.innerHTML = s.scan_running
    ? '<span class="badge run">SCANNING</span>'
    : '<span class="badge done">IDLE / DONE</span>';
  const eta=s.eta_min!=null?(s.eta_min>60?(s.eta_min/60).toFixed(1)+' h':s.eta_min+' min'):'—';
  const anyN=(s.coverage&&s.coverage.any)?s.coverage.any.n:0;
  const kpis=`<div class=kpis>
    <div class=kpi><div class=v>${(s.active||0).toLocaleString()}</div><div class=l>total files</div></div>
    <div class=kpi><div class=v>${anyN.toLocaleString()}</div><div class=l>any label</div></div>
    <div class=kpi><div class=v>${(s.coverage&&s.coverage.human)?s.coverage.human.n:0}</div><div class=l>human corrections</div></div>
    <div class=kpi><div class=v>${(s.errors||0).toLocaleString()}</div><div class=l>errors</div></div>
    </div>
    ${s.label_total?`<div style="margin-top:12px">
      <div class=prog><div style="width:${s.pct||0}%"></div></div>
      <div class=muted style="font-size:12px;margin-top:4px">label stage: ${(s.label_done||0).toLocaleString()} / ${s.label_total.toLocaleString()} · ${s.rate} f/s · eta ${eta}</div>
    </div>`:''}`;
  const app=document.getElementById('app');
  const cov=s.coverage||{};
  app.innerHTML=
    card('Progress',kpis,true)+
    classifierCard('PANNs', cov.panns, s.panns_dist||[], '#ae81ff', 'pending')+
    classifierCard('Audio', cov.audio, s.audio_dist||[], '#66d9ef', 'pending')+
    classifierCard('Path',  cov.path,  s.path_dist||[],  '#a6e22e', 'no path hint')+
    card('Sample type',pie(s.sample_type||[]))+
    card('Key distribution',bars(s.keys||[]))+
    card('BPM (loops)',bars(s.bpm_hist));
  const lb=document.getElementById('logbox');
  if(lb)lb.textContent=(s.log_tail||[]).join('\\n')||'(no log yet)';
}
tick();setInterval(tick,4000);
</script></body></html>"""


REVIEW_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Review Queue</title>
<style>
:root{--bg:#272822;--fg:#f8f8f2;--dim:#75715e;--card:#1e1f1c;--accent:#a6e22e;
--blue:#66d9ef;--pink:#f92672;--orange:#fd971f;--purple:#ae81ff;--yellow:#e6db74}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--fg);font:13px/1.4 ui-monospace,Menlo,Consolas,monospace;
display:flex;flex-direction:column;overflow:hidden}
header{padding:8px 14px;border-bottom:1px solid #3e3d32;display:flex;gap:8px;
align-items:center;flex-wrap:wrap;flex-shrink:0}
header h1{font-size:15px;margin:0;color:var(--accent)}
a{color:var(--blue);text-decoration:none}
button,select{background:#3e3d32;border:1px solid #5a594a;color:var(--fg);
padding:4px 10px;border-radius:5px;font:inherit;cursor:pointer}
button:hover{background:#4a4940}
#wrap{flex:1;display:flex;min-height:0}
#list{width:260px;flex-shrink:0;border-right:1px solid #3e3d32;overflow-y:auto}
#detail{flex:1;padding:18px 22px;overflow-y:auto;min-width:0}
.item{padding:8px 10px;border-bottom:1px solid #2a2b27;cursor:pointer}
.item:hover{background:#252622}.item.sel{background:#2d2e28}
.item.done{opacity:.45}
.iname{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px}
.pills{display:flex;flex-wrap:wrap;gap:3px}
.pill{font-size:10px;padding:1px 6px;border-radius:8px;color:#000;font-weight:bold}
.path{font-size:11px;color:var(--dim);word-break:break-all;margin-bottom:10px}
.fname{font-size:17px;font-weight:bold;margin-bottom:4px}
audio{width:100%;margin:10px 0}
table{border-collapse:collapse;margin:12px 0;font-size:13px}
td{padding:5px 12px 5px 0;vertical-align:middle}td:first-child{color:var(--dim);width:60px}
.conf{font-size:11px;color:var(--dim);margin-left:5px}
.disagree{color:var(--pink);font-size:10px;margin-left:6px}
.ibtns{display:grid;grid-template-columns:repeat(auto-fill,minmax(82px,1fr));gap:6px;margin:14px 0 10px}
.ibtn{font-size:13px;padding:6px 10px;min-height:42px;border-radius:5px;transition:opacity .1s}
.ibtn:hover{opacity:.8}.ibtn.active{font-weight:bold;box-shadow:0 0 0 2px #fff4}
.actions{display:flex;align-items:center;gap:10px;margin-top:6px}
.pos{color:var(--dim);font-size:12px;margin-left:auto}
.empty{color:var(--dim);padding:30px;text-align:center}
.kbhint{font-size:11px;color:var(--dim);margin-top:18px}
#count{color:var(--dim);font-size:12px}
#list-toggle{display:none}
#lmodal{display:none;position:fixed;inset:0;background:#0009;z-index:200;align-items:center;justify-content:center}
#lmodal.open{display:flex}
#lbox{background:var(--bg);border:1px solid #5a594a;border-radius:8px;padding:20px;
width:340px;max-width:92vw;max-height:80vh;display:flex;flex-direction:column;gap:10px}
#lbox h2{margin:0;font-size:14px;color:var(--accent)}
#llist{flex:1;overflow-y:auto;display:flex;flex-wrap:wrap;gap:6px;align-content:flex-start}
.ltag{display:flex;align-items:center;gap:4px;background:#3e3d32;border-radius:4px;
padding:3px 8px;font-size:12px}
.ltag button{background:none;border:none;color:var(--pink);cursor:pointer;padding:0 2px;font-size:13px;line-height:1}
#ladd{display:flex;gap:6px}
#ladd input{flex:1;background:#1e1f1c;border:1px solid #5a594a;color:var(--fg);
padding:5px 8px;border-radius:4px;font:inherit;font-size:13px}
#overlay{display:none;position:fixed;inset:0;background:#0009;z-index:100}
#overlay.open{display:block}
#overlay-panel{position:absolute;top:0;left:0;bottom:0;width:280px;max-width:88vw;
background:var(--bg);border-right:1px solid #3e3d32;overflow-y:auto;display:flex;flex-direction:column}
#overlay-hdr{font-size:12px;color:var(--dim);padding:8px 12px;border-bottom:1px solid #3e3d32;
display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
#overlay-list{flex:1;overflow-y:auto}

@media(max-width:680px){
  body{overflow:auto}
  #wrap{flex-direction:column;min-height:0;overflow:visible}
  #list{display:none}
  #list-toggle{display:inline-block}
  #detail{padding:12px 14px;overflow:visible}
  .fname{font-size:15px}
  .ibtns{grid-template-columns:repeat(auto-fill,minmax(90px,1fr));gap:8px;margin:12px 0 8px}
  .ibtn{font-size:14px;min-height:52px}
  .actions button{padding:10px 20px;font-size:15px}
  .actions .pos{font-size:13px}
  .kbhint{display:none}
  #count{display:none}
  audio{margin:8px 0}
}
#toast{position:fixed;bottom:28px;left:50%;transform:translateX(-50%) translateY(12px);
padding:10px 24px;border-radius:8px;font-size:16px;font-weight:bold;
background:#1e1f1c;border:2px solid;opacity:0;
transition:opacity .18s,transform .18s;pointer-events:none;z-index:300;white-space:nowrap}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
#audio-hint{font-size:12px;color:var(--orange);margin:3px 0 8px;min-height:16px}
audio.loading{opacity:.35;transition:opacity .25s}
</style></head><body>
<header>
  <h1>Review Queue</h1>
  <button id=list-toggle onclick="openOverlay()">☰ Queue</button>
  <select id=modesel onchange="loadQueue()">
    <option value=disagree>Classifiers disagree</option>
    <option value=all>All unlabeled</option>
  </select>
  <span id=count></span>
  <button onclick="loadQueue()">↻ New batch</button>
  <button onclick="openLModal()">⚙ Labels</button>
  <a href="/" style="margin-left:auto">← Dashboard</a>
  <a href="/map">Map</a>
</header>
<div id=overlay onclick="closeOverlay()">
  <div id=overlay-panel onclick="event.stopPropagation()">
    <div id=overlay-hdr><span id=overlay-count></span><button onclick="closeOverlay()">✕</button></div>
    <div id=overlay-list></div>
  </div>
</div>
<div id=lmodal onclick="closeLModal()">
  <div id=lbox onclick="event.stopPropagation()">
    <h2>Instrument labels</h2>
    <div id=llist></div>
    <div id=ladd>
      <input id=linput placeholder="new label…" maxlength=40
        onkeydown="if(event.key==='Enter')addLabelUI()">
      <button onclick="addLabelUI()">+ Add</button>
    </div>
    <button onclick="closeLModal()" style="align-self:flex-end">Done</button>
  </div>
</div>
<div id=wrap>
  <div id=list></div>
  <div id=detail><div class=empty>Loading…</div></div>
</div>
<div id=toast></div>
<script>
const COLORS={kick:'#f92672',snare:'#fd971f',hihat:'#e6db74',clap:'#ae81ff',
tom:'#fd971f',cymbal:'#e6db74',perc:'#a6e22e','808':'#f92672',bass:'#66d9ef',
synth:'#ae81ff',tonal:'#66d9ef',vocal:'#f92672',fx:'#75715e',drums:'#a6e22e'};

let INSTRUMENTS=[], queue=[], cur=-1, _toastTimer=null;

async function fetchLabels(){
  INSTRUMENTS=await fetch('/api/labels').then(r=>r.json());
  renderLModal();
  if(cur>=0)renderDetail(queue[cur],cur);
}

function renderLModal(){
  document.getElementById('llist').innerHTML=INSTRUMENTS.map(name=>`
    <div class=ltag><span>${name}</span>
    <button onclick="delLabelUI('${name}')" title="remove">✕</button></div>`).join('');
}

async function addLabelUI(){
  const inp=document.getElementById('linput');
  const name=inp.value.trim().toLowerCase();
  if(!name)return;
  const r=await fetch('/api/labels/add',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})}).then(r=>r.json());
  if(r.ok){inp.value='';await fetchLabels();}
  else alert(r.msg);
}

async function delLabelUI(name){
  await fetch('/api/labels/delete',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})});
  await fetchLabels();
}

function openLModal(){document.getElementById('lmodal').classList.add('open');}
function closeLModal(){document.getElementById('lmodal').classList.remove('open');}

function showToast(msg,color){
  const t=document.getElementById('toast');
  t.textContent=msg; t.style.color=color; t.style.borderColor=color;
  t.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer=setTimeout(()=>t.classList.remove('show'),1400);
}

function basename(p){return p.split('/').pop()}
function col(inst){return inst?(COLORS[inst]||'#ae81ff'):'#555'}

function pill(inst){
  if(!inst)return '';
  return `<span class=pill style="background:${col(inst)}">${inst}</span>`;
}

function disagreeing(it){
  const vals=[it.path_instrument,it.panns_instrument,it.audio_instrument].filter(Boolean);
  return new Set(vals).size > 1;
}

function listHTML(){
  if(!queue.length)return '<div class=empty>Queue empty.</div>';
  return queue.map((it,i)=>`
    <div class="item${it._done?' done':''}${i===cur?' sel':''}" onclick="select(${i})">
      <div class=iname>${basename(it.path)}</div>
      <div class=pills>
        ${pill(it.path_instrument)}${pill(it.panns_instrument)}${pill(it.audio_instrument)}
        ${it.human_instrument?`<span class=pill style="background:#a6e22e">✓ ${it.human_instrument}</span>`:''}
      </div>
    </div>`).join('');
}

function renderList(){
  const h=listHTML();
  document.getElementById('list').innerHTML=h;
  document.getElementById('overlay-list').innerHTML=h;
}

function renderDetail(it,i){
  const rows=[
    it.path_instrument?`<tr><td>path</td><td style="color:${col(it.path_instrument)}">${it.path_instrument}</td><td></td></tr>`:'',
    it.panns_instrument?`<tr><td>PANNs</td><td style="color:${col(it.panns_instrument)}">${it.panns_instrument}${it.panns_conf?`<span class=conf>${(it.panns_conf*100).toFixed(0)}%</span>`:''}</td><td>${disagreeing(it)?'<span class=disagree>⚡ disagrees</span>':''}</td></tr>`:'',
    it.audio_instrument?`<tr><td>audio</td><td style="color:${col(it.audio_instrument)}">${it.audio_instrument}</td><td></td></tr>`:'',
  ].join('');
  const btns=INSTRUMENTS.map(inst=>`<button class="ibtn${it.human_instrument===inst?' active':''}"
    style="border-color:${col(inst)};color:${col(inst)}"
    onclick="save('${inst}')">${inst}${it.human_instrument===inst?' ✓':''}</button>`).join('');
  const effType=it.human_sample_type||it.sample_type;
  const typeColor=t=>t==='loop'?'#66d9ef':t==='oneshot'?'#ae81ff':'#75715e';
  const dur=it.duration_s?it.duration_s.toFixed(2)+'s':'';
  const typeBtns=['oneshot','loop'].map(t=>`<button class="ibtn${effType===t?' active':''}"
    style="border-color:${typeColor(t)};color:${typeColor(t)};min-height:36px;font-size:12px"
    onclick="saveType('${t}')">${t}${it.human_sample_type===t?' ✓':''}</button>`).join('');
  document.getElementById('detail').innerHTML=`
    <div class=fname>${basename(it.path)}</div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
      ${effType?`<span class=pill style="background:${typeColor(effType)}">${effType}${it.human_sample_type?` ✓`:''}</span>`:'<span style="font-size:11px;color:var(--dim)">type unknown</span>'}
      ${dur?`<span style="font-size:11px;color:var(--dim)">${dur}</span>`:''}
    </div>
    <div class=path>${it.path}</div>
    <audio id=player controls preload=auto></audio>
    <div id=audio-hint>⏳ loading audio…</div>
    <div style="display:flex;gap:6px;margin:8px 0 12px">${typeBtns}</div>
    <table>${rows||'<tr><td colspan=3 style="color:var(--dim)">No classifier results yet.</td></tr>'}</table>
    <div class=ibtns>${btns}</div>
    <div class=actions>
      <button onclick="prev()">← Prev</button>
      <button onclick="next()">Skip →</button>
      <span class=pos>${i+1} / ${queue.length}</span>
    </div>
    <div class=kbhint>← → navigate · click label to save &amp; advance</div>`;
  const player=document.getElementById('player');
  player.classList.add('loading');
  const t0=Date.now();
  player.addEventListener('canplay',()=>{
    player.classList.remove('loading');
    const h=document.getElementById('audio-hint');
    if(h)h.textContent='';
  },{once:true});
  player.src='/api/audio?norm=1&path='+encodeURIComponent(it.path);
  player.play().catch(()=>{});
}

function select(i){
  cur=i;
  renderList();
  renderDetail(queue[i],i);
  closeOverlay();
  const el=document.getElementById('list').children[i];
  if(el)el.scrollIntoView({block:'nearest'});
}

function saveType(sample_type){
  const it=queue[cur];
  it.human_sample_type=sample_type;
  const typeColor=sample_type==='loop'?'#66d9ef':'#ae81ff';
  showToast('✓ '+sample_type, typeColor);
  fetch('/api/label_type',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:it.path,sample_type})});
  renderDetail(it,cur);
}

function save(instrument){
  const it=queue[cur];
  it._done=true; it.human_instrument=instrument;
  showToast('✓ '+instrument, col(instrument));
  renderList();
  next();
  fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:it.path,instrument})});
}

function next(){if(cur<queue.length-1)select(cur+1);}
function prev(){if(cur>0)select(cur-1);}
function openOverlay(){document.getElementById('overlay').classList.add('open');}
function closeOverlay(){document.getElementById('overlay').classList.remove('open');}

async function loadQueue(){
  document.getElementById('list').innerHTML='<div class=empty>Loading…</div>';
  document.getElementById('overlay-list').innerHTML='';
  document.getElementById('detail').innerHTML='<div class=empty>Select a sample.</div>';
  cur=-1; queue=[];
  const mode=document.getElementById('modesel').value;
  const d=await fetch('/api/review/queue?mode='+mode).then(r=>r.json());
  queue=d.items||[];
  const ctxt=`${queue.length} loaded / ${(d.total||0).toLocaleString()} total`;
  document.getElementById('count').textContent=ctxt;
  document.getElementById('overlay-count').textContent=ctxt;
  renderList();
  if(queue.length)select(0);
  else document.getElementById('detail').innerHTML='<div class=empty>Nothing to review in this mode.</div>';
}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.key==='ArrowRight')next();
  else if(e.key==='ArrowLeft')prev();
});

fetchLabels().then(()=>loadQueue());
</script></body></html>"""


MAP_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Sample Map</title>
<style>
:root{--bg:#272822;--fg:#f8f8f2;--dim:#75715e;--card:#1e1f1c;--blue:#66d9ef}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--fg);overflow:hidden;
font:13px/1.4 ui-monospace,Menlo,Consolas,monospace;display:flex;flex-direction:column}
header{padding:8px 14px;border-bottom:1px solid #3e3d32;display:flex;gap:10px;
align-items:center;flex-wrap:wrap;flex-shrink:0}
header h1{font-size:15px;margin:0;color:#a6e22e}
a{color:var(--blue);text-decoration:none}
#q{background:#1e1f1c;border:1px solid #3e3d32;color:var(--fg);padding:5px 9px;
border-radius:6px;font-family:inherit;width:200px;min-width:0;flex-shrink:1}
button{background:#3e3d32;border:1px solid #5a594a;color:var(--fg);padding:5px 11px;
border-radius:6px;font-family:inherit;cursor:pointer;white-space:nowrap}
button:hover{background:#4a4940}button:disabled{opacity:.6;cursor:default}
#wrap{flex:1;display:flex;min-height:0}
#stage{flex:1;position:relative;min-width:0}
canvas{display:block;width:100%;height:100%;cursor:grab;touch-action:none}
canvas:active{cursor:grabbing}
#side{width:320px;flex-shrink:0;border-left:1px solid #3e3d32;background:var(--card);
padding:12px;overflow:auto}
#side h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--blue);
margin:0 0 7px}
.pill{display:inline-block;padding:1px 8px;border-radius:9px;background:#3e3d32;
margin:2px 4px 2px 0;font-size:11px}
.hit{padding:5px 6px;border-bottom:1px solid #2d2e28;cursor:pointer}
.hit:hover,.hit:active{background:#2d2e28}.hit .s{color:var(--dim);float:right}
.legend{position:absolute;left:8px;bottom:8px;background:rgba(30,31,28,.92);
padding:7px 9px;border-radius:8px;max-width:70%}
.legend .lg{display:inline-block;margin:2px 7px 2px 0;font-size:11px;cursor:pointer;
user-select:none}
.legend .lg.off{opacity:.3;text-decoration:line-through}
.legend a{font-size:11px}.legend label{font-size:11px;margin-right:7px;cursor:pointer}
.dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:3px;
vertical-align:middle}
.muted{color:var(--dim)}
#hint{position:absolute;right:8px;top:8px;background:rgba(30,31,28,.85);
padding:5px 8px;border-radius:7px;font-size:11px;color:var(--dim);pointer-events:none}
@media(max-width:720px){
  #wrap{flex-direction:column}
  #stage{flex:3}
  #side{flex:2;width:100%;min-height:0;border-left:none;border-top:1px solid #3e3d32}
  #q{width:130px}
  header h1{font-size:13px}
  #hint{display:none}
  .legend{max-width:94%}
}
</style></head><body>
<header>
<h1>🗺 Sample Map</h1>
<a href="/">◂ Dashboard</a>
<a href="/settings" style="color:#ae81ff">⚙</a>
<input id=q placeholder="search → similar">
<button id=upd title="regenerate the 2D layout from all embeddings so far">↻ Update map</button>
<span class=muted id=count style="font-size:11px"></span>
</header>
<div id=wrap>
  <div id=stage>
    <canvas id=cv></canvas>
    <div id=hint>drag · scroll to zoom · click a point</div>
    <div class=legend id=legend></div>
  </div>
  <div id=side>
    <h2>Selected</h2><div id=sel class=muted>tap a point or search</div>
    <audio id=player controls preload=none style="width:100%;margin-top:8px"></audio>
    <audio id=xplayer preload=none style=display:none></audio>
    <label class=muted style="display:block;margin-top:3px;font-size:11px">
      <input type=checkbox id=auto checked> autoplay</label>
    <div id=labelRow style="display:none;margin-top:10px;display:flex;gap:6px;align-items:center">
      <select id=labelSel style="flex:1;background:#1e1f1c;border:1px solid #3e3d32;color:var(--fg);padding:4px 6px;border-radius:4px;font-family:inherit;font-size:12px">
        <option value="">— correct label —</option>
        <option>kick</option><option>snare</option><option>hihat</option>
        <option>tom</option><option>cymbal</option><option>clap</option>
        <option>perc</option><option>drums</option><option>bass</option>
        <option>synth</option><option>tonal</option><option>vocal</option>
        <option>fx</option>
      </select>
      <button id=btnLabel style="white-space:nowrap" disabled>✓ Save</button>
    </div>
    <div id=labelMsg style="font-size:11px;margin-top:3px;color:#a6e22e;min-height:14px"></div>
    <button id=btnSim style="margin-top:10px;width:100%" disabled>🔍 Find similar</button>
    <h2 style="margin-top:10px">Similar</h2><div id=hits class=muted>—</div>
  </div>
</div>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const dpr=window.devicePixelRatio||1;
let M=null,sel=-1,scale=1,tx=0,ty=0,base=1,actC=null,actT=null,actS=null,minD=0,maxD=0,stride=1;
function shown(i){
  if(!actC.has(M.c[i])||!actT.has(M.t?M.t[i]:0))return false;
  if(actS&&!actS.has(M.s?M.s[i]:0))return false;
  const d=M.d?M.d[i]:0;
  if(d>0){if(minD>0&&d<minD)return false;if(maxD>0&&d>maxD)return false;}
  return true;
}
function resize(){cv.width=cv.clientWidth*dpr;cv.height=cv.clientHeight*dpr;
  base=Math.min(cv.width,cv.height);draw();}
window.addEventListener('resize',resize);
function sx(i){return tx+M.x[i]*base*scale;}
function sy(i){return ty+(1-M.y[i])*base*scale;}
function fit(){scale=0.92;tx=(cv.width-base*scale)/2;ty=(cv.height-base*scale)/2;}
function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if(!M)return;
  const r=Math.max(1.2,1.6*Math.sqrt(scale))*dpr,W=cv.width,H=cv.height;
  // fraction of total map area currently visible; clamp to 1 when zoomed out
  const visF=Math.min(1,W/(base*scale))*Math.min(1,H/(base*scale));
  stride=Math.max(1,Math.ceil(M.n*visF/20000));
  for(let i=0;i<M.n;i+=stride){if(!shown(i))continue;const X=sx(i),Y=sy(i);
    if(X<-2||X>W+2||Y<-2||Y>H+2)continue;
    ctx.fillStyle=M.colors[M.c[i]]||'#888';ctx.fillRect(X-r/2,Y-r/2,r,r);}
  if(sel>=0){
    if(stride>1&&shown(sel)){const X=sx(sel),Y=sy(sel);
      ctx.fillStyle=M.colors[M.c[sel]]||'#888';ctx.fillRect(X-r/2,Y-r/2,r,r);}
    ctx.strokeStyle='#fff';ctx.lineWidth=2*dpr;ctx.beginPath();
    ctx.arc(sx(sel),sy(sel),8*dpr,0,7);ctx.stroke();}
}
// Convert client coords to canvas device-pixel coords (canvas-relative)
function cvPos(cx,cy){const r=cv.getBoundingClientRect();
  return [(cx-r.left)*dpr,(cy-r.top)*dpr];}
// --- mouse ---
cv.addEventListener('wheel',e=>{e.preventDefault();
  const f=e.deltaY<0?1.15:1/1.15,[mx,my]=cvPos(e.clientX,e.clientY);
  tx=mx-(mx-tx)*f;ty=my-(my-ty)*f;scale*=f;draw();},{passive:false});
let down=null,moved=false;
cv.addEventListener('mousedown',e=>{down=[...cvPos(e.clientX,e.clientY),tx,ty];moved=false;});
window.addEventListener('mousemove',e=>{if(!down)return;
  const[mx,my]=cvPos(e.clientX,e.clientY);
  if(Math.abs(mx-down[0])+Math.abs(my-down[1])>4)moved=true;
  tx=down[2]+(mx-down[0]);ty=down[3]+(my-down[1]);draw();});
window.addEventListener('mouseup',()=>{if(down&&!moved)pick(down[0],down[1]);down=null;});
// --- touch ---
let t0=null,t1=null,pd=0;
function tPos(t){return cvPos(t.clientX,t.clientY);}
cv.addEventListener('touchstart',e=>{e.preventDefault();
  if(e.touches.length===1){
    const[px,py]=tPos(e.touches[0]);
    down=[px,py,tx,ty];moved=false;t0=e.touches[0];t1=null;
  }else if(e.touches.length===2){
    down=null;t0=e.touches[0];t1=e.touches[1];
    pd=Math.hypot(t1.clientX-t0.clientX,t1.clientY-t0.clientY);
  }},{passive:false});
cv.addEventListener('touchmove',e=>{e.preventDefault();
  if(e.touches.length===1&&down){
    const[mx,my]=tPos(e.touches[0]);
    if(Math.abs(mx-down[0])+Math.abs(my-down[1])>4)moved=true;
    tx=down[2]+(mx-down[0]);ty=down[3]+(my-down[1]);draw();
  }else if(e.touches.length===2&&t0&&t1){
    const a=e.touches[0],b=e.touches[1];
    const nd=Math.hypot(b.clientX-a.clientX,b.clientY-a.clientY);
    const f=nd/pd;
    const[cx,cy]=cvPos((a.clientX+b.clientX)/2,(a.clientY+b.clientY)/2);
    tx=cx-(cx-tx)*f;ty=cy-(cy-ty)*f;scale*=f;pd=nd;t0=a;t1=b;draw();
  }},{passive:false});
cv.addEventListener('touchend',e=>{e.preventDefault();
  if(e.touches.length===0&&down&&!moved)pick(down[0],down[1]);
  if(e.touches.length<1){down=null;}
  if(e.touches.length<2){t0=null;t1=null;}},{passive:false});
function pick(px,py){let best=-1,bd=16*dpr*16*dpr;
  for(let i=0;i<M.n;i++){if(!shown(i))continue;
    const dx=sx(i)-px,dy=sy(i)-py,d=dx*dx+dy*dy;
    if(d<bd){bd=d;best=i;}}
  if(best>=0){sel=best;draw();inspect(best);}}
const FADE_MS=180;
let xfading=false;
function playPath(path){
  const a=document.getElementById('player');
  const x=document.getElementById('xplayer');
  const auto=document.getElementById('auto').checked;
  const url='/api/audio?path='+encodeURIComponent(path);
  // abort any in-progress crossfade
  if(xfading){x.pause();x.src='';x.load();x.volume=1;xfading=false;a.volume=1;}
  // no crossfade if nothing is playing
  if(a.paused||a.ended||!a.src){
    a.src=url;if(auto)a.play().catch(()=>{});return;}
  // crossfade: outgoing → x, incoming → a
  const vol=a.volume;
  x.src=a.src;x.currentTime=a.currentTime;x.volume=vol;x.play().catch(()=>{});
  a.volume=0;a.src=url;if(auto)a.play().catch(()=>{});
  xfading=true;
  const t0=performance.now();
  const tick=()=>{
    const p=Math.min(1,(performance.now()-t0)/FADE_MS);
    x.volume=vol*(1-p);a.volume=vol*p;
    if(p<1){requestAnimationFrame(tick);}
    else{x.pause();x.src='';x.load();x.volume=1;xfading=false;}};
  requestAnimationFrame(tick);
}
let _selPath=null;
async function inspect(i){const p=await(await fetch('/api/point?i='+i)).json();
  _selPath=p.path;showSel(p);playPath(p.path);
  document.getElementById('btnSim').disabled=false;
  document.getElementById('hits').innerHTML='<span class=muted>—</span>';}
document.getElementById('btnSim').onclick=()=>{
  if(_selPath)loadSimilar('path='+encodeURIComponent(_selPath));};
document.getElementById('btnLabel').onclick=async()=>{
  const instr=document.getElementById('labelSel').value;
  if(!_selPath||!instr)return;
  const btn=document.getElementById('btnLabel');
  btn.disabled=true;
  const r=await fetch('/api/label',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:_selPath,instrument:instr})});
  const d=await r.json();
  const msg=document.getElementById('labelMsg');
  if(d.ok){
    msg.style.color='#a6e22e';msg.textContent='Saved ✓';
    // update the pill in the sel panel immediately
    const pill=document.querySelector('#sel .pill');
    if(pill)pill.textContent=instr;
    // update map dot color
    if(sel>=0){const cats=M.cats;const ci=cats.indexOf(instr);if(ci>=0)M.c[sel]=ci;draw();}
  } else {
    msg.style.color='#f92672';msg.textContent=d.msg||'error';}
  btn.disabled=false;};
const SRC_LABEL={'path':'via path','panns':'via PANNs','audio':'via audio','none':'unknown','human':'✎ human'};
const SRC_COLOR={'human':'#f6d860'};
function showSel(p){
  const el=document.getElementById('sel');el.classList.remove('muted');
  const srcCol=SRC_COLOR[p.source]||'var(--dim)';
  // build classification breakdown rows
  const conf=p.panns_conf?` <span style="color:var(--dim);font-size:10px">${(p.panns_conf*100).toFixed(0)}%</span>`:'';
  const rawConf=p.panns_label_conf?` <span style="color:var(--dim);font-size:10px">${(p.panns_label_conf*100).toFixed(0)}%</span>`:'';
  const rawTip=(p.panns_topk||[]).map(t=>`${t[0]} ${(t[1]*100).toFixed(0)}%`).join(' · ');
  const rows=[
    p.path_instrument  ?`<tr><td style="color:var(--dim);padding-right:8px">path</td><td>${p.path_instrument}</td></tr>`:'',
    p.panns_instrument ?`<tr><td style="color:var(--dim);padding-right:8px">PANNs</td><td>${p.panns_instrument}${conf}</td></tr>`:'',
    p.audio_instrument ?`<tr><td style="color:var(--dim);padding-right:8px">audio</td><td>${p.audio_instrument}</td></tr>`:'',
    p.panns_label      ?`<tr><td style="color:var(--dim);padding-right:8px">raw</td><td title="${rawTip}">${p.panns_label}${rawConf}</td></tr>`:'',
  ].join('');
  const srcCol2=SRC_COLOR[p.source]||'var(--dim)';
  el.innerHTML=`<div style="word-break:break-all;font-weight:500">${p.name||'?'}</div>
  <div style="margin-top:5px">
  <span class=pill style="font-weight:600">${p.instrument||'?'}</span><span class=pill>${p.sample_type||''}</span>
  ${p.bpm?`<span class=pill>${p.bpm} bpm</span>`:''}
  ${p.key?`<span class=pill>${p.key}</span>`:''}
  ${p.duration_s?`<span class=pill>${p.duration_s}s</span>`:''}
  <span class=pill style="color:${srcCol2}">${SRC_LABEL[p.source]||p.source||''}</span></div>
  ${rows?`<table style="margin-top:6px;font-size:11px;border-collapse:collapse">${rows}</table>`:''}`
  // pre-fill the label dropdown with the current instrument
  const lsel=document.getElementById('labelSel');
  lsel.value=p.instrument||'';
  document.getElementById('labelRow').style.display='flex';
  document.getElementById('labelMsg').textContent='';
  document.getElementById('btnLabel').disabled=false;}
async function loadSimilar(qs){
  const h=document.getElementById('hits');h.innerHTML='<span class=muted>…</span>';
  const d=await(await fetch('/api/similar?k=24&'+qs)).json();
  if(!d.matched){h.innerHTML='<span class=muted>no match ('+d.n+' indexed)</span>';return;}
  h.classList.remove('muted');
  h.innerHTML=d.hits.map(x=>`<div class=hit data-p="${encodeURIComponent(x.path)}">
    <span class=s>${x.score}</span>${x.name}
    <div class=muted>${x.instrument||''} ${x.sample_type||''} ${x.bpm?x.bpm+'bpm':''} ${x.key||''}</div>
    </div>`).join('');
  h.querySelectorAll('.hit').forEach(el=>el.onclick=()=>{
    const path=decodeURIComponent(el.dataset.p);
    _selPath=path;showSelName(path);playPath(path);loadSimilar('path='+el.dataset.p);});}
function showSelName(path){const el=document.getElementById('sel');
  el.classList.remove('muted');
  el.innerHTML='<div style="word-break:break-all">'+path.split('/').pop()+'</div>';}
document.getElementById('q').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&e.target.value.trim())
    loadSimilar('q='+encodeURIComponent(e.target.value.trim()));});
function legend(){
  const el=document.getElementById('legend');
  let h='<div style="margin-bottom:4px"><b>instrument</b> '+
    '<a href=# id=lall>all</a> · <a href=# id=lnone>none</a></div>';
  h+=M.cats.map((c,i)=>`<span class="lg${actC.has(i)?'':' off'}" data-i="${i}">`+
    `<span class=dot style="background:${M.colors[i]}"></span>${c}</span>`).join('');
  h+='<div style="margin-top:5px"><b>type</b> '+
    `<label><input type=checkbox class=tt data-t=0 ${actT.has(0)?'checked':''}>oneshot</label>`+
    `<label><input type=checkbox class=tt data-t=1 ${actT.has(1)?'checked':''}>loop</label>`+
    `<label><input type=checkbox class=tt data-t=2 ${actT.has(2)?'checked':''}>other</label></div>`;
  h+='<div style="margin-top:5px"><b>source</b> '+
    `<label><input type=checkbox class=ss data-s=0 ${actS.has(0)?'checked':''}>path</label>`+
    `<label><input type=checkbox class=ss data-s=1 ${actS.has(1)?'checked':''}>PANNs</label>`+
    `<label><input type=checkbox class=ss data-s=2 ${actS.has(2)?'checked':''}>audio</label>`+
    `<label><input type=checkbox class=ss data-s=3 ${actS.has(3)?'checked':''}>other</label>`+
    `<label><input type=checkbox class=ss data-s=4 ${actS.has(4)?'checked':''}><span style="color:#f6d860">human</span></label></div>`;
  h+=`<div style="margin-top:5px"><b>length</b> `+
    `<input class=dn id=dmin type=number min=0 step=0.1 placeholder=min value="${minD||''}" `+
    `style="width:54px;padding:1px 4px;background:#272822;border:1px solid #3e3d32;color:var(--fg);border-radius:3px;font-family:inherit;font-size:11px"> – `+
    `<input class=dn id=dmax type=number min=0 step=0.1 placeholder=max value="${maxD||''}" `+
    `style="width:54px;padding:1px 4px;background:#272822;border:1px solid #3e3d32;color:var(--fg);border-radius:3px;font-family:inherit;font-size:11px"> s</div>`;
  el.innerHTML=h;
  el.querySelectorAll('.lg').forEach(s=>s.onclick=()=>{const i=+s.dataset.i;
    actC.has(i)?actC.delete(i):actC.add(i);legend();draw();});
  lall.onclick=e=>{e.preventDefault();M.cats.forEach((_,i)=>actC.add(i));legend();draw();};
  lnone.onclick=e=>{e.preventDefault();actC.clear();legend();draw();};
  el.querySelectorAll('.tt').forEach(cb=>cb.onchange=()=>{const t=+cb.dataset.t;
    cb.checked?actT.add(t):actT.delete(t);draw();});
  el.querySelectorAll('.ss').forEach(cb=>cb.onchange=()=>{const s=+cb.dataset.s;
    cb.checked?actS.add(s):actS.delete(s);draw();});
  el.querySelectorAll('.dn').forEach(inp=>inp.oninput=()=>{
    minD=parseFloat(document.getElementById('dmin').value)||0;
    maxD=parseFloat(document.getElementById('dmax').value)||0;
    draw();});}
async function loadMap(reset){
  M=await(await fetch('/api/map')).json();
  document.getElementById('count').textContent=
    M.n?M.n.toLocaleString()+' samples':'no projection yet — click Update map';
  if(actC===null){actC=new Set(M.cats.map((_,i)=>i));actT=new Set([0,1,2]);actS=new Set([0,1,2,3,4]);}
  legend();if(reset){resize();fit();}draw();}
const upd=document.getElementById('upd');
upd.onclick=async()=>{upd.disabled=true;upd.textContent='projecting…';
  await fetch('/api/reproject');pollReproj();};
async function pollReproj(){
  const s=await(await fetch('/api/reproject_status')).json();
  if(s.running){setTimeout(pollReproj,2000);return;}
  await loadMap(false);
  upd.disabled=false;upd.textContent='↻ Update map';
  if(s.ok===false)alert('projection failed: '+s.msg);}
// Keep selected point inside the canvas with a margin
function scrollToSel(){
  if(sel<0)return;
  const m=80*dpr,px=sx(sel),py=sy(sel);
  if(px<m)tx+=m-px;else if(px>cv.width-m)tx-=px-(cv.width-m);
  if(py<m)ty+=m-py;else if(py>cv.height-m)ty-=py-(cv.height-m);}
// Arrow-key navigation: jump to nearest visible point in that direction.
// Direction cone: only candidates with dot(normalised offset, dir) >= 0.3 (~73°).
// Score = dist / dot — penalises off-axis candidates so we always move "forward".
// y-axis: M.y=1 is top of screen (render uses 1-y), so ArrowUp means larger M.y.
let _inspectTimer=null;
document.addEventListener('keydown',e=>{
  if(!M||sel<0)return;
  const dir={ArrowRight:[1,0],ArrowLeft:[-1,0],ArrowUp:[0,1],ArrowDown:[0,-1]}[e.key];
  if(!dir)return;
  e.preventDefault();
  const[dx,dy]=dir,cx=M.x[sel],cy=M.y[sel];
  let best=-1,bestScore=Infinity;
  for(let i=0;i<M.n;i++){
    if(i===sel||!shown(i))continue;
    if(stride>1&&i%stride!==0)continue;
    const vx=M.x[i]-cx,vy=M.y[i]-cy,dist=Math.sqrt(vx*vx+vy*vy);
    if(!dist)continue;
    const dot=(vx*dx+vy*dy)/dist;
    if(dot<0.3)continue;
    const score=dist/dot;
    if(score<bestScore){bestScore=score;best=i;}}
  if(best<0)return;
  sel=best;scrollToSel();draw();
  clearTimeout(_inspectTimer);
  _inspectTimer=setTimeout(()=>inspect(best),200);});
loadMap(true);
</script></body></html>"""


SETTINGS_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Settings — Sample Tagger</title>
<style>
:root{--bg:#272822;--fg:#f8f8f2;--dim:#75715e;--card:#1e1f1c;--accent:#a6e22e;
--blue:#66d9ef;--pink:#f92672;--orange:#fd971f;--purple:#ae81ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:16px 20px;border-bottom:1px solid #3e3d32;display:flex;
align-items:center;gap:16px;flex-wrap:wrap}
h1{font-size:18px;margin:0;color:var(--accent)}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
main{padding:20px;max-width:760px}
.group{background:var(--card);border:1px solid #3e3d32;border-radius:10px;
padding:16px;margin-bottom:16px}
.group h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
color:var(--blue);margin:0 0 12px}
.field{display:grid;grid-template-columns:140px 1fr;gap:6px 10px;margin:10px 0;align-items:start}
.field label{color:var(--dim);font-size:12px;padding-top:6px}
.field input[type=text],.field input[type=number],.field select{
background:#272822;border:1px solid #3e3d32;color:var(--fg);padding:5px 8px;
border-radius:5px;font-family:inherit;font-size:13px;width:100%;box-sizing:border-box;min-width:0}
.field input[type=number]{width:90px}
.field .hint{font-size:11px;color:var(--dim);grid-column:2;margin-top:2px;line-height:1.4}
.field .chk-row{grid-column:2;margin:0}
.field .chk-row+.hint{margin-top:4px}
.chk-row{display:flex;align-items:center;gap:8px;margin:7px 0}
.chk-row input[type=checkbox]{width:15px;height:15px;cursor:pointer;accent-color:var(--accent)}
.chk-row label{color:var(--fg);font-size:13px;cursor:pointer}
button{background:#3e3d32;border:1px solid #5a594a;color:var(--fg);
padding:7px 16px;border-radius:6px;font-family:inherit;font-size:13px;cursor:pointer}
button:hover:not(:disabled){background:#4a4940}
button:disabled{opacity:.5;cursor:default}
#btnSave{background:var(--accent);color:#000;border-color:var(--accent);font-weight:bold}
#btnSave:hover:not(:disabled){background:#8fcf25}
#btnDiscover,#btnLabel{border-color:var(--blue);color:var(--blue)}
#btnDiscover:hover:not(:disabled),#btnLabel:hover:not(:disabled){background:#1e2e30}
#btnStop{border-color:var(--pink);color:var(--pink)}
#btnStop:hover:not(:disabled){background:#2e1a1a}
.run-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
.status-box{background:#2d2e28;border-radius:6px;padding:8px 12px;
font-size:12px;margin-bottom:12px;min-height:34px}
.badge{padding:2px 10px;border-radius:10px;font-size:12px;font-weight:bold}
.run{background:var(--accent);color:#000}.idle{background:#3e3d32;color:var(--dim)}
.msg{font-size:12px;min-height:18px;margin-top:8px}
.ok{color:var(--accent)}.err{color:var(--pink)}
</style></head><body>
<header><h1>⚙ Settings</h1>
<a href="/">◂ Dashboard</a>
<a href="/map">🗺 Map</a>
</header>
<main>

<div class=group>
<h2>Library</h2>
<div class=field>
  <label>Library path</label>
  <input type=text id=library_path>
</div>
</div>

<div class=group>
<h2>Stage 1 — Discover</h2>
<p style="font-size:12px;color:var(--dim);margin:0 0 12px">Walk the filesystem, register new files, mark deleted ones. Extracts path hints for free — no audio decoding.</p>

<div class=field>
  <label>Trust DB</label>
  <div class=chk-row style="margin:0">
    <input type=checkbox id=trust_db>
    <label for=trust_db>Skip mtime/size checks on known files</label>
  </div>
  <span class=hint>Skips a stat() call per file. Safe and much faster on network/cloud mounts where stat() is slow. Only new files are fully checked.</span>
</div>

<div class=field>
  <label>Force walk</label>
  <div class=chk-row style="margin:0">
    <input type=checkbox id=no_cache>
    <label for=no_cache>Ignore cached file list</label>
  </div>
  <span class=hint>The scanner caches the file list to avoid slow pcloud walks. Enable this once after adding new files, then turn it off again.</span>
</div>
</div>

<div class=group>
<h2>Stage 2 — Label</h2>
<p style="font-size:12px;color:var(--dim);margin:0 0 12px">Compute classifier results for indexed files. Each classifier writes to its own column — results are independent and never overwrite each other automatically.</p>

<div class=field>
  <label>Workers</label>
  <input type=number id=workers min=1 max=24>
  <span class=hint>Parallel processes. PANNs loads ~1 GB of model weights per worker.</span>
</div>

<div class=field>
  <label>Classifiers</label>
  <div>
    <div class=chk-row style="margin:2px 0">
      <input type=checkbox id=label_path>
      <label for=label_path><b>Path</b> — folder/filename heuristics (instant, no audio decode)</label>
    </div>
    <div class=chk-row style="margin:2px 0">
      <input type=checkbox id=label_audio>
      <label for=label_audio><b>Audio</b> — spectral analysis via librosa (medium speed)</label>
    </div>
    <div class=chk-row style="margin:2px 0">
      <input type=checkbox id=label_panns>
      <label for=label_panns><b>PANNs</b> — CNN14 neural net (slow, also produces the map embedding)</label>
    </div>
  </div>
  <span class=hint>Only files missing that column are processed. Already-labeled files are skipped unless Redo is set.</span>
</div>

<div class=field>
  <label>GPU Python</label>
  <input type=text id=gpu_python placeholder="e.g. /home/phlp/sample-tagger/venv_gpu/bin/python">
  <span class=hint>Path to a CUDA-enabled Python interpreter for the Label stage. Leave blank to use the default (CPU). Used only when PANNs is selected.</span>
</div>

<div class=field>
  <label>Redo</label>
  <input type=text id=redo placeholder="e.g. panns  or  all">
  <span class=hint>Comma-separated classifiers to force-overwrite (e.g. <code>panns</code> after a model update, or <code>all</code> to redo everything). Leave blank to only fill missing labels.</span>
</div>

<div class=field>
  <label>Limit</label>
  <input type=number id=limit min=0>
  <span class=hint>Stop after this many files per run (0 = no limit). Useful for test runs.</span>
</div>
</div>

<div class=group>
<h2>Analysis thresholds</h2>
<div class=field>
  <label>Analyze seconds</label>
  <input type=number id=analyze_seconds min=1 max=300 step=0.5>
  <span class=hint>decode up to N s per file for spectral analysis</span>
</div>
<div class=field>
  <label>Loop min duration (s)</label>
  <input type=number id=loop_min_sec min=0.1 max=10 step=0.1>
  <span class=hint>shorter clips → always oneshot</span>
</div>
<div class=field>
  <label>Loop bar tolerance</label>
  <input type=number id=loop_bar_tolerance min=0 max=0.5 step=0.01>
  <span class=hint>fraction of bar; how close to a beat grid counts as aligned</span>
</div>
<div class=field>
  <label>Harmonic ratio (tonal)</label>
  <input type=number id=harmonic_ratio_tonal min=0 max=1 step=0.01>
  <span class=hint>HPSS energy ratio threshold for tonal vs percussive</span>
</div>
<div class=field>
  <label>BPM min</label>
  <input type=number id=bpm_min min=20 max=120>
</div>
<div class=field>
  <label>BPM max</label>
  <input type=number id=bpm_max min=80 max=400>
</div>
<div class=field>
  <label>PANNs min duration (s)</label>
  <input type=number id=panns_min_duration min=0 max=5 step=0.1>
  <span class=hint>skip PANNs classification on clips shorter than this</span>
</div>
</div>

<div class=group>
<h2>Projection (UMAP / PCA)</h2>
<div class=field>
  <label>Method</label>
  <select id=proj_method>
    <option value=auto>auto (umap if installed, else pca)</option>
    <option value=umap>umap</option>
    <option value=pca>pca</option>
  </select>
</div>
<div class=field>
  <label>UMAP n_neighbors</label>
  <input type=number id=proj_n_neighbors min=2 max=200>
  <span class=hint>local neighbourhood size; larger = more global structure</span>
</div>
<div class=field>
  <label>UMAP min_dist</label>
  <input type=number id=proj_min_dist min=0 max=1 step=0.01>
  <span class=hint>tightness of clusters (0 = compact, 1 = spread)</span>
</div>
</div>

<div class=group>
<h2>Run control</h2>
<div class=status-box id=statusBox>
  <span class=badge id=runBadge>…</span>
  <span id=runInfo style="margin-left:8px;color:var(--dim)"></span>
</div>
<div class=run-row>
  <button id=btnSave>💾 Save config</button>
  <button id=btnDiscover disabled>🔍 Discover</button>
  <button id=btnLabel disabled>🏷 Label</button>
  <button id=btnStop disabled>■ Stop</button>
</div>
<div class=msg id=msg></div>
</div>

</main>
<script>
const FIELDS=['library_path','workers','trust_db','no_cache',
  'label_path','label_audio','label_panns','gpu_python','redo','limit',
  'analyze_seconds','loop_min_sec','loop_bar_tolerance','harmonic_ratio_tonal',
  'bpm_min','bpm_max','panns_min_duration','proj_method','proj_n_neighbors','proj_min_dist'];
const BOOLS=new Set(['trust_db','no_cache','label_path','label_audio','label_panns']);

function getForm(){
  const d={};
  for(const k of FIELDS){
    const el=document.getElementById(k);
    if(!el) continue;
    if(BOOLS.has(k)) d[k]=el.checked;
    else if(el.tagName==='SELECT') d[k]=el.value;
    else if(el.type==='number') d[k]=el.value===''?null:Number(el.value);
    else d[k]=el.value.trim();
  }
  return d;
}
function setForm(cfg){
  for(const k of FIELDS){
    const el=document.getElementById(k);
    if(!el) continue;
    if(BOOLS.has(k)) el.checked=!!cfg[k];
    else if(el.tagName==='SELECT') el.value=cfg[k]||'auto';
    else el.value=cfg[k]!=null?cfg[k]:'';
  }
}
function setMsg(txt,isErr){
  const el=document.getElementById('msg');
  el.textContent=txt; el.className='msg '+(isErr?'err':'ok');
}

async function loadConfig(){
  const r=await fetch('/api/config');
  setForm(await r.json());
}

document.getElementById('btnSave').onclick=async()=>{
  try{
    const r=await fetch('/api/config',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(getForm())});
    if(r.ok) setMsg('Config saved.');
    else setMsg('Save failed: '+(await r.text()),true);
  }catch(e){setMsg('Save error: '+e,true);}
};

async function runStage(stage){
  try{
    const r=await fetch('/api/run/'+stage,{method:'POST'});
    const d=await r.json();
    if(d.ok) setMsg(stage.charAt(0).toUpperCase()+stage.slice(1)+' started (PID '+d.pid+').');
    else setMsg('Failed: '+d.msg,true);
  }catch(e){setMsg('Error: '+e,true);}
  updateStatus();
}
document.getElementById('btnDiscover').onclick=async()=>runStage('discover');
document.getElementById('btnLabel').onclick=async()=>runStage('label');

document.getElementById('btnStop').onclick=async()=>{
  if(!confirm('Send SIGTERM to the running process?')) return;
  try{
    const r=await fetch('/api/run/stop',{method:'POST'});
    const d=await r.json();
    setMsg(d.pid?'Stop signal sent to PID '+d.pid+'.':'No running process found.');
  }catch(e){setMsg('Stop error: '+e,true);}
  updateStatus();
};

async function updateStatus(){
  try{
    const s=await fetch('/api/run/status').then(r=>r.json());
    const badge=document.getElementById('runBadge');
    const info=document.getElementById('runInfo');
    const running=s.running;
    if(running){
      badge.className='badge run'; badge.textContent='RUNNING';
      let txt='PID '+s.pid;
      if(s.progress&&s.progress.total){
        const done=s.progress.done||0;
        const rem=s.progress.total-done;
        txt+=' · '+done.toLocaleString()+' / '+s.progress.total.toLocaleString();
        if(s.progress.eta_min!=null) txt+=' · eta '+(s.progress.eta_min>60?(s.progress.eta_min/60).toFixed(1)+'h':s.progress.eta_min+'m');
      }
      info.textContent=txt;
    } else {
      badge.className='badge idle'; badge.textContent='IDLE';
      info.textContent='';
    }
    document.getElementById('btnDiscover').disabled=running;
    document.getElementById('btnLabel').disabled=running;
    document.getElementById('btnStop').disabled=!running;
  }catch(e){}
  setTimeout(updateStatus,3000);
}

loadConfig();
updateStatus();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode() if isinstance(body, str) else body
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _qs(self):
        return urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

    def serve_audio(self, path):
        if not path or not valid_sample(path) or not os.path.isfile(path):
            self._send(404, "not found", "text/plain"); return
        ct = AUDIO_CT.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        size = os.path.getsize(path)
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1); start = min(start, end); status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def serve_audio_normalized(self, path):
        if not path or not valid_sample(path) or not os.path.isfile(path):
            self._send(404, "not found", "text/plain"); return
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
            self.serve_audio(path); return
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            proc.kill()
        finally:
            proc.wait()

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        body = self._read_body()
        if route == "/api/config":
            try:
                data = json.loads(body) if body else {}
                cfg = save_config(data)
                self._send(200, json.dumps(cfg), "application/json")
            except Exception as e:
                self._send(400, str(e), "text/plain")
        elif route == "/api/run/start":
            self._send(200, json.dumps({"ok": False, "msg": "use /api/run/discover or /api/run/label"}), "application/json")
        elif route == "/api/run/stop":
            self._send(200, json.dumps(run_stop()), "application/json")
        elif route == "/api/run/discover":
            self._send(200, json.dumps(run_start("discover")), "application/json")
        elif route == "/api/run/label":
            self._send(200, json.dumps(run_start("label")), "application/json")
        elif route == "/api/label":
            try:
                data = json.loads(body) if body else {}
                self._send(200, json.dumps(label_api(data.get("path",""), data.get("instrument",""))), "application/json")
            except Exception as e:
                self._send(400, str(e), "text/plain")
        elif route == "/api/label_type":
            try:
                data = json.loads(body) if body else {}
                self._send(200, json.dumps(label_type_api(data.get("path",""), data.get("sample_type",""))), "application/json")
            except Exception as e:
                self._send(400, str(e), "text/plain")
        elif route == "/api/labels/add":
            try:
                data = json.loads(body) if body else {}
                self._send(200, json.dumps(add_label(data.get("name",""))), "application/json")
            except Exception as e:
                self._send(400, str(e), "text/plain")
        elif route == "/api/labels/delete":
            try:
                data = json.loads(body) if body else {}
                self._send(200, json.dumps(delete_label(data.get("name",""))), "application/json")
            except Exception as e:
                self._send(400, str(e), "text/plain")
        else:
            self._send(404, "not found", "text/plain")

    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        if route == "/" or route.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif route == "/map":
            self._send(200, MAP_PAGE, "text/html; charset=utf-8")
        elif route == "/review":
            self._send(200, REVIEW_PAGE, "text/html; charset=utf-8")
        elif route == "/api/labels":
            self._send(200, json.dumps(get_labels()), "application/json")
        elif route == "/api/review/queue":
            qs = self._qs()
            mode = (qs.get("mode") or ["disagree"])[0]
            self._send(200, json.dumps(review_queue(mode)), "application/json")
        elif route == "/settings":
            self._send(200, SETTINGS_PAGE, "text/html; charset=utf-8")
        elif route == "/api/config":
            self._send(200, json.dumps(load_config()), "application/json")
        elif route == "/api/run/status":
            self._send(200, json.dumps(run_status()), "application/json")
        elif route == "/api/stats":
            try:
                self._send(200, json.dumps(stats()), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"ready": False, "msg": str(e)}),
                           "application/json")
        elif route == "/api/log":
            try:
                with open(RUNLOG) as f:
                    lines = f.readlines()
                tail = [l.rstrip() for l in lines[-12:] if l.strip()]
                self._send(200, json.dumps(tail), "application/json")
            except OSError:
                self._send(200, json.dumps([]), "application/json")
        elif route == "/api/errors":
            self._send(200, json.dumps(recent_errors()), "application/json")
        elif route == "/api/similar":
            q = self._qs()
            query = (q.get("path") or q.get("q") or [""])[0]
            k = int((q.get("k") or ["24"])[0])
            self._send(200, json.dumps(similar_api(query, k)), "application/json")
        elif route == "/api/map":
            self._send(200, json.dumps(map_api()), "application/json")
        elif route == "/api/point":
            i = int((self._qs().get("i") or ["-1"])[0])
            self._send(200, json.dumps(point_api(i)), "application/json")
        elif route == "/api/audio":
            qs = self._qs()
            path = (qs.get("path") or [""])[0]
            if (qs.get("norm") or [""])[0] == "1":
                self.serve_audio_normalized(path)
            else:
                self.serve_audio(path)
        elif route == "/api/reproject":
            self._send(200, json.dumps(reproject_start()), "application/json")
        elif route == "/api/reproject_status":
            self._send(200, json.dumps(_REPROJ), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, *a):
        pass  # quiet


def migrate_db():
    con = sqlite3.connect(DB, timeout=10)
    try:
        for col in ("human_sample_type TEXT", "human_instrument TEXT"):
            try:
                con.execute(f"ALTER TABLE samples ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        con.commit()
    finally:
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
    con = sqlite3.connect(LABELS_DB, timeout=5)
    try:
        return [r[0] for r in con.execute("SELECT name FROM labels ORDER BY name").fetchall()]
    finally:
        con.close()


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
    con = sqlite3.connect(LABELS_DB, timeout=10)
    try:
        con.execute("DELETE FROM labels WHERE name=?", (name,))
        con.commit()
    finally:
        con.close()
    return {"ok": True}


def main():
    global DB
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()
    DB = args.db
    migrate_db()
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"dashboard: http://{args.host}:{args.port}  (db: {DB})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
