import os
import sys
import threading
import time
import sqlite3
import json
from contextlib import contextmanager

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(HERE, "samples.db")
LABELS_DB = os.path.join(HERE, "labels.db")
RUNLOG = os.path.join(HERE, "run.log")
ML_LOG = os.path.join(HERE, "ml.log")
CONFIG_FILE = os.path.join(HERE, "config.json")

PYTHON = os.path.join(HERE, "venv", "bin", "python")
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable

INSTR_COLORS = {
    "kick": "#f92672", "snare_clap": "#fd971f", "hats_cymbals": "#e6db74",
    "tom": "#e6a23c", "perc": "#a6e22e",
    "bass": "#66d9ef",
    "piano_keys": "#5c6bc0", "organ": "#42a5f5",
    "mallet": "#c0ca33",
    "guitar": "#26a69a", "strings": "#29b6f6", "brass": "#ffb300", "winds": "#9ccc65",
    "synth": "#9a6cff", "pad": "#7986cb",
    "vocal": "#ff5fa2",
    "sfx": "#75715e",
}
INSTRUMENTS = list(INSTR_COLORS)
AUDIO_CT = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
            ".aif": "audio/aiff", ".aiff": "audio/aiff", ".ogg": "audio/ogg"}
SAMPLE_TYPES = ("oneshot", "loop")

cache_lock = threading.Lock()
_SIM = None
_SIM_LAST_USE = 0.0
_SIM_TTL = 600
_MAP = None
_RUN_PROC = None
_ML_PROC = None
_ML_STATE = "idle"
_REPROJ = {"running": False, "ok": None, "msg": "idle", "ts": 0}

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

def q(con, sql, params=()):
    return con.execute(sql, params).fetchall()

def _db_mtime():
    m = 0.0
    for suffix in ("", "-wal"):
        try:
            m = max(m, os.path.getmtime(DB + suffix))
        except OSError:
            pass
    return m

from ..db import db_connect

def init(db_path):
    global DB
    DB = db_path
    con = db_connect(DB)
    con.close()
    
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

from ..config import load_config as load_core_config
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

def get_labels():
    with ro(LABELS_DB) as con:
        if not con: return []
        return [r[0] for r in con.execute("SELECT name FROM labels ORDER BY name").fetchall()]

def get_weakmap():
    """old-taxonomy name → current label (weak_map table, seeded by ml.export)."""
    with ro(LABELS_DB) as con:
        if not con: return {}
        try:
            return dict(con.execute("SELECT old_label, new_label FROM weak_map"))
        except sqlite3.OperationalError:
            return {}

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

def valid_sample(path):
    with ro() as con:
        if not con: return False
        return con.execute("SELECT 1 FROM samples WHERE path=? LIMIT 1",
                           (path,)).fetchone() is not None
