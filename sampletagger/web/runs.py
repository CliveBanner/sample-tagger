import os
import signal
import subprocess
import time
import re
from . import state

def _tagger_pid():
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

def scan_running():
    return _tagger_pid() is not None

def run_status():
    running, pid = False, None
    if state._RUN_PROC is not None:
        if state._RUN_PROC.poll() is None:
            running, pid = True, state._RUN_PROC.pid
        else:
            state._RUN_PROC = None
    if not running:
        pid = _tagger_pid()
        running = pid is not None
    progress = {}
    try:
        with open(state.RUNLOG) as f:
            for line in f:
                m = re.search(r"(\d+) files to process on \d+ workers", line)
                if m:
                    progress = {"total": int(m.group(1)), "done": 0,
                                "rate": 0, "eta_min": None}
                m = re.search(r"(\d+)/(\d+)\s+([\d.]+)/s\s+eta\s+([\d.]+)m", line)
                if m:
                    progress = {"done": int(m.group(1)), "total": int(m.group(2)),
                                "rate": float(m.group(3)), "eta_min": float(m.group(4))}
    except OSError:
        pass
    return {"running": running, "pid": pid, "progress": progress}

def run_start(stage):
    if scan_running():
        return {"ok": False, "msg": "a scan is already running"}
    cfg = state.load_config()
    gpu_py = cfg.get("gpu_python", "").strip()
    py = gpu_py if (stage == "label" and gpu_py and os.path.isfile(gpu_py)) else state.PYTHON

    cmd = [py, "-m", "sampletagger.cli", stage, "--db", state.DB, "-j", str(cfg.get("workers", 5))]
    if cfg.get("limit"): cmd += ["--limit", str(int(cfg["limit"]))]

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

    logf = open(state.RUNLOG, "a")
    state._RUN_PROC = subprocess.Popen(cmd, stdout=logf, stderr=logf)
    return {"ok": True, "pid": state._RUN_PROC.pid, "stage": stage}

def run_stop():
    pid = None
    if state._RUN_PROC is not None and state._RUN_PROC.poll() is None:
        pid = state._RUN_PROC.pid
        state._RUN_PROC.terminate()
        state._RUN_PROC = None
    else:
        pid = _tagger_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    return {"ok": True, "pid": pid}

def ml_run_start():
    if state._ML_PROC is not None and state._ML_PROC.poll() is None:
        return {"ok": False, "msg": "ML pipeline already running"}
    cmd = [state.PYTHON, "-m", "sampletagger.ml.cli", "pipeline", state.DB]
    logf = open(state.ML_LOG, "w")
    state._ML_PROC = subprocess.Popen(cmd, stdout=logf, stderr=logf)
    state._ML_STATE = "running"
    return {"ok": True, "pid": state._ML_PROC.pid}

def ml_run_stop():
    if state._ML_PROC is not None and state._ML_PROC.poll() is None:
        pid = state._ML_PROC.pid
        state._ML_PROC.terminate()
        state._ML_PROC = None
        state._ML_STATE = "idle"
        return {"ok": True, "pid": pid}
    return {"ok": True, "pid": None}

def ml_run_status():
    running = False
    pid = None
    if state._ML_PROC is not None:
        if state._ML_PROC.poll() is None:
            running, pid = True, state._ML_PROC.pid
            state._ML_STATE = "running"
        else:
            state._ML_STATE = "done" if state._ML_PROC.returncode == 0 else "error"
            state._ML_PROC = None
    head_path = os.path.join(state.HERE, "models", "head.joblib")
    last_trained = None
    try:
        last_trained = os.path.getmtime(head_path)
    except OSError:
        pass
    log_tail = []
    try:
        with open(state.ML_LOG) as f:
            log_tail = [l.rstrip() for l in f.readlines()[-20:] if l.strip()]
    except OSError:
        pass
    return {"running": running, "pid": pid, "state": state._ML_STATE,
            "log_tail": log_tail, "last_trained": last_trained}

def log_tail():
    try:
        with open(state.RUNLOG) as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-12:] if l.strip()]
    except OSError:
        return []
