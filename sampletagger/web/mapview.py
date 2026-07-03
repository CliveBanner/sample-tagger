import time
import os
import sqlite3
import subprocess
import sys
import threading
from .. import sim as simlib
from . import state

def get_sim():
    with state.cache_lock:
        state._SIM_LAST_USE = time.time()
        if state._SIM is None:
            state._SIM = simlib.SimIndex(state.DB)
        return state._SIM

def _maybe_evict_sim():
    with state.cache_lock:
        if state._SIM is not None and time.time() - state._SIM_LAST_USE > state._SIM_TTL:
            state._SIM = None

def similar_api(query, k=24):
    ix = get_sim()
    ix.ensure(max_age=0)
    matched, hits = ix.neighbors(query, k)
    if matched is None:
        return {"query": query, "matched": None, "hits": [], "n": len(ix.paths)}
    meta = simlib.fetch_meta(state.DB, [p for p, _ in hits])
    return {"query": query, "matched": matched, "matched_name": os.path.basename(matched),
            "n": len(ix.paths),
            "hits": [dict(path=p, name=os.path.basename(p), score=round(s, 3),
                          **meta.get(p, {})) for p, s in hits]}

def propagate_candidates(path, k=24):
    if not path:
        return {"items": []}
    ix = get_sim()
    ix.ensure(max_age=0)
    matched, hits = ix.neighbors(path, k)
    if matched is None or not hits:
        return {"items": []}
    score = {p: s for p, s in hits}
    cand_paths = list(score)
    with state.ro() as con:
        if not con:
            return {"items": []}
        qs = ",".join("?" * len(cand_paths))
        rows = con.execute(
            f"SELECT path, model_instrument, model_conf, path_instrument, panns_instrument, "
            f"human_instrument, duration_s, sample_type, rating "
            f"FROM samples WHERE path IN ({qs})", cand_paths).fetchall()
    items = []
    for r in rows:
        if r[5]:
            continue
        items.append({"path": r[0], "name": os.path.basename(r[0]),
                      "score": round(score.get(r[0], 0), 3),
                      "model_instrument": r[1],
                      "model_conf": round(r[2], 3) if r[2] else None,
                      "path_instrument": r[3], "panns_instrument": r[4],
                      "duration_s": r[6], "sample_type": r[7], "rating": r[8] or 0})
    items.sort(key=lambda d: -d["score"])
    return {"items": items}

def search_text_api(query, k=24):
    if not query:
        return {"query": query, "hits": [], "n": 0}
    
    from ..ml.clap import get_clap
    import numpy as np
    model = get_clap()
    emb = model.get_text_embedding([query], use_tensor=False)
    q_vec = np.mean(emb, axis=0)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    from ..embeddings import load as load_emb
    paths, X = load_emb(state.DB, dtype=np.float32, mmap=True, model="clap")
    if len(paths) == 0:
        return {"query": query, "hits": [], "n": 0}
        
    scores = X @ q_vec
    
    top_indices = np.argsort(scores)[::-1][:k]
    hits = [(paths[i], scores[i]) for i in top_indices]
    
    meta = simlib.fetch_meta(state.DB, [p for p, _ in hits])
    return {"query": query, "n": len(paths),
            "hits": [dict(path=p, name=os.path.basename(p), score=round(float(s), 3),
                          **meta.get(p, {})) for p, s in hits]}

def build_map():
    with state.cache_lock:
        proj_db = state.DB + ".proj"
        try:
            sidecar_mtime = os.path.getmtime(proj_db)
        except OSError:
            sidecar_mtime = 0
        db_mtime = state._db_mtime()
        if (state._MAP is not None
                and state._MAP.get("_sidecar_mtime") == sidecar_mtime
                and state._MAP.get("_db_mtime") == db_mtime):
            return state._MAP

    fam_labels = []
    with state.ro() as con:
        if not con:
            rows = []
        else:
            try:
                cols = ("s.instrument, s.human_instrument, s.model_instrument, "
                        "s.path_instrument, s.panns_instrument, s.audio_instrument, "
                        "s.sample_type, s.duration_s, s.label_source, s.cluster_l1")
                if sidecar_mtime:
                    con.execute("ATTACH DATABASE ? AS pj", (f"file:{proj_db}?mode=ro",))
                    rows = con.execute(
                        f"SELECT p.path, p.x, p.y, {cols} "
                        "FROM pj.projection p JOIN samples s ON s.path=p.path").fetchall()
                else:
                    rows = con.execute(
                        f"SELECT p.path, p.x, p.y, {cols} "
                        "FROM projection p JOIN samples s ON s.path=p.path").fetchall()
            except sqlite3.OperationalError:
                rows = []
            from .clusters import sonic_family_labels
            fam_labels = sonic_family_labels(con)
            
    colors_dict = state.get_colors()
    cats = sorted(colors_dict.keys())
    cidx = {c: i for i, c in enumerate(cats)}
    none_idx = len(cats)
    tcode = {"oneshot": 0, "loop": 1}
    LS_CODE = {"single": 0, "cluster": 1, "map": 2, "propagate": 3, "llm": 4}
    LS_NONE = 5
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
    
    with state.cache_lock:
        state._MAP = {"_sidecar_mtime": sidecar_mtime if paths else 0, "_db_mtime": db_mtime,
                "paths": paths, "x": xs, "y": ys, "t": ts, "d": ds,
                "fields": fields, "ls": ls,
                "cats": cats, "colors": [colors_dict[c] for c in cats],
                "famCats": fam_labels, "famColors": fam_colors,
                "n": len(paths)}
        return state._MAP

def map_api():
    m = build_map()
    return {
        "x": m["x"], "y": m["y"], "t": m["t"], "d": m["d"],
        "fields": m["fields"], "ls": m["ls"],
        "cats": m["cats"], "colors": m["colors"],
        "famCats": m["famCats"], "famColors": m["famColors"],
        "n": m["n"], "sidecar_mtime": m.get("_sidecar_mtime", 0),
    }

def point_api(i):
    m = build_map()
    if i < 0 or i >= len(m["paths"]):
        return {}
    path = m["paths"][i]
    meta = simlib.fetch_meta(state.DB, [path]).get(path, {})
    with state.ro() as con:
        from .clusters import sonic_for
        sonic = sonic_for(con, path) if con else None
    return dict(path=path, name=os.path.basename(path), sonic=sonic, **meta)

def _do_reproject():
    try:
        r = subprocess.run([state.PYTHON, "-m", "sampletagger.projection", "--db", state.DB],
                           capture_output=True, text=True, timeout=3600)
        ok = r.returncode == 0
        out = (r.stdout if ok else r.stderr).strip().splitlines()
        with state.cache_lock:
            state._REPROJ["ok"] = ok
            state._REPROJ["msg"] = out[-1][:200] if out else ("done" if ok else "failed")
    except Exception as e:
        with state.cache_lock:
            state._REPROJ["ok"] = False
            state._REPROJ["msg"] = str(e)[:200]
    finally:
        with state.cache_lock:
            state._REPROJ["running"] = False
            state._REPROJ["ts"] = time.time()
            state._MAP = None

def reproject_start():
    with state.cache_lock:
        if state._REPROJ["running"]:
            return dict(state._REPROJ)
        state._REPROJ.update(running=True, ok=None, msg="projecting…")
    threading.Thread(target=_do_reproject, daemon=True).start()
    with state.cache_lock:
        return dict(state._REPROJ)
