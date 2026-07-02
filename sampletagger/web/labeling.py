import sqlite3
import time
import os
import numpy as np
from . import state
from .mapview import get_sim

def label_type_api(path, sample_type):
    if not path:
        return {"ok": False, "msg": "no path"}
    if sample_type not in state.SAMPLE_TYPES and sample_type != "":
        return {"ok": False, "msg": f"unknown type: {sample_type}"}
    con = sqlite3.connect(state.DB, timeout=10)
    try:
        con.execute(
            "UPDATE samples SET human_sample_type=?, ts=? WHERE path=?",
            (sample_type or None, time.time(), path))
        con.commit()
    finally:
        con.close()
    return {"ok": True, "path": path, "sample_type": sample_type}

def label_api(path, instrument):
    if not path:
        return {"ok": False, "msg": "no path"}
    if instrument and instrument not in state.get_labels():
        return {"ok": False, "msg": f"unknown instrument: {instrument}"}
    con = sqlite3.connect(state.DB, timeout=10)
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
    with state.cache_lock:
        state._MAP = None
        state._CLUSTERS = None
    return {"ok": True, "path": path, "instrument": instrument}

def _bulk_label(paths, instrument, source, only_unlabeled=True):
    if not paths:
        return 0
    con = sqlite3.connect(state.DB, timeout=20)
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
    if not paths or not instrument:
        return {"ok": False, "msg": "missing paths or instrument"}
    if instrument not in state.get_labels():
        return {"ok": False, "msg": f"unknown instrument: {instrument}"}
    n = _bulk_label(paths, instrument, "propagate", only_unlabeled=True)
    with state.cache_lock:
        state._MAP = None
        state._CLUSTERS = None
    return {"ok": True, "n": n, "instrument": instrument}

def label_cluster(cid, instrument, exclude=None):
    if instrument not in state.get_labels():
        return {"ok": False, "msg": f"unknown instrument: {instrument}"}
    exclude = set(exclude or [])
    with state.ro() as con:
        if not con:
            return {"ok": False, "msg": "no db"}
        rows = con.execute(
            "SELECT path FROM samples WHERE cluster_id=? "
            "AND (human_instrument IS NULL OR human_instrument='')", (cid,)).fetchall()
    targets = [p for (p,) in rows if p not in exclude]
    n = _bulk_label(targets, instrument, "cluster", only_unlabeled=True)
    with state.cache_lock:
        state._MAP = None
        state._CLUSTERS = None
    return {"ok": True, "n": n, "instrument": instrument}

def label_map(data):
    indices = data.get("indices", [])
    instrument = data.get("instrument")
    sidecar_mtime = data.get("sidecar_mtime")
    mode = data.get("mode", "all")
    
    if not indices or not instrument:
        return {"ok": False, "msg": "missing indices or instrument"}
    if instrument not in state.get_labels():
        return {"ok": False, "msg": f"unknown instrument: {instrument}"}
        
    from .mapview import build_map
    m = build_map()
    if not m or m.get("_sidecar_mtime") != sidecar_mtime:
        return {"ok": False, "stale": True}
        
    paths = [m["paths"][i] for i in indices if 0 <= i < m["n"]]
    if not paths:
        return {"ok": False, "msg": "no valid indices"}
        
    n = _bulk_label(paths, instrument, "map", only_unlabeled=(mode == "unlabeled"))
    with state.cache_lock:
        state._MAP = None
        state._CLUSTERS = None
    return {"ok": True, "n": n, "instrument": instrument}

def rate_api(path, rating):
    if not path:
        return {"ok": False, "msg": "no path"}
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        return {"ok": False, "msg": "bad rating"}
    if rating < 0 or rating > 5:
        return {"ok": False, "msg": "rating out of range"}
    con = sqlite3.connect(state.DB, timeout=10)
    try:
        con.execute("UPDATE samples SET rating=?, ts=? WHERE path=?",
                    (rating, time.time(), path))
        con.commit()
    finally:
        con.close()
    return {"ok": True, "path": path, "rating": rating}

def add_label(name):
    name = name.strip().lower()
    if not name or len(name) > 40:
        return {"ok": False, "msg": "invalid name"}
    con = sqlite3.connect(state.LABELS_DB, timeout=10)
    try:
        con.execute("INSERT OR IGNORE INTO labels(name,created_at) VALUES(?,?)",
                    (name, time.time()))
        con.commit()
    finally:
        con.close()
    return {"ok": True, "name": name}

def delete_label(name):
    con = sqlite3.connect(state.LABELS_DB, timeout=10)
    try:
        con.execute("DELETE FROM labels WHERE name=?", (name,))
        con.commit()
    finally:
        con.close()
    cleared = 0
    scon = sqlite3.connect(state.DB, timeout=20)
    try:
        cur = scon.execute(
            "UPDATE samples SET human_instrument=NULL, label_source=NULL, ts=? "
            "WHERE human_instrument=?", (time.time(), name))
        cleared = cur.rowcount
        scon.commit()
    finally:
        scon.close()
    with state.cache_lock:
        state._MAP = None
        state._CLUSTERS = None
    return {"ok": True, "cleared": max(cleared, 0)}

def review_queue(mode="unified", limit=80):
    if not os.path.exists(state.DB):
        return {"items": [], "total": 0}
        
    params = ()
    if mode == "gold":
        try:
            with open(os.path.join(state.HERE, "gold_candidates.txt")) as f:
                paths = [l.strip() for l in f if l.strip()]
            if not paths:
                return {"items": [], "total": 0}
            qs = ",".join("?" * len(paths))
            where = f"path IN ({qs}) AND (human_instrument IS NULL OR human_instrument='')"
            params = tuple(paths)
        except OSError:
            return {"items": [], "total": 0}
    else:
        where = "status != 'missing' AND source != 'human' AND (human_instrument IS NULL OR human_instrument='')"
    
    with state.ro() as con:
        if not con:
            return {"items": [], "total": 0}
        total = con.execute(f"SELECT COUNT(*) FROM samples WHERE {where}", params).fetchone()[0]
        
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
        rows = con.execute(query, params).fetchall()

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

    grain_ids = {r[12] for r in selected_rows if r[12] is not None}
    grain_lbl, fam_lbl = {}, []
    if grain_ids:
        from .clusters import sonic_family_labels
        with state.ro() as con:
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
