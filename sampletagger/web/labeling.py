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

def label_api(path, labels):
    """Save a human label SET (ordered: labels[0] = dominant). The set is the truth
    (sample_labels table); human_instrument is kept as the rank-1 projection for
    map/stats/queue queries. Empty list clears everything."""
    if not path:
        return {"ok": False, "msg": "no path"}
    labels = [l for l in (labels or []) if l]
    seen = set()
    labels = [l for l in labels if not (l in seen or seen.add(l))]   # dedupe, keep order
    valid = set(state.get_labels())
    bad = [l for l in labels if l not in valid]
    if bad:
        return {"ok": False, "msg": f"unknown instrument(s): {', '.join(bad)}"}
    if not state.valid_sample(path):
        return {"ok": False, "msg": "unknown path"}
    primary = labels[0] if labels else None
    ts = time.time()
    con = sqlite3.connect(state.DB, timeout=10)
    try:
        con.execute("DELETE FROM sample_labels WHERE path=?", (path,))
        con.executemany("INSERT INTO sample_labels (path, label, rank, ts) VALUES (?,?,?,?)",
                        [(path, l, i + 1, ts) for i, l in enumerate(labels)])
        con.execute(
            """UPDATE samples SET
                 human_instrument=?,
                 label_source=?,
                 ts=?,
                 instrument = COALESCE(?, model_instrument, panns_instrument, path_instrument),
                 source = CASE
                    WHEN ? IS NOT NULL THEN 'human'
                    WHEN model_instrument IS NOT NULL THEN 'model'
                    WHEN panns_instrument IS NOT NULL THEN 'panns'
                    WHEN path_instrument IS NOT NULL THEN 'path'
                    ELSE 'none' END
               WHERE path=?""",
            (primary, ("single" if primary else None), ts,
             primary, primary, path))
        con.commit()
    finally:
        con.close()
    with state.cache_lock:
        state._MAP = None
    return {"ok": True, "path": path, "labels": labels}

def _bulk_label(paths, labels, source, only_unlabeled=True):
    """Bulk-apply a label SET (labels[0] = primary) to paths. Selects the affected
    paths first so sample_labels rows are written for exactly the rows updated."""
    labels = [l for l in (labels or []) if l]
    if not paths or not labels:
        return 0
    primary = labels[0]
    con = sqlite3.connect(state.DB, timeout=20)
    try:
        ts = time.time()
        qs = ",".join("?" * len(paths))
        if only_unlabeled:
            targets = [r[0] for r in con.execute(
                f"SELECT path FROM samples WHERE path IN ({qs}) "
                "AND (human_instrument IS NULL OR human_instrument='')", paths)]
        else:
            targets = [r[0] for r in con.execute(
                f"SELECT path FROM samples WHERE path IN ({qs})", paths)]
        if not targets:
            return 0
        con.executemany(
            """UPDATE samples SET
                 human_instrument=?, label_source=?, ts=?,
                 instrument=?, source='human'
               WHERE path=?""",
            [(primary, source, ts, primary, p) for p in targets])
        # a bulk apply is a new judgment: replace each target's whole set
        con.executemany("DELETE FROM sample_labels WHERE path=?",
                        [(p,) for p in targets])
        con.executemany(
            "INSERT INTO sample_labels (path, label, rank, ts) VALUES (?,?,?,?)",
            [(p, l, i + 1, ts) for p in targets for i, l in enumerate(labels)])
        con.commit()
        return len(targets)
    finally:
        con.close()

def label_propagate(paths, labels):
    """Propagate a full label set to similar files — near-duplicates of a
    crossover sound get the whole judgment, not just the primary."""
    if isinstance(labels, str):
        labels = [labels]
    labels = [l for l in (labels or []) if l]
    if not paths or not labels:
        return {"ok": False, "msg": "missing paths or labels"}
    valid = set(state.get_labels())
    bad = [l for l in labels if l not in valid]
    if bad:
        return {"ok": False, "msg": f"unknown instrument(s): {', '.join(bad)}"}
    n = _bulk_label(paths, labels, "propagate", only_unlabeled=True)
    with state.cache_lock:
        state._MAP = None
    return {"ok": True, "n": n, "labels": labels}

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
        
    n = _bulk_label(paths, [instrument], "map", only_unlabeled=(mode == "unlabeled"))
    with state.cache_lock:
        state._MAP = None
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
        ts = time.time()
        scon.execute("DELETE FROM sample_labels WHERE label=?", (name,))
        # samples whose primary was the deleted label: promote the next-ranked
        # label to primary, or clear entirely if the set is now empty
        cur = scon.execute(
            """UPDATE samples SET
                 human_instrument = (SELECT label FROM sample_labels sl
                                     WHERE sl.path=samples.path ORDER BY rank LIMIT 1),
                 label_source = CASE WHEN EXISTS (SELECT 1 FROM sample_labels sl
                                                  WHERE sl.path=samples.path)
                                     THEN label_source ELSE NULL END,
                 ts=?
               WHERE human_instrument=?""", (ts, name))
        cleared = cur.rowcount
        scon.commit()
    finally:
        scon.close()
    with state.cache_lock:
        state._MAP = None
    return {"ok": True, "cleared": max(cleared, 0)}

def review_queue(mode="unified", limit=80):
    if not os.path.exists(state.DB):
        return {"items": [], "total": 0}
        
    params = ()
    if mode == "gold":
        where = "gold_candidate=1 AND (human_instrument IS NULL OR human_instrument='')"
    elif mode.startswith("class_"):
        # a file belongs to "Target: X" if X is confidently predicted (model_labels)
        # OR if X's head is the file's most uncertain one (model_margin_label) —
        # the latter is where a label teaches that head the most, and those rows
        # sort to the top automatically since the global margin IS that head's.
        where = ("status != 'missing' AND source != 'human' "
                 "AND (human_instrument IS NULL OR human_instrument='') "
                 "AND (model_margin_label = ? OR EXISTS "
                 "     (SELECT 1 FROM model_labels ml "
                 "      WHERE ml.path = samples.path AND ml.label = ?))")
        params = (mode[len("class_"):], mode[len("class_"):])
    else:
        where = "status != 'missing' AND source != 'human' AND (human_instrument IS NULL OR human_instrument='')"
    
    with state.ro() as con:
        if not con:
            return {"items": [], "total": 0}
        total = con.execute(f"SELECT COUNT(*) FROM samples WHERE {where}", params).fetchone()[0]
        
        # Disagreement must compare in ONE vocabulary: weak classifiers still emit
        # old-taxonomy names (tonal/drums/hihat/...), so map them through weak_map
        # before comparing. wm = weak_map + identity for valid labels; anything
        # unmappable (e.g. 'tonal') maps to NULL and can't create fake disagreement.
        wm = dict(state.get_weakmap())
        for l in state.get_labels():
            wm.setdefault(l, l)
        wm_values = ",".join(["(?,?)"] * len(wm)) or "(NULL,NULL)"
        wm_params = [x for kv in wm.items() for x in kv]
        query = f"""
        WITH wm(old, new) AS (VALUES {wm_values})
        SELECT path, path_instrument, panns_instrument, panns_conf,
               duration_s, sample_type, human_sample_type, human_instrument,
               model_instrument, model_conf, rating, cluster_id, cluster_l1
        FROM (
          SELECT samples.*,
                 wp.new AS m_panns,
                 ROW_NUMBER() OVER (
            PARTITION BY COALESCE(model_instrument, 'unknown')
            ORDER BY COALESCE(model_margin, model_conf, 1.0) - (
              CASE WHEN
                    (path_instrument IS NOT NULL AND wp.new IS NOT NULL AND path_instrument != wp.new)
                 OR (model_instrument IS NOT NULL AND path_instrument IS NOT NULL AND model_instrument != path_instrument)
              THEN 2.0 ELSE 0.0 END
            ) ASC
          ) as rn
          FROM samples
          LEFT JOIN wm wp ON wp.old = samples.panns_instrument
          WHERE {where}
        )
        WHERE rn <= 100
        ORDER BY rn ASC
        """
        rows = con.execute(query, tuple(wm_params) + params).fetchall()

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
              "duration_s": r[4], "sample_type": r[5],
              "human_sample_type": r[6], "human_instrument": r[7],
              "model_instrument": r[8],
              "model_conf": round(r[9], 3) if r[9] else None,
              "rating": r[10] or 0}
             for r in selected_rows]

    # attach label sets: human (ordered by rank) and model (above-threshold, by conf)
    lbl_sets, mdl_sets = {}, {}
    all_paths = [it["path"] for it in items]
    if all_paths:
        qs2 = ",".join("?" * len(all_paths))
        with state.ro() as con:
            if con:
                try:
                    for p, l in con.execute(
                            f"SELECT path, label FROM sample_labels WHERE path IN ({qs2}) "
                            "ORDER BY rank", all_paths):
                        lbl_sets.setdefault(p, []).append(l)
                    for p, l, c in con.execute(
                            f"SELECT path, label, conf FROM model_labels WHERE path IN ({qs2}) "
                            "ORDER BY conf DESC", all_paths):
                        mdl_sets.setdefault(p, []).append([l, c])
                except sqlite3.OperationalError:
                    pass
    for it in items:
        it["human_labels"] = lbl_sets.get(
            it["path"], [it["human_instrument"]] if it["human_instrument"] else [])
        it["model_labels"] = mdl_sets.get(it["path"], [])

    grain_ids = {r[12] for r in selected_rows if r[12] is not None}
    grain_lbl, fam_lbl = {}, []
    if grain_ids:
        from .sonic import sonic_family_labels
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
