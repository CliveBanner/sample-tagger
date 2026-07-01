import sqlite3
import os
from collections import defaultdict, Counter
from . import state

def build_clusters():
    with state.cache_lock:
        db_mtime = state._db_mtime()
        if state._CLUSTERS is not None and state._CLUSTERS_MTIME == db_mtime:
            return state._CLUSTERS

    with state.ro() as con:
        if not con:
            with state.cache_lock:
                state._CLUSTERS = []
                state._CLUSTERS_MTIME = db_mtime
            return []
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
            with state.cache_lock:
                state._CLUSTERS = []
                state._CLUSTERS_MTIME = db_mtime
            return []

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
    with state.cache_lock:
        state._CLUSTERS = out
        state._CLUSTERS_MTIME = db_mtime
    return out

def clusters_list(mode="value", limit=300):
    items = [c for c in build_clusters() if c["n"] >= 2]
    if mode == "messy":
        items = [c for c in items if c["agreement"] < 0.6]
        items.sort(key=lambda c: -c["n"])
    else:
        items = [c for c in items if c["agreement"] >= 0.5]
        items.sort(key=lambda c: -(c["n"] * c["agreement"]))
    total = len(items)
    items = items[:limit]
    return {"items": [{**c, "medoid_name": os.path.basename(c["medoid"]) if c["medoid"] else ""}
                      for c in items], "total": total}

def cluster_detail(cid):
    with state.ro() as con:
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
    c = Counter(m["model_instrument"] for m in members
                if not m["human_instrument"] and m["model_instrument"])
    return {"id": cid, "members": members, "label": c.most_common(1)[0][0] if c else None}

def sonic_family_labels(con):
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
    with state.ro() as con:
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
    with state.ro() as con:
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
    with state.ro() as con:
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
