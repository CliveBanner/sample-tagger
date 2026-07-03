"""Read-only sonic-family views (audio-only clustering, static data in the DB).

The producer (scripts/sonic_label.py) was retired in the strip-down; the tables it
wrote (sonic_clusters, cluster_id/cluster_l1) are stable and power the map's
"sonic family" overview and the dashboard card.
"""

import sqlite3
import os
from . import state

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
