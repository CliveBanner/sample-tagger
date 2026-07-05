import sqlite3
from . import state
from . import runs
from .mapview import _maybe_evict_sim

def stats():
    _maybe_evict_sim()
    with state.ro() as con:
        if not con:
            return {"ready": False, "msg": "samples.db not found yet"}
        con.row_factory = sqlite3.Row
        rs = runs.run_status()
        prog = rs.get("progress", {})
        processed = state.q(con, "SELECT COUNT(*) FROM samples WHERE status != 'missing'")[0][0]
        errors = state.q(con, "SELECT COUNT(*) FROM samples WHERE status='error'")[0][0]

        total = prog.get("total") or processed
        rate = prog.get("rate") or 0
        eta_min = prog.get("eta_min")

        active = state.q(con, "SELECT COUNT(*) FROM samples WHERE status != 'missing'")[0][0]
        cov_row = state.q(con, """
            SELECT COUNT(path_instrument), COUNT(panns_instrument),
                   COUNT(human_instrument),
                   SUM(CASE WHEN path_instrument IS NOT NULL
                             OR panns_instrument IS NOT NULL THEN 1 ELSE 0 END)
            FROM samples WHERE status != 'missing'""")[0]
            
        coverage = {
            "active": active,
            "path":  {"n": cov_row[0], "total": active},
            "panns": {"n": cov_row[1], "total": active},
            "human": {"n": cov_row[2], "total": active},
            "any":   {"n": cov_row[3] or 0, "total": active},
        }

        def inst_dist(col):
            rows = con.execute(
                f"SELECT {col}, COUNT(*) n FROM samples "
                f"WHERE status != 'missing' AND {col} IS NOT NULL "
                f"GROUP BY {col} ORDER BY n DESC LIMIT 14").fetchall()
            return [{"label": r[0], "n": r[1]} for r in rows]

        path_dist  = inst_dist("path_instrument")
        panns_dist = inst_dist("panns_instrument")
        # human counts come from the label sets so secondary labels are visible
        try:
            human_dist = [dict(label=r[0], n=r[1]) for r in state.q(
                con, "SELECT label, COUNT(*) FROM sample_labels "
                     "GROUP BY label ORDER BY COUNT(*) DESC")]
        except sqlite3.OperationalError:
            human_dist = inst_dist("human_instrument")
        sample_type = inst_dist("sample_type")
        keys = inst_dist("key")

        bpm_rows = state.q(con, "SELECT (bpm/10)*10 AS bucket, COUNT(*) FROM samples "
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
            conf_rows = state.q(con, "SELECT CAST(model_conf * 10 AS INTEGER) AS bucket, COUNT(*) FROM samples "
                               "WHERE model_conf IS NOT NULL GROUP BY bucket ORDER BY bucket")
            model_conf_hist = [dict(label=f"{(b/10):.1f}+", n=n) for b, n in conf_rows]
            model_cov = state.q(con, "SELECT COUNT(*) FROM samples WHERE model_conf IS NOT NULL")[0][0]
            coverage["model"] = {"n": model_cov, "total": active}

        try:
            with open(state.RUNLOG) as f:
                log_tail = [l.rstrip() for l in f.readlines()[-12:] if l.strip()]
        except OSError:
            log_tail = []

        try:
            sonic_dist = [{"label": r[1], "n": r[0]} for r in con.execute(
                "SELECT size, label FROM sonic_clusters WHERE level=1 ORDER BY size DESC")]
        except sqlite3.OperationalError:
            sonic_dist = []
            
        return {
            "ready": True,
            "scan_running": runs.scan_running(),
            "active": active, "processed": processed, "errors": errors,
            "rate": round(rate, 2), "eta_min": round(eta_min, 1) if eta_min is not None else None,
            "pct": round(100 * prog.get("done", 0) / total, 1) if total else None,
            "label_done": prog.get("done", 0), "label_total": total,
            "coverage": coverage,
            "path_dist": path_dist, "panns_dist": panns_dist,
            "human_dist": human_dist, "sample_type": sample_type,
            "model_dist": model_dist, "model_conf_hist": model_conf_hist,
            "keys": keys, "bpm_hist": bpm_hist, "log_tail": log_tail,
            "sonic_dist": sonic_dist,
        }

def recent_errors(limit=15):
    with state.ro() as con:
        if not con: return []
        rows = con.execute(
            "SELECT path, error FROM samples WHERE status='error' "
            "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(path=p, error=e) for p, e in rows]
