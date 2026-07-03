import json
import sqlite3
from . import state
from ..ml import gold as goldlib


def gold_status():
    return goldlib.status(state.DB)


def gold_sample(data):
    per_class = int(data.get("per_class", 25))
    include_none = int(data.get("include_none", 50))
    if not (0 < per_class <= 500 and 0 <= include_none <= 2000):
        return {"ok": False, "msg": "per_class 1-500, include_none 0-2000"}
    res = goldlib.sample_gold(state.DB, per_class=per_class, include_none=include_none)
    res["status"] = goldlib.status(state.DB)
    return res


def gold_freeze(data):
    res = goldlib.freeze(state.DB)
    res["status"] = goldlib.status(state.DB)
    return res


def ml_metrics():
    with state.ro() as con:
        if not con:
            return []
        try:
            rows = con.execute(
                "SELECT version, ts, val_n, macro_f1, per_class_f1, coverage, notes "
                "FROM metrics ORDER BY ts").fetchall()
        except sqlite3.OperationalError:   # table not created yet
            return []

    def _j(s):
        try:
            return json.loads(s) if s else None
        except ValueError:
            return None

    return [{"version": r[0], "ts": r[1], "val_n": r[2], "macro_f1": r[3],
             "per_class_f1": _j(r[4]), "coverage": _j(r[5]), "notes": r[6]}
            for r in rows]
