"""Gold eval-set campaign, DB-backed (replaces scripts/sample_gold.py + freeze_val.py).

State lives in two samples columns:
  gold_candidate=1   file selected for the campaign (label it in the review UI, mode=gold)
  is_val=1           frozen into the eval set (never trained on)

Freeze only accepts label_source='single' rows so bulk/propagate labels can never
contaminate the eval set.
"""

import random
import sqlite3


def sample_gold(db, per_class=25, include_none=50, seed=42):
    """Top up gold candidates to `per_class` per predicted class, plus a slice of
    source='none' files. Additive: existing candidates are kept and counted."""
    rng = random.Random(seed)
    con = sqlite3.connect(db, timeout=30)
    try:
        # Stratify over model_labels (the full multi-label prediction), not the
        # top-1 column: classes that are usually secondary predictions (pad behind
        # synth, drums behind kick) would otherwise get starved buckets. A file can
        # sit in several class pools but is picked at most once.
        have = dict(con.execute(
            "SELECT ml.label, COUNT(DISTINCT ml.path) FROM model_labels ml "
            "JOIN samples s ON s.path=ml.path WHERE s.gold_candidate=1 "
            "GROUP BY ml.label"))

        pool = {}
        for p, inst in con.execute(
                "SELECT ml.path, ml.label FROM model_labels ml "
                "JOIN samples s ON s.path=ml.path "
                "WHERE s.gold_candidate=0 AND (s.human_instrument IS NULL OR s.human_instrument='') "
                "AND s.status != 'missing'"):
            pool.setdefault(inst, []).append(p)

        picked = []
        picked_set = set()
        added_per_class = {}
        for inst, paths in sorted(pool.items()):
            need = per_class - have.get(inst, 0)
            if need > 0:
                candidates = [p for p in paths if p not in picked_set]
                take = rng.sample(candidates, min(need, len(candidates)))
                picked += take
                picked_set.update(take)
                added_per_class[inst] = len(take)

        none_pool = [r[0] for r in con.execute(
            "SELECT path FROM samples WHERE gold_candidate=0 "
            "AND (human_instrument IS NULL OR human_instrument='') "
            "AND source='none' AND status != 'missing'")]
        need_none = include_none - have.get("~none~", 0)
        if need_none > 0 and none_pool:
            take = rng.sample(none_pool, min(need_none, len(none_pool)))
            picked += take
            added_per_class["(none)"] = len(take)

        if picked:
            con.executemany("UPDATE samples SET gold_candidate=1 WHERE path=?",
                            [(p,) for p in picked])
            con.commit()
        return {"ok": True, "added": len(picked), "added_per_class": added_per_class}
    finally:
        con.close()


def status(db):
    """Campaign progress + frozen-val summary for the UI."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    try:
        total, labeled, single, frozen = con.execute(
            "SELECT COUNT(*), "
            "SUM(human_instrument IS NOT NULL AND human_instrument!=''), "
            "SUM(human_instrument IS NOT NULL AND human_instrument!='' AND label_source='single'), "
            "SUM(is_val=1) FROM samples WHERE gold_candidate=1").fetchone()
        # per-class support counts every label in the set (a [kick,drums] file
        # supports both classes), not just the primary
        per_class = [{"label": r[0], "n": r[1]} for r in con.execute(
            "SELECT sl.label, COUNT(*) FROM sample_labels sl "
            "JOIN samples s ON s.path=sl.path WHERE s.gold_candidate=1 "
            "GROUP BY sl.label ORDER BY 2 DESC")]
        val_total, val_per_class = con.execute(
            "SELECT COUNT(*) FROM samples WHERE is_val=1").fetchone()[0], \
            [{"label": r[0], "n": r[1]} for r in con.execute(
                "SELECT sl.label, COUNT(*) FROM sample_labels sl "
                "JOIN samples s ON s.path=sl.path WHERE s.is_val=1 "
                "GROUP BY sl.label ORDER BY 2 DESC")]
        return {"total": total or 0, "labeled": labeled or 0, "single": single or 0,
                "frozen": frozen or 0, "remaining": (total or 0) - (labeled or 0),
                "per_class": per_class,
                "val": {"total": val_total, "per_class": val_per_class}}
    finally:
        con.close()


def freeze(db):
    """Freeze labeled gold candidates into the eval set. Only label_source='single'
    qualifies — bulk/propagate-labeled candidates are reported as skipped."""
    con = sqlite3.connect(db, timeout=30)
    try:
        skipped = con.execute(
            "SELECT COUNT(*) FROM samples WHERE gold_candidate=1 AND is_val=0 "
            "AND human_instrument IS NOT NULL AND human_instrument!='' "
            "AND label_source != 'single'").fetchone()[0]
        cur = con.execute(
            "UPDATE samples SET is_val=1 WHERE gold_candidate=1 AND is_val=0 "
            "AND human_instrument IS NOT NULL AND human_instrument!='' "
            "AND label_source='single'")
        con.commit()
        return {"ok": True, "frozen": cur.rowcount, "skipped_non_single": skipped}
    finally:
        con.close()
