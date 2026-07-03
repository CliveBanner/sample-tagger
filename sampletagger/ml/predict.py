import os
import time
import json
import sqlite3
from collections import Counter
import numpy as np
import joblib

from .export import get_latest_features, load_labels, load_ml_cfg
from .train import predict_probs


def run_predict(args):
    db_dir = os.path.dirname(os.path.abspath(args.db))
    features_path = get_latest_features(db_dir)
    if not features_path:
        print("No features exported.")
        return

    model_path = os.path.join(db_dir, "models", "head.joblib")
    if not os.path.exists(model_path):
        print("No model found. Run `sample-tagger-ml train` first.")
        return

    print(f"Loading {features_path}...")
    data = np.load(features_path, allow_pickle=True)
    X_all = data['X']
    paths = data['paths']
    # Labels read live from the DB (the .npz embedding cache can be stale on labels).
    lbl = load_labels(args.db, paths)
    human = lbl['human']
    weak_path = lbl['weak_path']

    print(f"Loading {model_path}...")
    saved = joblib.load(model_path)
    if not saved.get("multi_label"):
        print("Old single-label model found — retrain first (`sample-tagger-ml train`).")
        return
    W, b = saved["W"], saved["b"]
    classes = np.array(saved["classes"])
    version = saved["version"]
    thresholds = saved.get("thresholds", {})

    cfg = load_ml_cfg(db_dir)
    conf_threshold = cfg.get("conf_threshold", 0.6)
    thr_vec = np.array([thresholds.get(c, conf_threshold) for c in classes], dtype=np.float32)

    print("Predicting (per-class sigmoid)...")
    t0 = time.time()
    probs = predict_probs(X_all, W, b)
    top1_idx = np.argmax(probs, axis=1)
    top1 = probs[np.arange(len(probs)), top1_idx]
    # Multi-label uncertainty: distance of the most borderline head from its 0.5
    # decision boundary (0 = some head maximally unsure, 1 = all heads decided).
    # top1-top2 is meaningless with independent sigmoids — a confident crossover
    # (two heads at 0.9) has near-zero top-margin but zero uncertainty.
    margins = 2.0 * np.min(np.abs(probs - 0.5), axis=1)
    # which head is the uncertain one — lets the review queue target active
    # learning per class ("files where the SYNTH head is unsure")
    margin_label = classes[np.argmin(np.abs(probs - 0.5), axis=1)]
    model_inst = classes[top1_idx]
    print(f"Predicted {len(X_all)} in {time.time()-t0:.2f}s. Writing to DB...")

    # Multi-label rows: every class above threshold (top-1 always included so the
    # model's best guess is inspectable even when it's unsure).
    ml_rows = []
    for i in range(len(paths)):
        above = np.flatnonzero(probs[i] >= thr_vec)
        if top1_idx[i] not in above:
            above = np.append(above, top1_idx[i])
        for j in above:
            ml_rows.append((paths[i], classes[j], round(float(probs[i, j]), 4)))

    # Resolve final instrument: human > model(top-1 conf >= its own class threshold) > path weak
    updates = []
    for i in range(len(paths)):
        m_conf = float(top1[i])
        m_thr = float(thr_vec[top1_idx[i]])
        final_inst, source = None, "none"
        if human[i]:
            final_inst, source = human[i], "human"
        elif m_conf >= m_thr:
            final_inst, source = model_inst[i], "model"
        elif weak_path[i]:
            final_inst, source = weak_path[i], "path"
        updates.append((model_inst[i], m_conf, version, float(margins[i]),
                        margin_label[i], final_inst, source, paths[i]))

    con = sqlite3.connect(args.db, timeout=30)
    try:
        con.execute("BEGIN TRANSACTION")
        # Direct assignment (not COALESCE): predict recomputes the resolved label
        # for every row, so an unresolved sample must be *cleared*, not left stale.
        con.executemany("""
            UPDATE samples SET
                model_instrument = ?,
                model_conf = ?,
                model_version = ?,
                model_margin = ?,
                model_margin_label = ?,
                instrument = ?,
                source = ?
            WHERE path = ?
        """, updates)
        con.execute("DELETE FROM model_labels")
        con.executemany("INSERT OR REPLACE INTO model_labels (path, label, conf) VALUES (?,?,?)",
                        ml_rows)
        con.commit()
        print(f"Updated {len(updates)} rows, wrote {len(ml_rows)} model_labels "
              f"({len(ml_rows)/max(len(updates),1):.2f} labels/file).")

        # record resolved coverage on this model's metrics row
        src_counts = Counter(u[5] for u in updates)
        con.execute("CREATE TABLE IF NOT EXISTS metrics (version TEXT PRIMARY KEY, ts REAL, "
                    "val_n INTEGER, macro_f1 REAL, per_class_f1 TEXT, coverage TEXT, notes TEXT)")
        row = con.execute("SELECT coverage FROM metrics WHERE version=?", (version,)).fetchone()
        cov = {}
        if row and row[0]:
            try:
                cov = json.loads(row[0])
            except ValueError:
                pass
        cov["resolved"] = dict(src_counts)
        con.execute("UPDATE metrics SET coverage=? WHERE version=?",
                    (json.dumps(cov), version))
        con.commit()
    finally:
        con.close()
