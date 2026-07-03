import os
import numpy as np
from sklearn.metrics import classification_report
import joblib

from .export import get_latest_features, load_label_sets, load_ml_cfg
from .train import predict_probs


def run_report(args):
    db_dir = os.path.dirname(os.path.abspath(args.db))
    features_path = get_latest_features(db_dir)
    model_path = os.path.join(db_dir, "models", "head.joblib")

    if not features_path or not os.path.exists(model_path):
        print("Missing features or model.")
        return

    saved = joblib.load(model_path)
    if not saved.get("multi_label"):
        print("Old single-label model found — retrain first (`sample-tagger-ml train`).")
        return
    W, b = saved["W"], saved["b"]
    classes = list(saved["classes"])
    cls_idx = {c: j for j, c in enumerate(classes)}

    print(f"Loading {features_path}...")
    data = np.load(features_path, allow_pickle=True)
    X_all = data['X']
    paths = data['paths']
    # Label sets read live from the DB (npz label columns go stale).
    sets = load_label_sets(args.db)

    rows, Y = [], []
    for i, p in enumerate(paths):
        labels = sets.get(p)
        if not labels:
            continue
        y = np.zeros(len(classes), dtype=np.int8)
        hit = False
        for l in labels:
            j = cls_idx.get(l)
            if j is not None:
                y[j] = 1
                hit = True
        if hit:
            rows.append(i)
            Y.append(y)

    if not rows:
        print("No human label sets found to report on.")
        return

    threshold = load_ml_cfg(db_dir).get("conf_threshold", 0.5)
    Y = np.array(Y, dtype=np.int8)
    probs_h = predict_probs(X_all[rows], W, b)
    pred = (probs_h >= threshold).astype(np.int8)

    print(f"\n--- Per-class report (all {len(rows)} human-labeled files, "
          f"threshold {threshold}) ---")
    print(classification_report(Y, pred, target_names=classes, zero_division=0))

    print("\n--- Coverage vs Threshold (top-1 score, all data) ---")
    max_probs = predict_probs(X_all, W, b).max(axis=1)
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        cov = np.mean(max_probs >= t) * 100
        print(f"Threshold {t:.2f}: {cov:.1f}% auto-labeled")
