import os
import time
import json
import sqlite3
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import joblib

from .export import get_class_set, get_latest_features, load_labels, load_label_sets, load_ml_cfg


def build_dataset(cache_path, db, classes, cfg):
    """Multi-label dataset: Y is a multi-hot matrix over `classes`.
    Human rows use their full label set (sample_labels); weak rows are a single
    positive. Row weights: single=1.0, bulk=bulk_weight, weak=weak_weight.

    NEGATIVES POLICY: every class absent from a row's label set is a hard negative
    for that class's head. This assumes labelers tag ALL clearly-applicable classes
    (see docs/taxonomy.md), which holds for correlated pairs the labeler thinks
    about (guitar+synth) but not exhaustively. Bulk/weak rows carry low weight, so
    their false negatives are cheap; single-label human rows are the ones where a
    missing secondary actually teaches the model a wrong negative."""
    print(f"Loading {cache_path}...")
    data = np.load(cache_path, allow_pickle=True)
    X_all = data['X']
    paths = data['paths']
    # Labels read live from the DB (the .npz embedding cache can be stale on labels).
    lbl = load_labels(db, paths)
    sets = load_label_sets(db)
    human = lbl['human']
    weak_path = lbl['weak_path']
    weak_panns = lbl['weak_panns']
    label_source = lbl['label_source']
    is_val = lbl['is_val']

    wl_map = cfg.get("weak_label_map", {})
    weak_weight = cfg.get("weak_weight", 0.2)
    bulk_weight = cfg.get("bulk_weight", 0.5)

    cls_idx = {c: j for j, c in enumerate(classes)}

    rows, Y, w, single_rows = [], [], [], []
    val_rows, Y_val = [], []

    def multi_hot(labels):
        y = np.zeros(len(classes), dtype=np.int8)
        hit = False
        for l in labels:
            j = cls_idx.get(l)
            if j is not None:
                y[j] = 1
                hit = True
        return y if hit else None

    for i in range(len(paths)):
        h = human[i]
        if is_val[i] == 1:
            if h:
                y = multi_hot(sets.get(paths[i], [h]))
                if y is not None:
                    val_rows.append(i)
                    Y_val.append(y)
            continue

        if h:
            y = multi_hot(sets.get(paths[i], [h]))
            if y is None:
                continue
            rows.append(i)
            Y.append(y)
            if label_source[i] == "single":
                w.append(1.0)
                single_rows.append(len(rows) - 1)
            else:
                w.append(bulk_weight)
        else:
            w_l = weak_path[i]
            if not w_l or w_l not in cls_idx:
                w_l = wl_map.get(weak_panns[i], weak_panns[i])
            w_l = wl_map.get(w_l, w_l)
            if w_l and w_l in cls_idx:
                rows.append(i)
                y = np.zeros(len(classes), dtype=np.int8)
                y[cls_idx[w_l]] = 1
                Y.append(y)
                w.append(weak_weight)

    X = X_all[rows] if rows else np.zeros((0, X_all.shape[1]), np.float32)
    X_v = X_all[val_rows] if val_rows else np.zeros((0, X_all.shape[1]), np.float32)
    return (X, np.array(Y, dtype=np.int8), np.array(w),
            np.array(single_rows, dtype=np.int64),
            X_v, np.array(Y_val, dtype=np.int8), X_all)


def fit_ovr(X, Y, w, classes, verbose=True):
    """One binary logistic head per class. Returns stacked (W, b) so predict is a
    single matmul + sigmoid; classes with no positives get a -inf head (prob 0)."""
    n_feat = X.shape[1]
    W = np.zeros((len(classes), n_feat), dtype=np.float32)
    b = np.full(len(classes), -100.0, dtype=np.float32)
    for j, c in enumerate(classes):
        y = Y[:, j]
        pos = int(y.sum())
        if pos < 2 or pos > len(y) - 2:
            if verbose:
                print(f"  {c:14s} skipped ({pos} positives)")
            continue
        clf = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
        clf.fit(X, y, sample_weight=w)
        W[j] = clf.coef_[0]
        b[j] = clf.intercept_[0]
        if verbose:
            print(f"  {c:14s} {pos} positives")
    return W, b


def predict_probs(X, W, b, batch=8192):
    out = np.empty((X.shape[0], W.shape[0]), dtype=np.float32)
    for s in range(0, X.shape[0], batch):
        z = X[s:s + batch] @ W.T + b
        out[s:s + batch] = 1.0 / (1.0 + np.exp(-z))
    return out


def _report(Y_true, probs, classes, threshold):
    pred = (probs >= threshold).astype(np.int8)
    # guarantee at least the top-1 label so empty predictions don't zero every class
    top1 = probs.argmax(axis=1)
    pred[np.arange(len(pred)), top1] |= (probs[np.arange(len(pred)), top1] > 0)
    print(classification_report(Y_true, pred, target_names=classes, zero_division=0))
    return classification_report(Y_true, pred, target_names=classes,
                                 zero_division=0, output_dict=True)

def calibrate_thresholds(Y_v, probs_v, classes, target, fallback, min_support=10):
    thresholds = {}
    for j, c in enumerate(classes):
        y_true = Y_v[:, j]
        p = probs_v[:, j]
        support = y_true.sum()
        if support < min_support:
            thresholds[c] = fallback
            continue
            
        best_t = fallback
        for t in np.arange(0.30, 0.99, 0.02):
            pred = (p >= t)
            tp = (pred & (y_true == 1)).sum()
            fp = (pred & (y_true == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if precision >= target:
                best_t = float(t)
                break
        thresholds[c] = float(round(best_t, 2))
    return thresholds


def run_train(args):
    db_dir = os.path.dirname(os.path.abspath(args.db))
    features_path = get_latest_features(db_dir)
    if not features_path:
        print("No features exported. Run `sample-tagger-ml export` first.")
        return

    cfg = load_ml_cfg(db_dir)
    classes = sorted(get_class_set(db_dir))
    if not classes:
        print("No classes found in labels.db")
        return
    print(f"Taxonomy: {len(classes)} classes (multi-label, one-vs-rest)")

    X, Y, w, single_rows, X_v, Y_v, X_all = build_dataset(features_path, args.db, classes, cfg)
    print(f"Training set: {len(X)} rows ({len(single_rows)} single human gold)")

    print("Fitting per-class heads...")
    t0 = time.time()
    W, b = fit_ovr(X, Y, w, classes)
    print(f"Fitted in {time.time()-t0:.2f}s")

    threshold = cfg.get("conf_threshold", 0.6)
    target_precision = cfg.get("target_precision", 0.9)
    macro_f1, per_class_f1 = None, {}
    threshold_dict = {c: threshold for c in classes}
    thr_vec = np.full(len(classes), threshold, dtype=np.float32)

    if len(X_v) > 0:
        print(f"\n--- Validation Set Evaluation ({len(X_v)} held-out, per-class binary F1) ---")
        probs_v = predict_probs(X_v, W, b)
        print("Pre-calibration (flat threshold):")
        _report(Y_v, probs_v, classes, threshold)
        
        print(f"\nCalibrating thresholds (target_precision={target_precision})...")
        threshold_dict = calibrate_thresholds(Y_v, probs_v, classes, target_precision, threshold)
        thr_vec = np.array([threshold_dict[c] for c in classes], dtype=np.float32)
        
        print("Post-calibration (per-class threshold):")
        rep = _report(Y_v, probs_v, classes, thr_vec)
        macro_f1 = rep["macro avg"]["f1-score"]
        per_class_f1 = {k: round(v["f1-score"], 4) for k, v in rep.items()
                        if isinstance(v, dict) and k in classes}

    model_version = f"ovr_v{int(time.time())}"
    out_path = os.path.join(db_dir, "models", "head.joblib")
    joblib.dump({
        "W": W, "b": b,
        "classes": classes,
        "version": model_version,
        "multi_label": True,
        "thresholds": threshold_dict,
    }, out_path)
    print(f"Saved {out_path} ({model_version})")

    probs_all = predict_probs(X_all, W, b)
    coverage = int(np.sum((probs_all >= thr_vec).any(axis=1)))

    con = sqlite3.connect(args.db, timeout=30)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS metrics (version TEXT PRIMARY KEY, ts REAL, "
                    "val_n INTEGER, macro_f1 REAL, per_class_f1 TEXT, coverage TEXT, notes TEXT)")
        con.execute("INSERT OR REPLACE INTO metrics (version, ts, val_n, macro_f1, per_class_f1, coverage, notes) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (model_version, time.time(), len(X_v), macro_f1,
                     json.dumps(per_class_f1),
                     json.dumps({"thresholds": threshold_dict, "conf_ge_threshold": coverage,
                                 "total": len(X_all)}),
                     f"train={len(X)} single={len(single_rows)} multi_label=1"))
        con.commit()
    finally:
        con.close()

    # K-fold CV on the single-human rows (small but honest sanity check)
    if len(single_rows) > 20:
        print("Running K-Fold CV on single human labels (per-class binary F1)...")
        Xs, Ys = X[single_rows], Y[single_rows]
        pred = np.zeros_like(Ys, dtype=np.float32)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in kf.split(Xs):
            Wc, bc = fit_ovr(Xs[tr], Ys[tr], np.ones(len(tr)), classes, verbose=False)
            pred[te] = predict_probs(Xs[te], Wc, bc)
        _report(Ys, pred, classes, thr_vec)
    else:
        print("Not enough single human labels for CV (<20).")
