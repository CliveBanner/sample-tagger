import os
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from .export import get_class_set, get_latest_features, load_labels, load_ml_cfg

def build_dataset(cache_path, db, valid_classes, cfg):
    print(f"Loading {cache_path}...")
    data = np.load(cache_path, allow_pickle=True)
    X_all = data['X']
    paths = data['paths']
    # Labels read live from the DB (the .npz embedding cache can be stale on labels).
    lbl = load_labels(db, paths)
    human = lbl['human']
    weak_path = lbl['weak_path']
    weak_panns = lbl['weak_panns']
    label_source = lbl['label_source']
    is_val = lbl['is_val']
    
    wl_map = cfg.get("weak_label_map", {})
    weak_weight = cfg.get("weak_weight", 0.2)
    bulk_weight = cfg.get("bulk_weight", 0.5)
    
    valid_set = set(valid_classes)
    
    X_train, y_train, w_train, train_paths = [], [], [], []
    X_human, y_human = [], []  # For CV
    X_val, y_val = [], []      # For held-out eval
    
    for i in range(len(paths)):
        h = human[i]
        src = label_source[i]
        
        if is_val[i] == 1:
            if h and h in valid_set:
                X_val.append(X_all[i])
                y_val.append(h)
            continue
            
        if h and h in valid_set:
            is_single = (src == "single")
            
            if is_single:
                X_human.append(X_all[i])
                y_human.append(h)
                w_train.append(1.0)
            else:
                w_train.append(bulk_weight)
                
            X_train.append(X_all[i])
            y_train.append(h)
            train_paths.append(paths[i])
        else:
            # Try weak labels
            w_l = weak_path[i]
            if not w_l or w_l not in valid_set:
                w_l = weak_panns[i]
            
            w_l = wl_map.get(w_l, w_l)
            if w_l and w_l in valid_set:
                X_train.append(X_all[i])
                y_train.append(w_l)
                w_train.append(weak_weight)
                train_paths.append(paths[i])
                
    return (np.array(X_train), np.array(y_train), np.array(w_train), np.array(train_paths),
            np.array(X_human), np.array(y_human),
            np.array(X_val), np.array(y_val), X_all)

def run_train(args):
    db_dir = os.path.dirname(os.path.abspath(args.db))
    features_path = get_latest_features(db_dir)
    if not features_path:
        print("No features exported. Run `sample-tagger-ml export` first.")
        return
        
    cfg = load_ml_cfg(db_dir)
    valid_classes = get_class_set(db_dir)
    if not valid_classes:
        print("No classes found in labels.db")
        return
        
    print(f"Taxonomy: {len(valid_classes)} classes")
    
    X, y, w, p, X_h, y_h, X_v, y_v, X_all = build_dataset(features_path, args.db, valid_classes, cfg)
    print(f"Training set: {len(X)} samples ({len(X_h)} single human ground-truth)")
    

    print("Fitting final model...")
    t0 = time.time()
    clf = LogisticRegression(solver='lbfgs', 
                             max_iter=1000, class_weight='balanced')
    clf.fit(X, y, sample_weight=w)
    print(f"Fitted in {time.time()-t0:.2f}s")
    
    if len(X_v) > 0:
        print(f"\n--- Validation Set Evaluation ({len(X_v)} held-out samples) ---")
        y_pred_v = clf.predict(X_v)
        print(classification_report(y_v, y_pred_v, zero_division=0))
    
    model_version = f"lr_v{int(time.time())}"
    out_path = os.path.join(db_dir, "models", "head.joblib")
    
    joblib.dump({
        "model": clf,
        "classes": clf.classes_,
        "version": model_version,
        "valid_classes": valid_classes
    }, out_path)
    
    print(f"Saved {out_path} ({model_version})")

    # Calculate metrics
    macro_f1 = 0.0
    per_class_f1 = {}
    if len(X_v) > 0:
        rep = classification_report(y_v, y_pred_v, zero_division=0, output_dict=True)
        macro_f1 = rep.get("macro avg", {}).get("f1-score", 0.0)
        per_class_f1 = {k: v["f1-score"] for k, v in rep.items() if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg")}
    
    threshold = cfg.get("conf_threshold", 0.6)
    coverage = 0
    if len(X_all) > 0:
        y_prob = clf.predict_proba(X_all)
        coverage = int(np.sum(y_prob.max(axis=1) >= threshold))
        
    metrics = {
        "version": model_version,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        f"coverage@{threshold}": coverage,
        "total_files": len(X_all)
    }
    import json
    metrics_path = os.path.join(db_dir, "models", "metrics.jsonl")
    with open(metrics_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
        

    if len(X_h) > 10:
        print("Running K-Fold CV on single human labels...")
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred_cv = np.empty_like(y_h)
            for train_idx, test_idx in skf.split(X_h, y_h):
                clf_cv = LogisticRegression(solver='lbfgs', 
                                            max_iter=1000, class_weight='balanced')
                clf_cv.fit(X_h[train_idx], y_h[train_idx])
                y_pred_cv[test_idx] = clf_cv.predict(X_h[test_idx])
            print(classification_report(y_h, y_pred_cv, zero_division=0))
        except ValueError as e:
            print(f"Skipping CV: {e}")
    else:
        print("Not enough single human labels for CV (<10).")
