import os
import time
import sqlite3
import numpy as np
import joblib

from .export import get_latest_features, load_labels, load_ml_cfg

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
    clf = saved["model"]
    classes = saved["classes"]
    version = saved["version"]
    
    cfg = load_ml_cfg(db_dir)
    conf_threshold = cfg.get("conf_threshold", 0.6)
    
    print("Predicting...")
    t0 = time.time()
    probs = clf.predict_proba(X_all)
    top1_idx = np.argmax(probs, axis=1)
    
    # Extract top 2 for margin calculation (A4)
    # Using np.partition to efficiently get top 2
    if probs.shape[1] > 1:
        part = np.partition(probs, -2, axis=1)
        top1_prob = part[:, -1]
        top2_prob = part[:, -2]
        margins = top1_prob - top2_prob
    else:
        margins = np.ones(len(probs))
    
    model_inst = classes[top1_idx]
    model_conf = probs[np.arange(len(probs)), top1_idx]
    
    print(f"Predicted {len(X_all)} in {time.time()-t0:.2f}s. Writing to DB...")
    
    # Resolve final instrument
    # human_instrument > (model_instrument if model_conf >= threshold) > path weak > none
    updates = []
    
    for i in range(len(paths)):
        m_inst = model_inst[i]
        m_conf = float(model_conf[i])
        h_inst = human[i]
        w_path = weak_path[i]
        
        final_inst = None
        source = "none"
        
        if h_inst:
            final_inst = h_inst
            source = "human"
        elif m_conf >= conf_threshold:
            final_inst = m_inst
            source = "model"
        elif w_path:
            final_inst = w_path
            source = "path"
            
        updates.append((m_inst, m_conf, version, float(margins[i]), final_inst, source, paths[i]))
        
    con = sqlite3.connect(args.db, timeout=10)
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
                instrument = ?,
                source = ?
            WHERE path = ?
        """, [(u[0], u[1], u[2], u[3], u[4], u[5], u[6]) for u in updates])
        con.commit()
        print(f"Updated {len(updates)} rows in DB.")
    finally:
        con.close()
