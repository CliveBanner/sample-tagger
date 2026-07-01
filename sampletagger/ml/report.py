import os
import sqlite3
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from .export import get_latest_features
from ..config import load_config

def run_report(args):
    db_dir = os.path.dirname(os.path.abspath(args.db))
    features_path = get_latest_features(db_dir)
    model_path = os.path.join(db_dir, "models", "head.joblib")
    
    if not features_path or not os.path.exists(model_path):
        print("Missing features or model.")
        return
        
    print(f"Loading {features_path}...")
    data = np.load(features_path, allow_pickle=True)
    X_all = data['X']
    human = data['human']
    
    saved = joblib.load(model_path)
    clf = saved["model"]
    
    # Filter for human labels
    mask = np.array([bool(h) for h in human])
    X_h = X_all[mask]
    y_h = human[mask]
    
    if len(X_h) == 0:
        print("No human labels found to report on.")
        return
        
    y_pred = clf.predict(X_h)
    
    print("\n--- Classification Report (Human labels only) ---")
    print(classification_report(y_h, y_pred, zero_division=0))
    
    # Coverage vs Threshold
    probs = clf.predict_proba(X_all)
    max_probs = np.max(probs, axis=1)
    
    print("\n--- Coverage vs Threshold (All data) ---")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for t in thresholds:
        cov = np.mean(max_probs >= t) * 100
        print(f"Threshold {t:.2f}: {cov:.1f}% auto-labeled")
