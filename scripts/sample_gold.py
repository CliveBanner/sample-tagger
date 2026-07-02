import sqlite3
import random
import os

def main():
    random.seed(42)
    db_path = "samples.db"
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return
        
    con = sqlite3.connect(db_path)
    
    # 1. Stratified by predicted class (model_instrument)
    rows = con.execute("SELECT path, model_instrument FROM samples WHERE (human_instrument IS NULL OR human_instrument='') AND model_instrument IS NOT NULL").fetchall()
    
    by_cls = {}
    for p, inst in rows:
        by_cls.setdefault(inst, []).append(p)
        
    picked = []
    for inst, paths in by_cls.items():
        k = min(23, len(paths))
        picked += random.sample(paths, k)
        
    # 2. Plus a slice from source='none'
    none_rows = con.execute("SELECT path FROM samples WHERE (human_instrument IS NULL OR human_instrument='') AND source = 'none'").fetchall()
    none_paths = [r[0] for r in none_rows]
    k_none = min(50, len(none_paths))
    picked += random.sample(none_paths, k_none)
    
    out_file = "gold_candidates.txt"
    with open(out_file, "w") as f:
        for p in picked:
            f.write(f"{p}\n")
            
    print(f"Selected {len(picked)} candidates to {out_file}")

if __name__ == "__main__":
    main()
