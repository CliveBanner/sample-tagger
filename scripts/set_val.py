import sqlite3
import os

def main():
    if not os.path.exists("gold_candidates.txt"):
        print("gold_candidates.txt not found. Run sample_gold.py first.")
        return
        
    with open("gold_candidates.txt") as f:
        paths = [l.strip() for l in f if l.strip()]
        
    if not paths:
        print("No paths in gold_candidates.txt")
        return
        
    con = sqlite3.connect("samples.db")
    qs = ",".join("?" * len(paths))
    res = con.execute(f"UPDATE samples SET is_val=1 WHERE path IN ({qs}) AND label_source='single'", paths)
    con.commit()
    print(f"Set is_val=1 on {res.rowcount} out of {len(paths)} gold candidates.")

if __name__ == "__main__":
    main()
