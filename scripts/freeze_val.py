import sqlite3
import random
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="samples.db", help="Path to samples.db")
    parser.add_argument("--frac", type=float, default=0.2, help="Fraction to hold out")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible split)")
    args = parser.parse_args()

    random.seed(args.seed)
    con = sqlite3.connect(args.db)

    # Stratify by class: hold out `frac` of EACH class's single labels, so rare
    # classes still get validation coverage (a global random sample would miss them).
    rows = con.execute(
        "SELECT human_instrument, path, is_val FROM samples "
        "WHERE label_source='single' AND human_instrument IS NOT NULL "
        "AND human_instrument!=''").fetchall()
    if not rows:
        print("No single human labels found.")
        return

    by_cls = {}
    for inst, path, isv in rows:
        d = by_cls.setdefault(inst, {"avail": [], "val": 0})
        if isv:
            d["val"] += 1
        else:
            d["avail"].append(path)

    picked = []
    for inst, d in by_cls.items():
        total = len(d["avail"]) + d["val"]
        need = round(total * args.frac) - d["val"]   # top up to the per-class target
        if need > 0 and d["avail"]:
            picked += random.sample(d["avail"], min(need, len(d["avail"])))

    if not picked:
        print("Validation set already at target across all classes. Doing nothing.")
        return

    con.execute("BEGIN TRANSACTION")
    con.executemany("UPDATE samples SET is_val=1 WHERE path=?", [(p,) for p in picked])
    con.commit()
    con.close()

    total_single = sum(len(d["avail"]) + d["val"] for d in by_cls.values())
    print(f"Froze {len(picked)} new validation samples across {len(by_cls)} classes "
          f"(stratified, seed={args.seed}). Single labels total: {total_single}.")

if __name__ == "__main__":
    main()
