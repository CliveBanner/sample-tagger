#!/usr/bin/env python3
"""
Dump cluster summaries for LLM-assisted labeling.

Usage:
    python3 scripts/cluster_dump.py [--db samples.db] [--min-n 10] [--top 12] \
        [--unlabeled-only] [--output cluster_dump.json]
"""
import argparse, json, sqlite3, os, sys
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="samples.db")
    ap.add_argument("--min-n", type=int, default=10, help="skip clusters smaller than this")
    ap.add_argument("--top", type=int, default=12, help="member paths to include per cluster")
    ap.add_argument("--unlabeled-only", action="store_true",
                    help="skip clusters where all members already have human_instrument")
    ap.add_argument("--output", default="cluster_dump.json")
    ap.add_argument("--max-clusters", type=int, default=0, help="cap (0=all)")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    # one row per cluster, ordered by size descending (big clusters first = most coverage)
    cluster_ids = [r[0] for r in con.execute("""
        SELECT cluster_id
        FROM samples
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
        HAVING COUNT(*) >= ?
        ORDER BY COUNT(*) DESC
    """, (args.min_n,)).fetchall()]

    if args.max_clusters:
        cluster_ids = cluster_ids[:args.max_clusters]

    out = []
    for cid in cluster_ids:
        rows = con.execute("""
            SELECT path, cluster_d, model_instrument, human_instrument
            FROM samples WHERE cluster_id=?
            ORDER BY cluster_d ASC
        """, (cid,)).fetchall()

        total = len(rows)
        unlabeled = [r for r in rows if not r["human_instrument"]]
        n_unlab = len(unlabeled)

        if args.unlabeled_only and n_unlab == 0:
            continue

        model_labels = [r["model_instrument"] for r in rows if r["model_instrument"]]
        counts = Counter(model_labels)
        dominant = counts.most_common(1)[0] if counts else (None, 0)
        agreement = dominant[1] / len(model_labels) if model_labels else 0

        # use unlabeled members for path sample (they're what we'd label)
        sample_pool = unlabeled if unlabeled else list(rows)
        top_paths = [r["path"] for r in sample_pool[:args.top]]
        # show last 2-3 path segments to keep output readable
        top_names = ["/".join(p.split("/")[-3:]) for p in top_paths]

        out.append({
            "id": cid,
            "n": total,
            "n_unlab": n_unlab,
            "agreement": round(agreement, 3),
            "model_label": dominant[0],
            "paths": top_names,
        })

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {len(out)} clusters → {args.output}", file=sys.stderr)
    print(f"total unlabeled samples covered: "
          f"{sum(c['n_unlab'] for c in out):,}", file=sys.stderr)

if __name__ == "__main__":
    main()
