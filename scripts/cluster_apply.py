#!/usr/bin/env python3
"""
Apply LLM cluster label assignments to samples.db.

Input JSON: list of {"id": <cluster_id>, "label": "<instrument>", "confidence": "high|med|low"}
Only writes to samples that have no human_instrument yet.
Skips entries where label is null/empty/"skip"/"mixed".

Usage:
    python3 scripts/cluster_apply.py [--db samples.db] [--input cluster_labels.json] \
        [--min-confidence med] [--dry-run]
"""
import argparse, json, sqlite3, sys

CONF_RANK = {"high": 3, "med": 2, "low": 1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="samples.db")
    ap.add_argument("--input", default="cluster_labels.json")
    ap.add_argument("--min-confidence", default="med", choices=["low","med","high"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.input) as f:
        assignments = json.load(f)

    min_rank = CONF_RANK[args.min_confidence]

    con = sqlite3.connect(args.db)

    total_written = 0
    skipped_conf = 0
    skipped_label = 0

    for entry in assignments:
        cid = entry["id"]
        label = (entry.get("label") or "").strip().lower()
        conf = (entry.get("confidence") or "low").strip().lower()

        if not label or label in ("skip", "mixed", "?", ""):
            skipped_label += 1
            continue

        if CONF_RANK.get(conf, 0) < min_rank:
            skipped_conf += 1
            continue

        if args.dry_run:
            n = con.execute(
                "SELECT COUNT(*) FROM samples WHERE cluster_id=? AND human_instrument IS NULL",
                (cid,)
            ).fetchone()[0]
            print(f"[dry] cluster {cid}: would write '{label}' ({conf}) to {n} samples")
            total_written += n
        else:
            cur = con.execute("""
                UPDATE samples
                SET human_instrument=?, label_source='llm'
                WHERE cluster_id=? AND human_instrument IS NULL
            """, (label, cid))
            n = cur.rowcount
            total_written += n
            if n:
                print(f"cluster {cid}: {n} × {label} ({conf})")

    if not args.dry_run:
        con.commit()

    print(f"\n{'[dry] would write' if args.dry_run else 'wrote'} "
          f"{total_written:,} samples across {len(assignments)-skipped_label-skipped_conf} clusters",
          file=sys.stderr)
    if skipped_conf:
        print(f"skipped {skipped_conf} entries below --min-confidence={args.min_confidence}",
              file=sys.stderr)
    if skipped_label:
        print(f"skipped {skipped_label} entries with no/skip label", file=sys.stderr)

if __name__ == "__main__":
    main()
