#!/usr/bin/env python3
"""
similar.py — find samples that sound like a given one (CLI).

  ./venv/bin/python similar.py "Be Loyal"            # match by filename substring
  ./venv/bin/python similar.py -k 30 --db samples.db "kick 01"
"""
import argparse
import os
from simlib import SimIndex, fetch_meta, DEFAULT_DB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="exact path or filename substring")
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()

    ix = SimIndex(args.db)
    n = ix.load()
    matched, hits = ix.neighbors(args.query, args.k)
    if matched is None:
        print(f"no sample matching {args.query!r} (index has {n} embeddings)")
        return
    meta = fetch_meta(args.db, [p for p, _ in hits])
    print(f"query: {matched}\n({n} embeddings indexed)\n")
    print(f"{'sim':>5s}  {'instr':7s} {'type':7s} {'bpm':>4s} {'key':>4s}  file")
    for p, score in hits:
        m = meta.get(p, {})
        print(f"{score:5.3f}  {str(m.get('instrument','')):7s} "
              f"{str(m.get('sample_type','')):7s} {str(m.get('bpm') or ''):>4s} "
              f"{str(m.get('key') or ''):>4s}  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
