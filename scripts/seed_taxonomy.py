#!/usr/bin/env python3
"""
Seed labels.db with the optimized 17-label instrument taxonomy and run a
path->label WEAK pass into samples.path_instrument.

The labels were chosen for embedding-space separability (see chat analysis).
This writes provisional labels to `path_instrument` (the weak-signal column used
as weak supervision in training and shown as the "path" hint) — NOT to
human_instrument, so ground truth stays clean.

Ordered = first-match priority (specific before catch-alls).

Usage:
  python3 scripts/seed_taxonomy.py [--db samples.db] [--labels-db labels.db] [--dry-run]
"""
import argparse
import os
import re
import sqlite3
import time

# (label, regex patterns). Order matters: first match wins.
TAXONOMY = [
    ("kick",        [r"\bkick", r"\bbd\b", r"bassdrum", r"\bkik"]),
    ("snare_clap",  [r"\bsnare", r"\bsn[rd]?\b", r"\bsd\b", r"\bclap", r"\bsnap", r"rimshot", r"\brim\b"]),
    ("hats_cymbals",[r"hi.?hat", r"\bhat\b", r"\bhh\b", r"\bhats\b", r"cymbal", r"\bcrash", r"\bride\b", r"splash"]),
    ("tom",         [r"\btom\b", r"\btoms\b"]),
    ("perc",        [r"\bperc", r"conga", r"bongo", r"tabla", r"djembe", r"shaker", r"tambou", r"cowbell", r"clave"]),
    ("piano_keys",  [r"piano", r"grand", r"steinway", r"rhodes", r"wurli", r"clav", r"harpsichord", r"\bepiano"]),
    ("organ",       [r"organ", r"hammond", r"\bb3\b"]),
    ("mallet",      [r"marimba", r"xylophon", r"vibraphon", r"glocken", r"kalimba", r"\bbell", r"chime", r"\bvibes\b"]),
    ("guitar",      [r"guitar", r"\bgtr", r"\bstrat", r"\btele\b"]),
    ("strings",     [r"string", r"violin", r"cello", r"viola", r"orchestr", r"pizz", r"\bharp\b"]),
    ("brass",       [r"brass", r"trumpet", r"trombone", r"\bsax", r"\bhorn", r"\btuba"]),
    ("winds",       [r"flute", r"clarinet", r"\boboe", r"whistle", r"harmonica", r"didger"]),
    ("vocal",       [r"vocal", r"\bvox\b", r"acapella", r"choir", r"\bvoice", r"\bsing", r"dialog"]),
    ("bass",        [r"\bbass", r"\bsub\b", r"reese", r"\b808"]),
    ("pad",         [r"\bpad"]),
    ("synth",       [r"synth", r"\bsaw\b", r"\bsine", r"\blead", r"\bpluck", r"\barp\b"]),
    ("sfx",         [r"\bfx\b", r"effect", r"riser", r"\bsweep", r"impact", r"whoosh", r"\bnoise", r"\bstab\b", r"foley", r"ambien", r"\bfield", r"nature"]),
]

LABELS = [name for name, _ in TAXONOMY]


def classify(path, compiled):
    p = path.lower()
    for name, pats in compiled:
        if any(rx.search(p) for rx in pats):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="samples.db")
    ap.add_argument("--labels-db", default=None, help="defaults to labels.db next to --db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    labels_db = args.labels_db or os.path.join(os.path.dirname(os.path.abspath(args.db)), "labels.db")
    compiled = [(n, [re.compile(x) for x in pats]) for n, pats in TAXONOMY]

    # 1) seed labels.db (replace taxonomy)
    if not args.dry_run:
        lc = sqlite3.connect(labels_db, timeout=10)
        lc.execute("CREATE TABLE IF NOT EXISTS labels (name TEXT PRIMARY KEY, created_at REAL)")
        lc.execute("DELETE FROM labels")
        lc.executemany("INSERT INTO labels(name,created_at) VALUES(?,?)",
                       [(n, time.time()) for n in LABELS])
        lc.commit(); lc.close()
        print(f"seeded {labels_db} with {len(LABELS)} labels")
    else:
        print(f"[dry] would seed {labels_db} with {len(LABELS)} labels: {', '.join(LABELS)}")

    # 2) path -> path_instrument weak pass
    con = sqlite3.connect(args.db, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    rows = con.execute("SELECT path FROM samples").fetchall()
    updates, counts = [], {n: 0 for n in LABELS}
    matched = 0
    for (p,) in rows:
        lab = classify(p, compiled)
        updates.append((lab, p))
        if lab:
            counts[lab] += 1
            matched += 1
    total = len(rows)
    if not args.dry_run:
        con.executemany("UPDATE samples SET path_instrument=? WHERE path=?", updates)
        con.commit()
    con.close()

    print(f"\npath weak labels: {matched}/{total} matched ({100*matched//max(total,1)}%)"
          f"{' [dry-run, not written]' if args.dry_run else ' written to path_instrument'}")
    for n in LABELS:
        print(f"  {counts[n]:7d}  {n}")


if __name__ == "__main__":
    main()
