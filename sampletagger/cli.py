import argparse
import os
import sys
import time
from multiprocessing import cpu_count
from .db import db_connect
from .stages import run_discover, run_label, run_relabel_panns
from .sim import sim_cmd

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sim":
        ap = argparse.ArgumentParser(prog="sample-tagger sim")
        ap.add_argument("query", help="exact path or filename substring")
        ap.add_argument("-k", type=int, default=20)
        ap.add_argument("--db", default=os.path.join(os.getcwd(), "samples.db"))
        args = ap.parse_args(sys.argv[2:])
        sim_cmd(args)
        return

    ap = argparse.ArgumentParser(
        description="Two-stage audio sample classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", help="root of the sample library")
    ap.add_argument("--db",
                    default=os.path.join(os.getcwd(), "samples.db"),
                    help="SQLite database path")
    ap.add_argument("-j", "--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--stage", choices=["discover", "label", "relabel-panns"], required=True)
    ap.add_argument("--classifiers", default="panns")
    ap.add_argument("--redo", default="")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--trust-db", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("-n", "--dry-run", action="store_true")
    
    args = ap.parse_args()

    con = None if (args.dry_run and args.stage != "relabel-panns") else db_connect(args.db)
    t0 = time.time()

    if args.stage == "relabel-panns":
        run_relabel_panns(con, args)
        return
    elif args.stage == "discover":
        run_discover(con, args, t0)
    elif args.stage == "label":
        run_label(con, args, t0)

if __name__ == "__main__":
    main()
