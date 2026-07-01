import argparse
import os
import time
from multiprocessing import cpu_count
from .db import db_connect
from .stages import run_discover, run_label, run_relabel_panns
from .sim import sim_cmd

def main():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--db", default=os.path.join(os.getcwd(), "samples.db"), help="SQLite database path")
    parent.add_argument("-j", "--workers", type=int, default=max(1, cpu_count() - 1))
    parent.add_argument("--limit", type=int, default=0)
    parent.add_argument("-n", "--dry-run", action="store_true")

    ap = argparse.ArgumentParser(
        description="Two-stage audio sample classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_discover = sub.add_parser("discover", parents=[parent], help="Stat sample library")
    p_discover.add_argument("path", help="root of the sample library")
    p_discover.add_argument("--no-cache", action="store_true")
    p_discover.add_argument("--trust-db", action="store_true")

    p_label = sub.add_parser("label", parents=[parent], help="Extract features and tag")
    p_label.add_argument("--classifiers", default="panns")
    p_label.add_argument("--redo", default="")

    p_relabel = sub.add_parser("relabel-panns", parents=[parent], help="Reconstruct raw PANNs label")
    p_relabel.add_argument("--gpu", action="store_true")
    p_relabel.add_argument("--batch", type=int, default=4096)

    p_sim = sub.add_parser("sim", parents=[parent], help="Find similar samples")
    p_sim.add_argument("query", help="exact path or filename substring")
    p_sim.add_argument("-k", type=int, default=20)

    args = ap.parse_args()

    if args.cmd == "sim":
        sim_cmd(args)
        return

    con = None if (args.dry_run and args.cmd != "relabel-panns") else db_connect(args.db)
    t0 = time.time()

    if args.cmd == "relabel-panns":
        run_relabel_panns(con, args)
    elif args.cmd == "discover":
        run_discover(con, args, t0)
    elif args.cmd == "label":
        run_label(con, args, t0)

if __name__ == "__main__":
    main()
