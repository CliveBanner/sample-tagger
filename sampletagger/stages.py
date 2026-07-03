import os
import time
from multiprocessing import Pool
from .constants import AUDIO_EXTS
from .db import db_known_set, db_discover_upsert, db_label_update
from .workers import discover_one, label_one, _init_label_worker

def gather(root):
    if os.path.isfile(root):
        yield root; return
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                yield os.path.join(dp, f)

def run_discover(con, args, t0):
    filelist_cache = args.db + ".filelist"
    if not args.no_cache and os.path.exists(filelist_cache):
        print(f"Loading file list from cache ({filelist_cache}) ...", flush=True)
        with open(filelist_cache) as f:
            all_files = [l.rstrip("\n") for l in f if l.strip()]
    else:
        print(f"Walking {args.path} ...", flush=True)
        all_files = list(gather(args.path))
        if not args.dry_run:
            with open(filelist_cache, "w") as f:
                f.write("\n".join(all_files) + "\n")
    
    known = db_known_set(con) if con else {}
    found_set = set(all_files)
    
    # mark files that disappeared from disk
    missing = [p for p, (_, _, st) in known.items()
               if p not in found_set and st != "missing"]
    if missing and not args.dry_run:
        con.executemany("UPDATE samples SET status=\'missing\', ts=? WHERE path=?",
                        [(time.time(), p) for p in missing])
        con.commit()
        print(f"Marked {len(missing)} files as missing.", flush=True)
    
    # discover new / changed files. With --limit, stop comparing once we have
    # enough — the mtime check stats every known file (slow on the FUSE mount).
    if args.trust_db:
        todo = [p for p in all_files if p not in known]
        if args.limit:
            todo = todo[:args.limit]
    else:
        todo = []
        for p in all_files:
            if args.limit and len(todo) >= args.limit:
                break
            if p not in known:
                todo.append(p)
            else:
                mtime, size, _ = known[p]
                try:
                    st = os.stat(p)
                    if (round(st.st_mtime, 3), st.st_size) != (round(mtime, 3), size):
                        todo.append(p)
                except OSError:
                    pass

    print(f"{len(all_files)} files on disk, {len(missing)} missing, "
          f"{len(todo)} new/changed to register"
          f"{' (stopped at --limit)' if args.limit and len(todo) >= args.limit else ''}.",
          flush=True)
    if not todo:
        print("Nothing to do."); return
    
    n, errors = 0, 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(discover_one, todo, chunksize=64):
            n += 1
            if not r["ok"]:
                errors += 1
            elif not args.dry_run:
                db_discover_upsert(con, r["path"], r["mtime"], r["size"],
                                   r.get("path_instrument"))
            if con and n % 500 == 0:
                con.commit()
            if n % 2000 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  {rate:.0f}/s  err={errors}", flush=True)
    if con:
        con.commit()
    print(f"\nDiscovery done: {n} registered, {errors} errors, "
          f"{(time.time()-t0):.1f}s", flush=True)
    

def run_label(con, args, t0):
    classifiers = {c.strip().lower() for c in args.classifiers.split(",") if c.strip()}
    redo_set = ({c.strip().lower() for c in args.redo.split(",") if c.strip()}
                if args.redo else set())
    if "all" in redo_set:
        redo_set = set(classifiers)
    
    do_panns = "panns" in classifiers

    # build todo: files missing at least one requested classifier result
    where_parts = []
    if "panns" in classifiers and "panns" not in redo_set:
        where_parts.append("panns_instrument IS NULL")
    if "path" in classifiers and "path" not in redo_set:
        where_parts.append("path_instrument IS NULL")
    
    if where_parts or redo_set:
        where = ("(" + " OR ".join(where_parts) + ")" if where_parts else "1=1")
        todo = [r[0] for r in con.execute(
            f"SELECT path FROM samples WHERE status != \'missing\' AND ({where})")]
    else:
        todo = []
    
    print(f"Label stage  classifiers={sorted(classifiers)}  redo={sorted(redo_set)}", flush=True)
    print(f"{len(todo)} files to process on {args.workers} workers.", flush=True)
    
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("Nothing to do."); return
    
    n, errors = 0, 0
    with Pool(args.workers, initializer=_init_label_worker,
              initargs=(do_panns,)) as pool:
        for result in pool.imap_unordered(label_one, todo, chunksize=8):
            n += 1
            if result.get("status") == "error":
                errors += 1
            if not args.dry_run:
                db_label_update(con, result["path"], result, redo_set)
                if n % 200 == 0:
                    con.commit()
            if n % 200 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                eta = (len(todo) - n) / rate if rate else 0
                print(f"  {n}/{len(todo)}  {rate:5.1f}/s  eta {eta/60:5.1f}m  err={errors}",
                      flush=True)
    if con:
        con.commit()
    print(f"\nLabeling done: {n} processed, {errors} errors, "
          f"{(time.time()-t0)/60:.1f}m", flush=True)
    if con:
        print(f"index: {args.db}", flush=True)
    

