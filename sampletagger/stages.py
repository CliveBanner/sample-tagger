import os
import time
import json
import sqlite3
from multiprocessing import Pool
from .constants import AUDIO_EXTS, DIM
from .db import db_known_set, db_discover_upsert, db_label_update
from .workers import discover_one, label_one, _init_label_worker
from .panns import load_head
import numpy as np
import torch

def gather(root):
    if os.path.isfile(root):
        yield root; return
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                yield os.path.join(dp, f)

def run_relabel_panns(con, args):
    """Reconstruct the raw PANNs label (panns_label/panns_label_conf) for every
    stored embedding WITHOUT decoding any audio.

    CNN14's clipwise output is exactly sigmoid(fc_audioset(embedding)), and the
    2048-d embedding is already in the DB, so the whole library is relabeled with
    a single matrix multiply — no filesystem reads, no librosa, seconds not hours.

    Stores the model's raw output verbatim (527-way AudioSet vocabulary, e.g.
    "Bass drum", "Music", "Water") — no taxonomy mapping, no thresholding:
      panns_label / panns_label_conf  top-1 class + sigmoid score
      panns_topk                      JSON [[label, score], ...] of the top 5
    Top-1 is often the generic "Music" tag, so the top-5 is what carries the
    instrument signal. Collapsing to the instrument taxonomy is a later,
    separately-runnable step.
    """
    from panns_inference import AudioTagging, labels as panns_labels
    TOPK = 5

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    print(f"Loading CNN14 classifier head (fc_audioset) on {device} ...", flush=True)
    at = AudioTagging(checkpoint_path=None, device=device)
    # On CUDA panns_inference wraps the net in DataParallel; the real module
    # (and fc_audioset) then lives under .module.
    model = getattr(at.model, "module", at.model)
    head = model.fc_audioset.eval()

    # Stream embeddings on a separate read-only connection so the cursor isn't
    # disturbed by our batched UPDATEs on `con`.
    rcon = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    total = rcon.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    if args.limit:
        total = min(total, args.limit)
    con.execute("PRAGMA busy_timeout=30000")
    print(f"{total} embeddings -> reconstructing PANNs labels (no audio decode)", flush=True)

    BATCH = args.batch
    n = 0
    pend_paths, pend_vecs = [], []
    t0 = time.time()

    def flush(paths, vecs):
        arr = np.frombuffer(b"".join(vecs), dtype=np.float32).reshape(len(vecs), -1)
        with torch.no_grad():
            clip = torch.sigmoid(head(torch.from_numpy(arr).to(device))).cpu().numpy()
        # top-K class indices per row (unsorted from argpartition, then sorted desc)
        part = np.argpartition(-clip, TOPK, axis=1)[:, :TOPK]
        ts = time.time()
        rows = []
        for r, p in enumerate(paths):
            idx = part[r][np.argsort(-clip[r, part[r]])]
            pairs = [[panns_labels[i], round(float(clip[r, i]), 4)] for i in idx]
            rows.append((pairs[0][0], pairs[0][1], json.dumps(pairs), ts, p))
        if not args.dry_run:
            con.executemany("UPDATE samples SET panns_label=?, panns_label_conf=?, "
                            "panns_topk=?, ts=? WHERE path=?", rows)
            con.commit()

    cur = rcon.execute("SELECT path, vec FROM embeddings")
    for p, v in cur:
        if v is None or len(v) != DIM * 4:
            continue
        pend_paths.append(p)
        pend_vecs.append(v)
        n += 1
        if len(pend_paths) >= BATCH:
            flush(pend_paths, pend_vecs)
            pend_paths, pend_vecs = [], []
            rate = n / (time.time() - t0)
            print(f"  {n}/{total}  {rate:6.0f}/s", flush=True)
        if args.limit and n >= args.limit:
            break
    if pend_paths:
        flush(pend_paths, pend_vecs)
    rcon.close()
    print(f"\nRelabel done: {n} files in {time.time()-t0:.1f}s "
          f"({'dry-run, nothing written' if args.dry_run else 'panns_label updated'})",
          flush=True)



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
    
    # discover new / changed files
    if args.trust_db:
        todo = [p for p in all_files if p not in known]
    else:
        todo = []
        for p in all_files:
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
          f"{len(todo)} new/changed to register.", flush=True)
    
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("Nothing to do."); return
    
    n, errors = 0, 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(discover_one, todo, chunksize=64):
            n += 1
            if r["ok"] and not args.dry_run:
                db_discover_upsert(con, r["path"], r["mtime"], r["size"],
                                   r.get("path_instrument"))
            else:
                errors += 1
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
    
    do_audio = "audio" in classifiers
    do_panns = "panns" in classifiers
    
    # build todo: files missing at least one requested classifier result
    where_parts = []
    if "panns" in classifiers and "panns" not in redo_set:
        where_parts.append("panns_instrument IS NULL")
    if "audio" in classifiers and "audio" not in redo_set:
        where_parts.append("audio_instrument IS NULL")
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
              initargs=(do_audio, do_panns)) as pool:
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
    

