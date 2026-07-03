import numpy as np
import torch
import os
import sqlite3
import laion_clap
import librosa
from tqdm import tqdm

_CLAP_MODEL = None
_DEVICE = None
import threading
_CLAP_LOCK = threading.Lock()

def get_clap():
    global _CLAP_MODEL, _DEVICE
    with _CLAP_LOCK:
        if _CLAP_MODEL is None:
            _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[clap] loading model on {_DEVICE}...")
            _CLAP_MODEL = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            _CLAP_MODEL.load_ckpt(os.path.expanduser('~/clap_ckpt/music_audioset_epoch_15_esc_90.14.pt'))
    return _CLAP_MODEL

def embed_paths(paths, db, out_npz):
    model = get_clap()

    # paths/X row alignment must survive resumes: keep the ordered list
    paths_existing, X_existing, existing_paths = [], [], set()
    if os.path.exists(out_npz):
        try:
            data = np.load(out_npz, allow_pickle=True)
            paths_existing = list(data['paths'])
            X_existing = list(data['X'])
            existing_paths = set(paths_existing)
            print(f"[clap] resuming from {len(existing_paths)} existing embeddings.")
        except Exception as e:
            print(f"[clap] failed to load existing npz: {e}")
            
    from ..config import load_config
    cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(db)), "config.json"))
    root = cfg.library_path
    
    X_new = []
    paths_new = []
    
    to_process = [p for p in paths if p not in existing_paths]
    if not to_process:
        print("[clap] all paths already embedded.")
        return
        
    print(f"[clap] embedding {len(to_process)} paths...")
    
    for p in tqdm(to_process):
        full_path = os.path.join(root, p)
        try:
            y, sr = librosa.load(full_path, sr=48000, mono=True, duration=10.0)
            if y.shape[0] == 0:
                continue
                
            if y.shape[0] < 48000:
                y = np.tile(y, int(np.ceil(48000 / y.shape[0])))[:48000]
                
            y = y.reshape(1, -1)
            emb = model.get_audio_embedding_from_data(x=y, use_tensor=False)
            emb = emb[0]
            
            emb = emb / np.linalg.norm(emb)
            
            X_new.append(emb)
            paths_new.append(p)
        except Exception as e:
            pass
            
    if len(X_new) > 0:
        X_final = np.array(X_existing + X_new, dtype=np.float32)
        paths_final = paths_existing + paths_new
        np.savez_compressed(out_npz, X=X_final, paths=np.array(paths_final, dtype=object))
        print(f"[clap] saved {len(X_final)} embeddings to {out_npz}")

def fetch_batch(batch_idx, paths, root):
    import subprocess
    tmp_dir = f"/tmp/clap_stage_{batch_idx}"
    os.makedirs(tmp_dir, exist_ok=True)
    list_path = f"{tmp_dir}/files.txt"
    with open(list_path, "w") as f:
        for p in paths:
            f.write(p + "\n")
    try:
        subprocess.run(["rclone", "copy", "--files-from", list_path, root, tmp_dir], check=True, capture_output=True)
    except FileNotFoundError:
        # fallback to direct copy if rclone not found
        import shutil
        for p in paths:
            src = os.path.join(root, p)
            dst = os.path.join(tmp_dir, p)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
    return tmp_dir, paths

FIXED_LEN = 480000   # 10 s @ 48 kHz — laion_clap pads/crops to this internally anyway,
                     # so pre-padding keeps batched results identical to per-file calls

def _decode_one(job):
    """Pool worker: decode one file to a fixed-length 48 kHz mono array."""
    path_rel, full_path = job
    try:
        y, sr = librosa.load(full_path, sr=48000, mono=True, duration=10.0)
        if y.shape[0] == 0:
            return (path_rel, None)
        if y.shape[0] < 48000:   # tile short one-shots (silence-padding kills timbre)
            y = np.tile(y, int(np.ceil(48000 / y.shape[0])))[:48000]
        if y.shape[0] < FIXED_LEN:
            y = np.pad(y, (0, FIXED_LEN - y.shape[0]))
        return (path_rel, y[:FIXED_LEN].astype(np.float32))
    except Exception:
        return (path_rel, None)


def embed_paths_staged(paths, db, out_npz):
    """Full-library embed: rclone-staged download (prefetch thread) + parallel decode
    pool (webapp `workers` config) + batched model inference in the main process."""
    import concurrent.futures
    import shutil
    import time
    from multiprocessing import Pool

    # ORDER MATTERS: paths and X must stay row-aligned across resumes, so existing
    # paths are kept as a LIST (a set would scramble the pairing on every restart).
    paths_existing, X_existing, existing = [], [], set()
    if os.path.exists(out_npz):
        try:
            data = np.load(out_npz, allow_pickle=True)
            paths_existing = list(data['paths'])
            X_existing = list(data['X'])
            existing = set(paths_existing)
            print(f"[clap] resuming from {len(existing)} existing embeddings.", flush=True)
        except Exception as e:
            print(f"[clap] failed to load existing npz: {e}", flush=True)

    from ..config import load_config
    cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(db)), "config.json"))
    root = cfg.library_path
    workers = max(1, int(getattr(cfg, "workers", 4)))

    to_process = [p for p in paths if p not in existing]
    if not to_process:
        print("[clap] all paths already embedded.", flush=True)
        return

    print(f"[clap] staged embedding {len(to_process)} paths "
          f"({workers} decode workers, batched inference)...", flush=True)
    BATCH_SIZE = 500
    EMBED_BATCH = 16
    CHECKPOINT_EVERY = 2000
    batches = [to_process[i:i + BATCH_SIZE] for i in range(0, len(to_process), BATCH_SIZE)]

    X_new, paths_new = [], []
    errors = 0
    since_ckpt = 0
    t0 = time.time()

    def checkpoint():
        X_final = np.array(X_existing + X_new, dtype=np.float32)
        paths_final = paths_existing + paths_new
        np.savez_compressed(out_npz, X=X_final, paths=np.array(paths_final, dtype=object))

    # decode pool BEFORE the model loads keeps the forked children lightweight
    with Pool(workers) as pool:
        model = get_clap()

        def flush_embed(buf_p, buf_y):
            nonlocal since_ckpt
            if not buf_p:
                return
            emb = model.get_audio_embedding_from_data(x=np.stack(buf_y), use_tensor=False)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            X_new.extend(emb.astype(np.float32))
            paths_new.extend(buf_p)
            since_ckpt += len(buf_p)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_batch, 0, batches[0], root)
            for i in range(len(batches)):
                tmp_dir, batch_paths = future.result()
                if i + 1 < len(batches):
                    future = executor.submit(fetch_batch, i + 1, batches[i + 1], root)

                jobs = [(p, os.path.join(tmp_dir, p)) for p in batch_paths
                        if os.path.exists(os.path.join(tmp_dir, p))]
                errors += len(batch_paths) - len(jobs)
                buf_p, buf_y = [], []
                for p, y in pool.imap_unordered(_decode_one, jobs, chunksize=4):
                    if y is None:
                        errors += 1
                        continue
                    buf_p.append(p)
                    buf_y.append(y)
                    if len(buf_p) >= EMBED_BATCH:
                        flush_embed(buf_p, buf_y)
                        buf_p, buf_y = [], []
                flush_embed(buf_p, buf_y)
                shutil.rmtree(tmp_dir, ignore_errors=True)

                if since_ckpt >= CHECKPOINT_EVERY:
                    checkpoint()
                    since_ckpt = 0
                done = len(paths_new)
                rate = done / max(time.time() - t0, 1)
                eta_h = (len(to_process) - done) / max(rate, 0.01) / 3600
                print(f"[clap] {len(existing) + done}/{len(existing) + len(to_process)}  "
                      f"{rate:.2f}/s  err={errors}  eta {eta_h:.1f}h", flush=True)

    if paths_new:
        checkpoint()
        print(f"[clap] saved {len(X_existing) + len(X_new)} embeddings to {out_npz} "
              f"({errors} errors)", flush=True)


PROMPTS = {
    "kick": ["a kick drum one-shot", "an 808 kick drum sample", "a punchy bass drum hit"],
    "snare_clap": ["a snare drum hit", "a clapping sound", "a snare and clap sample"],
    "hats_cymbals": ["a hi-hat sample", "a crash cymbal hit", "a closed hi-hat", "a ride cymbal"],
    "tom": ["a tom drum hit", "a tom-tom sample", "a floor tom hit"],
    "perc": ["a percussion instrument hit", "a tambourine or shaker", "a bongo or conga hit", "woodblock percussion"],
    "drums": ["a drum loop", "a full drum beat pattern", "a breakbeat loop"],
    "drumhit": ["a single drum hit", "a drum sample"],
    "bass": ["a bass guitar note", "a sub bass synth", "a deep bass sound", "an 808 sub bass"],
    "synth": ["a synthesized instrument sound", "an analog synthesizer", "a synth lead", "a synthesizer arpeggio"],
    "pad": ["a sustained ambient synthesizer pad", "a warm atmospheric pad sound", "a background synth pad", "a lush synthesizer pad"],
    "piano_keys": ["an acoustic piano sound", "an electric piano or keyboard", "a piano chord", "a rhodes keyboard"],
    "organ": ["a hammond organ sound", "a church organ", "an electronic organ pad"],
    "guitar": ["an acoustic guitar strum", "an electric guitar note", "a guitar riff"],
    "strings": ["orchestral strings playing", "a violin or cello sound", "a string section"],
    "brass": ["a trumpet or trombone sound", "a brass section", "a french horn"],
    "winds": ["a flute or saxophone sound", "a woodwind instrument", "a clarinet or oboe"],
    "mallet": ["a vibraphone or marimba", "a glockenspiel or xylophone", "a mallet percussion instrument"],
    "pluck": ["a plucked string instrument", "a harp or koto", "a mandolin or banjo"],
    "vocal": ["a human singing voice", "a vocal chop or loop", "a choir singing", "spoken dialogue"],
    "sfx": ["a sound effect", "a noise sweep or riser", "an explosion or impact sound", "foley or ambient noise"]
}

def text_anchor(class_name):
    if class_name not in PROMPTS:
        return None
    model = get_clap()
    prompts = PROMPTS[class_name]
    embs = model.get_text_embedding(prompts, use_tensor=False)
    mean_emb = np.mean(embs, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)
    return mean_emb

def run_clap_embed(args):
    con = sqlite3.connect(args.db)
    is_full = getattr(args, 'full', False)
    if is_full:
        rows = con.execute("SELECT path FROM samples WHERE status != 'missing'").fetchall()
    else:
        rows = con.execute("SELECT path FROM samples WHERE is_val=1 OR (human_instrument IS NOT NULL AND human_instrument != '')").fetchall()
    con.close()
    
    paths = [r[0] for r in rows]
    db_dir = os.path.dirname(os.path.abspath(args.db))
    out_npz = os.path.join(db_dir, "models", "clap_full.npz" if is_full else "clap_pilot.npz")
    
    if is_full:
        embed_paths_staged(paths, args.db, out_npz)
        
        # Export sidecars for full re-embed
        print("[clap] exporting sidecars for full embed...")
        data = np.load(out_npz, allow_pickle=True)
        # Convert 512-d embeddings to fp16 to save space
        X_fp16 = data['X'].astype(np.float16)
        paths_arr = data['paths']
        base = args.db[:-3] if args.db.endswith(".db") else args.db
        np.save(base + ".clap.npy", X_fp16)
        with open(base + ".clap.paths", "w") as f:
            for p in paths_arr:
                f.write(p + "\n")
        print(f"[clap] saved sidecar {base}.clap.npy")
    else:
        embed_paths(paths, args.db, out_npz)

def build_aligned_datasets(clap_npz, db, classes, cfg):
    from .export import get_latest_features, load_labels, load_label_sets
    panns_npz = get_latest_features(os.path.dirname(os.path.abspath(db)))
    
    data_c = np.load(clap_npz, allow_pickle=True)
    paths_c = data_c['paths']
    X_c_all = data_c['X']
    
    data_p = np.load(panns_npz, allow_pickle=True)
    paths_p = data_p['paths']
    X_p_all = data_p['X']
    p_idx = {p: i for i, p in enumerate(paths_p)}
    
    valid = []
    X_c_list = []
    X_p_list = []
    for i, p in enumerate(paths_c):
        if p in p_idx:
            valid.append(p)
            X_c_list.append(X_c_all[i])
            X_p_list.append(X_p_all[p_idx[p]])
            
    paths = valid
    X_c_all = np.array(X_c_list, dtype=np.float32)
    X_p_all = np.array(X_p_list, dtype=np.float32)
    
    lbl = load_labels(db, paths)
    sets = load_label_sets(db)
    human = lbl['human']
    weak_path = lbl['weak_path']
    weak_panns = lbl['weak_panns']
    label_source = lbl['label_source']
    is_val = lbl['is_val']

    wl_map = cfg.get("weak_label_map", {})
    weak_weight = cfg.get("weak_weight", 0.2)
    bulk_weight = cfg.get("bulk_weight", 0.5)

    cls_idx = {c: j for j, c in enumerate(classes)}

    rows, Y, w = [], [], []
    val_rows, Y_val = [], []
    train_labeled_rows = []

    def multi_hot(labels):
        y = np.zeros(len(classes), dtype=np.int8)
        hit = False
        for l in labels:
            j = cls_idx.get(l)
            if j is not None:
                y[j] = 1
                hit = True
        return y if hit else None

    for i in range(len(paths)):
        h = human[i]
        if is_val[i] == 1:
            if h:
                y = multi_hot(sets.get(paths[i], [h]))
                if y is not None:
                    val_rows.append(i)
                    Y_val.append(y)
            continue

        if h:
            y = multi_hot(sets.get(paths[i], [h]))
            if y is None:
                continue
            rows.append(i)
            Y.append(y)
            train_labeled_rows.append(len(rows) - 1)
            if label_source[i] == "single":
                w.append(1.0)
            else:
                w.append(bulk_weight)
        else:
            w_l = weak_path[i]
            if not w_l or w_l not in cls_idx:
                w_l = wl_map.get(weak_panns[i], weak_panns[i])
            w_l = wl_map.get(w_l, w_l)
            if w_l and w_l in cls_idx:
                rows.append(i)
                y = np.zeros(len(classes), dtype=np.int8)
                y[cls_idx[w_l]] = 1
                Y.append(y)
                w.append(weak_weight)

    X_c_t = X_c_all[rows] if rows else np.zeros((0, X_c_all.shape[1]), np.float32)
    X_c_v = X_c_all[val_rows] if val_rows else np.zeros((0, X_c_all.shape[1]), np.float32)
    
    X_p_t = X_p_all[rows] if rows else np.zeros((0, X_p_all.shape[1]), np.float32)
    X_p_v = X_p_all[val_rows] if val_rows else np.zeros((0, X_p_all.shape[1]), np.float32)
    
    Y = np.array(Y, dtype=np.int8)
    w = np.array(w)
    Y_v = np.array(Y_val, dtype=np.int8)
    
    # For zero-shot train (only human labels)
    X_c_zs_train = X_c_t[train_labeled_rows] if train_labeled_rows else np.zeros((0, X_c_t.shape[1]), np.float32)
    Y_zs_train = Y[train_labeled_rows] if train_labeled_rows else np.zeros((0, len(classes)), dtype=np.int8)
    
    return {
        "X_c_t": X_c_t, "X_c_v": X_c_v,
        "X_p_t": X_p_t, "X_p_v": X_p_v,
        "Y_t": Y, "w_t": w, "Y_v": Y_v,
        "X_c_zs_train": X_c_zs_train, "Y_zs_train": Y_zs_train
    }

def run_clap_eval(args):
    from .export import get_class_set, load_ml_cfg
    from .train import fit_ovr, calibrate_thresholds, _report, predict_probs
    
    db_dir = os.path.dirname(os.path.abspath(args.db))
    npz_path = os.path.join(db_dir, "models", "clap_pilot.npz")
    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}. Run clap-embed first.")
        return
        
    cfg = load_ml_cfg(db_dir)
    classes = sorted(get_class_set(db_dir))
    target_precision = cfg.get("target_precision", 0.9)
    fallback = cfg.get("conf_threshold", 0.6)
    
    print("Building aligned datasets...")
    ds = build_aligned_datasets(npz_path, args.db, classes, cfg)
    Y_v = ds['Y_v']
    
    print("\n--- (1) Baseline: PANNs OvR ---")
    W1, b1 = fit_ovr(ds['X_p_t'], ds['Y_t'], ds['w_t'], classes, verbose=False)
    probs_1 = predict_probs(ds['X_p_v'], W1, b1)
    thr1 = calibrate_thresholds(Y_v, probs_1, classes, target_precision, fallback)
    rep1 = _report(Y_v, probs_1, classes, np.array([thr1[c] for c in classes], dtype=np.float32))
    
    print("\n--- (2) CLAP zero-shot ---")
    anchors = []
    for c in classes:
        anc = text_anchor(c)
        if anc is None:
            anc = np.zeros(512, dtype=np.float32)
        anchors.append(anc)
    anchors = np.array(anchors, dtype=np.float32)
    
    probs_2_train = ds['X_c_zs_train'] @ anchors.T
    probs_2 = ds['X_c_v'] @ anchors.T
    
    # Calibrate on zs_train
    thr2_dict = {}
    fallback_zs = 0.1
    for j, c in enumerate(classes):
        y_true = ds['Y_zs_train'][:, j]
        p = probs_2_train[:, j]
        support = y_true.sum()
        if support < 5:
            thr2_dict[c] = fallback_zs
            continue
        best_t = fallback_zs
        for t in np.arange(0.01, 0.99, 0.01):
            pred = (p >= t)
            tp = (pred & (y_true == 1)).sum()
            fp = (pred & (y_true == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if precision >= target_precision:
                best_t = float(t)
                break
        thr2_dict[c] = float(round(best_t, 2))
    
    rep2 = _report(Y_v, probs_2, classes, np.array([thr2_dict[c] for c in classes], dtype=np.float32))
    
    print("\n--- (3) CLAP OvR ---")
    W3, b3 = fit_ovr(ds['X_c_t'], ds['Y_t'], ds['w_t'], classes, verbose=False)
    probs_3 = predict_probs(ds['X_c_v'], W3, b3)
    thr3 = calibrate_thresholds(Y_v, probs_3, classes, target_precision, fallback)
    rep3 = _report(Y_v, probs_3, classes, np.array([thr3[c] for c in classes], dtype=np.float32))
    
    print("\n--- (4) Concat [PANNs|CLAP] OvR ---")
    X_cat_t = np.concatenate([ds['X_p_t'], ds['X_c_t']], axis=1)
    X_cat_v = np.concatenate([ds['X_p_v'], ds['X_c_v']], axis=1)
    W4, b4 = fit_ovr(X_cat_t, ds['Y_t'], ds['w_t'], classes, verbose=False)
    probs_4 = predict_probs(X_cat_v, W4, b4)
    thr4 = calibrate_thresholds(Y_v, probs_4, classes, target_precision, fallback)
    rep4 = _report(Y_v, probs_4, classes, np.array([thr4[c] for c in classes], dtype=np.float32))
    
    print("\n--- (5) Ensemble (PANNs OvR + CLAP ZS) ---")
    probs_5 = (probs_1 + probs_2) / 2.0
    thr5 = calibrate_thresholds(Y_v, probs_5, classes, target_precision, fallback)
    rep5 = _report(Y_v, probs_5, classes, np.array([thr5[c] for c in classes], dtype=np.float32))
    
    print("\n=== SUMMARY ===")
    print(f"(1) PANNs OvR (baseline):   {rep1['macro avg']['f1-score']:.4f}")
    print(f"(2) CLAP zero-shot:         {rep2['macro avg']['f1-score']:.4f}")
    print(f"(3) CLAP trained OvR:       {rep3['macro avg']['f1-score']:.4f}")
    print(f"(4) Concat trained OvR:     {rep4['macro avg']['f1-score']:.4f}")
    print(f"(5) Ensemble (1 + 2):       {rep5['macro avg']['f1-score']:.4f}")
