# CLAP evaluation plan — decide the labeling rethink with data (2026-07-03)

## Context

Fine-grained flat labeling is struggling: synth/pad mix orthogonal axes, round-one active
learning was flat (macro F1 0.569 → 0.565), and the suspected ceiling is the PANNs
embedding itself (AudioSet-trained, event-oriented — not made for synth-vs-acoustic or
role distinctions). CLAP (contrastive language-audio) embeddings offer two things PANNs
can't: **text→audio search** ("warm analog pad") and **zero-shot classification** from
text prompts, no training labels needed.

The 470-file frozen gold set makes this decidable objectively before committing to a full
226k re-embed. Three-way benchmark, all scored on the same val set:

| candidate | labels needed | what it tests |
|---|---|---|
| (1) current: PANNs emb + trained OvR | existing | baseline = 0.569 macro F1 |
| (2) CLAP zero-shot (text prompts per class) | none | is labeling even necessary? |
| (3) CLAP emb + trained OvR (same labels as 1) | existing | is the embedding the ceiling? |
| (4) concat [PANNs\|CLAP] + trained OvR | existing | do the embeddings complement each other? |
| (5) ensemble: mean of (1) and (2) scores, blend tuned on train | existing | cheap fusion without retraining |

Decision rule: (2) or (3) clearly beats (1) → full re-embed, retrieval-first path
(text search + coarse labels). (4)/(5) clearly beat everything → keep both models
(accepting double forward passes + both sidecars permanently — only worth a real
margin, it cuts against the strip-down). Nothing beats (1) → embeddings aren't the
problem; go to the faceted-labeling rethink instead (separate plan).

## Phase 1 — Pilot on labeled files only (~1 day, no full re-embed)

Only ~10.4k files need decoding: the 470 val + ~10k train-labeled. At FUSE speed
(~2.5 files/s) that's ~70 min of decode.

1. **[cmd]** Install CLAP into the venv:
   `./venv/bin/pip install laion_clap` — use the **music checkpoint**
   `music_audioset_epoch_15_esc_90.14.pt` (~2 GB, auto-download or manual to
   `~/clap_ckpt/`). CPU inference is fine for 10k files; if the `gpu_python` env has
   CUDA torch, use it (4 GB GTX 970 fits the audio tower).
2. **[edit]** New `sampletagger/ml/clap.py`:
   - `embed_paths(paths, db, out_npz)` — decode (librosa → 48 kHz mono, ≤10 s window,
     tile short one-shots like panns.py does), batch through
     `model.get_audio_embedding_from_data`, save `{paths, X}` npz (512-d, L2-normed).
   - `text_anchor(prompts)` — mean of `get_text_embedding` over prompt variants.
   - Resumable: skip paths already in the npz (FUSE hiccups shouldn't restart the run).
3. **[edit]** `sampletagger/ml/cli.py`: subcommands `clap-embed` (embed val+labeled
   paths) and `clap-eval` (run the three-way benchmark below).
4. **[edit]** Prompt table (start in `clap.py`, move to labels.db later if adopted):
   3–5 prompts per class, e.g. kick: "a kick drum one-shot", "an 808 kick drum sample",
   "a punchy bass drum hit"; pad: "a sustained ambient synthesizer pad",
   "a warm atmospheric pad sound". Include the crossover phrasing for synth
   ("a synthesized instrument sound", "an analog synthesizer emulating strings").
5. **[cmd]** `sample-tagger-ml clap-embed samples.db && sample-tagger-ml clap-eval samples.db`
   The eval computes:
   - (2) zero-shot: cosine(file, class-anchor) per class; per-class thresholds
     calibrated on the *train-labeled* files (never the val set) at target_precision;
     score val with per-class binary F1.
   - (3) trained: reuse `fit_ovr`/`calibrate_thresholds`/`_report` from train.py on
     CLAP features — identical protocol to the baseline, only the embedding differs.
6. **[check]** Three macro F1 numbers side by side, plus per-class: watch synth, pad,
   mallet, organ specifically — that's where PANNs is suspected of being blind.
   Write the numbers into this doc.

   **Results (2026-07-03)**:
   * (1) PANNs OvR (baseline on pilot set): **0.5176**
   * (2) CLAP zero-shot: **0.3591**
   * (3) CLAP trained OvR: **0.5540**
   * (4) Concat trained OvR: **0.5855**
   * (5) Ensemble (1 + 2): **0.5483**

   *Notes on specific classes*: 
   - **CLAP trained OvR (3)** beat the PANNs baseline overall, showing it captures much better features for our dataset out of the box (e.g. `synth` at 0.46 F1 vs being very low in PANNs).
   - **Concat trained OvR (4)** hit the highest score (**0.5855**), confirming that PANNs and CLAP capture complementary information.
   - Zero-shot (2) was low (0.3591) without iteration, but still helped when ensembled in (5).

## Phase 2 — only if CLAP wins: full re-embed (one overnight run)

7. **[edit]** Extend `clap-embed` for the full library with **local staging**: worker
   pool `rclone copy`s batches of ~500 files to a local tmp dir, decode+embed from
   local disk, delete batch. Decouples the FUSE bottleneck from the GPU/CPU compute.
   Resume via the npz done-set. Estimate: FUSE-bound ~25 h naive; staged, the copy
   pipeline overlaps compute — realistically overnight-to-a-day.
8. **[cmd]** Export sidecar `samples.clap.npy` + `.clap.paths` (fp16, 512-d ≈ 230 MB —
   9× smaller than the PANNs sidecar).
9. **[edit]** `sampletagger/embeddings.py`: `load(db, model="panns"|"clap")` picking the
   sidecar pair; `ml/export.py` gains an `ml_params` key `feature_model`.
10. **[edit]** Text search in the webapp: `GET /api/search_text?q=...` — encode the
    query with the CLAP text tower (load lazily, cache), cosine over the sidecar,
    return top-k with meta; search box on the map page focuses/highlights results.
    This is the retrieval-first payoff and works with zero labels.
11. **[check]** Text-search smoke: "vinyl breakbeat", "warm analog pad", "female vocal
    chop", "808 sub bass" — listen to top-10 each in the map UI.
12. Decisions after that: retrain production model on CLAP features; re-project the map
    from CLAP; whether the fine-grained taxonomy is still needed at all (coarse
    classes + text search may cover every retrieval need).

## Not in this plan

- The faceted-labeling redesign — only if Phase 1 says embeddings aren't the problem.
- Deleting/refactoring the PANNs pipeline — keep it until CLAP is proven; the sidecars
  coexist.

## Honest caveats

- Zero-shot quality is prompt-sensitive; spend the 30 minutes iterating prompts on
  *train* files (never val) before reading the val number.
- The GTX 970 (4 GB, CUDA compute 5.2) may be unsupported by current torch builds —
  CPU fallback works, just slower (~0.5–1 s/file for the pilot's 10k).
- CLAP's 10 s window vs PANNs' 30 s: long loops get truncated differently; fine for
  one-shots and loops alike in practice, but keep the window consistent between
  pilot and full run.
