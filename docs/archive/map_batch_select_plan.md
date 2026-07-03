# Map batch selection — label a group of samples at once

Goal: on the Sample Map, draw a selection (rectangle/lasso) around a group of
points, then apply one instrument label to all of them in a single action.

Critical files: `sampletagger/web/static/map.js`, `sampletagger/web/static/map.html`,
`sampletagger/web/api.py` (new endpoint + bulk-label helper).

---

## STATUS: implemented — revisit findings (2026-06-30)

Phases 1–4 shipped (the lasso, marked optional below, was built directly):
- `_bulk_label` helper (`api.py:349`); `label_propagate`/`label_cluster` refactored onto it.
- `label_map` endpoint + sidecar-mtime stale guard (`api.py:627`).
- `map_api` emits `sidecar_mtime` (`api.py:620`).
- Lasso select (bbox pre-filter + ray-casting), `batchSel` populated, `draw()`
  renders the polygon, apply handler with stale→reload.

### ⚠ Must address — label provenance collision (NEW, highest priority)

`human_instrument` now has **five writers**, distinguished only by `label_source`:
`single` (true human), `propagate`/`cluster`/`map` (human-seeded **bulk**), and
`llm` (**machine** — `scripts/cluster_apply.py`, LLM cluster guesses). But:

- `export.py:52` selects only `human_instrument, path_instrument, panns_instrument`
  — it **never reads `label_source`**, so training is provenance-blind.
- `train.py:51` treats every non-empty `human_instrument` as weight-1.0 gold (same
  for the K-fold `X_human`). ⇒ LLM-applied labels are trained on as ground truth and
  the CV accuracy report is **circular/inflated**.
- `is_val` (the frozen 20% validation slice from `classification_plan.md`) is declared
  in `db.py:13` but **set nowhere** — there is no honest held-out eval at all.

**Fix (do before trusting any accuracy number):**
1. `export.py` — also emit `label_source` per row.
2. `train.py` — define ground truth as `label_source='single'` (true human) only;
   demote bulk/`llm` labels to weak (a sample-weight tier between human 1.0 and
   weak 0.2), or exclude them. Report CV on `single` only.
3. Implement the `is_val` freeze: flag ~20% of `label_source='single'` rows, exclude
   from training, evaluate on them. Never let `propagate`/`cluster`/`map`/`llm` rows
   enter the val slice (leakage — bulk labels are correlated by construction).
4. Consider routing `llm` (and optionally `map`/`cluster`) labels to a separate
   column (e.g. `auto_instrument`) instead of `human_instrument`, so the ground-truth
   column stays clean. Cross-reference `docs/classification_efficiency_plan.md`.

### Decisions / minor cleanups in the shipped code

- **Default mode contradicts the goal.** `batchUnlabeledOnly` is *checked* by default
  (`map.html`), so `mode="unlabeled"` — re-labeling an already-labeled group silently
  skips those. The original ask was "change the label of a group." Decide: default to
  `all` (override) and keep the checkbox as the safe opt-in, or keep as-is.
- **Dead fallbacks in `_bulk_label` (`only_unlabeled=False`).** `instrument` is always
  non-null/validated, so `COALESCE(?, model_instrument, …)` and the `CASE WHEN ? IS
  NOT NULL …` always resolve to `instrument`/`'human'`; the fallback branches never
  fire. Harmless, but simplify to avoid implying behavior that can't happen.
- **Touch can't select** (touchstart always pans; `selectMode` ignored). Acceptable
  for v1; revisit if tablet labeling matters.
- **`draw()` highlight vs `shown()`:** when `stride==1` a selected-but-filtered-out
  point still gets the white highlight; when `stride>1` it doesn't. Cosmetic.

---

## Key constraints (drive the design)

1. **The client has no paths.** `map_api()` (`api.py:569`) sends only
   `x,y,c,t,d,s,cats,colors,n,mi`; the `paths` array stays server-side in the
   `build_map()` cache (`api.py:566`), and `point_api(i)` (`api.py:577`) resolves a
   single index → path on demand. ⇒ **Batch labeling must POST point _indices_**,
   and the server translates index → path via the cached `paths` array.

2. **Indices are only stable for a given projection.** `build_map()` is cached and
   keyed by `DB + ".proj"` mtime (`_sidecar_mtime`). If a reproject runs between map
   load and label submit, indices shift. ⇒ send the map's `_sidecar_mtime` as a
   token and have the server reject a stale submit (client reloads).

3. **Drag is already pan + click-to-pick** (`map.js:42-47`). The selection gesture
   must not collide with panning.

4. **Bulk-label logic already exists, duplicated.** `label_propagate` (`api.py:337`)
   and `label_cluster` (`api.py:438`) each loop per-path `UPDATE … WHERE path=? AND
   (human_instrument IS NULL …)`. The new endpoint should reuse a shared helper.

---

## Phase 1 — Backend: shared bulk-label helper + map endpoint

1. Factor a helper in `api.py`:
   ```python
   def _bulk_label(paths, instrument, source, only_unlabeled=True):
       # one connection, executemany; respects only_unlabeled guard; returns n changed
   ```
   Refactor `label_propagate` and `label_cluster` to call it (source `'propagate'` /
   `'cluster'`, `only_unlabeled=True`) — removes the duplicated loops.

2. Expose `build_map()`'s `paths` + `_sidecar_mtime` to the new endpoint (already in
   the `_MAP` cache; just read `m = build_map()`).

3. New POST route `/api/label_map`:
   ```
   data = {indices: [int], instrument: str, sidecar_mtime: float, mode: "all"|"unlabeled"}
   ```
   - `m = build_map()`; if `data.sidecar_mtime != m["_sidecar_mtime"]` → return
     `{ok:False, stale:True}` (client reloads the map).
   - Validate `instrument in get_labels()`.
   - Translate: `paths = [m["paths"][i] for i in indices if 0 <= i < m["n"]]`.
   - `n = _bulk_label(paths, instrument, source="map", only_unlabeled=(mode=="unlabeled"))`.
     Default `mode="all"` (explicit manual selection = override) per the user's
     "change the label of a group"; offer `unlabeled` as a safety toggle.
   - Invalidate caches: `_MAP = None; _CLUSTERS = None`. Return `{ok:True, n, instrument}`.

4. Register in `POST_ROUTES` (`api.py:994`):
   `"/api/label_map": lambda data: label_map(data)`.

**Verify:** `curl -X POST /api/label_map -d '{"indices":[0,1,2],"instrument":"kick","sidecar_mtime":<m>,"mode":"all"}'`
returns `{ok:true,n:3}`; a stale `sidecar_mtime` returns `{ok:false,stale:true}`;
DB rows for those paths show `human_instrument='kick'`, `label_source='map'`.

## Phase 2 — Frontend: selection gesture + state

Add to `map.js`:

1. **Mode toggle.** A "▭ Select" button in the header (`map.html`) that flips
   `selectMode`. When on, the cursor is a crosshair and drag draws a selection
   instead of panning. Also support **Shift+drag** as a shortcut without toggling.

2. **Rectangle select (baseline).** On `mousedown` in select mode, record the start
   device-pixel point; on `mousemove`, store the current point and `draw()` a
   translucent rectangle overlay; on `mouseup`, compute the selection:
   ```js
   selIdx = new Set();
   for (let i=0;i<M.n;i++){ if(!shown(i)) continue;       // respect legend filters
       const X=sx(i),Y=sy(i);
       if (X>=x0&&X<=x1&&Y>=y0&&Y<=y1) selIdx.add(i); }
   ```
   Loop over **all** `M.n` (not `stride` — stride is render-only) so nothing is
   missed; 226k point-in-rect tests on mouseup is sub-frame.
   Respecting `shown(i)` is a feature: pre-filter by instrument/type/source/length
   in the legend, then select only what's visible.

3. **Render selected.** In `draw()`, after the normal pass, redraw `selIdx` points
   with a highlight (brighter fill or white outline) so the selection is obvious,
   plus draw the in-progress selection rectangle.

4. **Additive / subtractive (optional).** Shift adds to `selIdx`, Alt removes;
   plain select replaces. Click on empty space clears.

## Phase 3 — Frontend: apply panel

1. Reuse the existing label dropdown. When `selIdx.size > 0`, show a batch panel in
   the side bar (mirror `labelRow`, `map.html:28`): `"N selected"`, the `labelSel`
   dropdown, a `mode` checkbox ("only unlabeled"), and an **Apply to N** button.
2. On Apply: POST `{indices:[...selIdx], instrument, sidecar_mtime:M._sidecar_mtime, mode}`
   to `/api/label_map`.
   - The client needs `_sidecar_mtime`; **add it to the `map_api()` payload**
     (`api.py:570` `out`) so the client can echo it back.
   - On `{ok:true}`: toast `"Labeled N ✓"`, update `M.c[i]` for each selected index
     to the new cat (like the single-label path at `map.js:128`), `draw()`, clear
     `selIdx`.
   - On `{stale:true}`: toast "map changed — reloading", call `loadMap(false)`,
     clear selection.

**Verify (manual, via /run or browser):** box-select a visible blob, pick an
instrument, Apply → dots recolor immediately, toast shows the count, and a page
reload shows the labels persisted (served from DB).

## Phase 4 (optional) — Lasso select

Replace/augment the rectangle with a freeform polygon (UMAP blobs aren't
axis-aligned). Collect mousemove points into a path, draw the polyline, and test
membership with a ray-casting point-in-polygon. Same `selIdx` plumbing and same
endpoint — purely a client-side geometry swap. Do this after the rectangle works.

---

## Notes / gotchas

- **Selection size.** A blob can be thousands of points; a POST body of N ints is
  small (tens of KB) and `_bulk_label` via `executemany` handles it fine. If huge
  selections become common, an alternative is to send the rect + active filters and
  let the server do the geometry against its own `x/y` arrays — defer unless needed.
- **Cache invalidation.** `/api/label_map` must null `_MAP` and `_CLUSTERS` (labels
  feed both the map colors and cluster summaries).
- **Touch.** Phase 2 is mouse-first; a two-finger vs one-finger gesture split for
  selection on touch is out of scope for v1 (toggle button still works on tablets).
- **Override semantics.** Default `mode="all"` overwrites existing `human_instrument`
  for selected points (matches "change the label of a group"); the "only unlabeled"
  toggle gives the propagate/cluster-style safe behavior.

## Sequencing

Phase 1 (backend, testable via curl) → Phase 2+3 (rectangle select + apply, the
usable feature) → Phase 4 (lasso, nicer geometry) last.
