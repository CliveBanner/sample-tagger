# Web UI usability improvements (labeling-focused)

## Context

The webapp's frontend now lives in `sampletagger/web/static/` (index/map/review/settings .html+.js +
shared `app.css`). The map explorer is already strong (pan/zoom, touch, filters, similarity,
arrow-key nav, crossfade audio). The weak spot is the **review/labeling page** — and that's the
exact tool the upcoming ~700-label active-learning campaign runs in. Today labeling requires a
**mouse click per label** (only ←/→ are bound), there's no audio-replay key, no one-key "accept
the suggestion", no undo, and no session momentum (no per-session count, no auto-next-batch).
Making labeling keyboard-driven is the single highest-leverage UX change for the project's
near-term goal.

This plan is behavior-additive (no route/JSON breakage). Critical files:
`sampletagger/web/static/review.js`, `review.html`, `index.js`, `app.css`, and a small backend
touch in `sampletagger/web/api.py` (`review_queue` modes, `/api/review/queue`).

## Tier 1 — make review labeling keyboard-driven (the core win)

All in `review.js` (+ small `review.html`/`app.css` for hints), guarded so typing in the label
manager input is unaffected (existing handler already skips INPUT/SELECT).

1. **Type-to-label.** Capture letter keys to build a short buffer that prefix-matches the
   `INSTRUMENTS` list; show the matched label inline; **Enter commits** (calls existing `save()`),
   Esc clears. Unique-prefix auto-highlights. This turns each label into ~2–4 keystrokes with no
   mouse. Reuse `save(instrument)` (`review.js:141`) unchanged.
2. **Number keys 1–9** bound to the first 9 labels (and the on-screen buttons get a small index
   badge so the mapping is visible). Fast path for the common classes.
3. **Spacebar = replay audio from 0** (pause if playing). Labeling by ear means constant replays;
   today there's no key. Reuse the `#player` element in `renderDetail`.
4. **Accept-suggestion key (Enter when buffer empty).** Highlight one *recommended* label
   (model prediction once it exists, else PANNs → path fallback) by pre-selecting that `.ibtn`
   and binding Enter to save it. Makes the plan's "confirm the guess" flow one keystroke. Today
   `renderDetail` only marks `human_instrument`; add a `suggested` highlight distinct from the
   saved ✓.
5. **Undo last label** (`z` / Backspace): revert the last `save()` (clear `human_instrument`,
   re-POST empty/again, step back). Cheap insurance once keyboard labeling speeds up mistakes.
6. **Skip-as-unsure** (`s` or `0`): advance without labeling but mark seen, so a re-fetched batch
   doesn't resurface it immediately. Aligns with the plan's "unsure/skip without injecting noise."
7. **Update the on-screen `kbhint`** to list the new shortcuts.

## Tier 2 — campaign support (momentum + targeting)

8. **Session counter + auto-next-batch** (`review.js`/`review.html`): show "labeled N this
   session"; when the queue is exhausted, auto-call `loadQueue()` for the next batch (or show a
   prominent "Load next batch" CTA) instead of dead-ending.
9. **Per-class label progress on the dashboard** (`index.js`): a "labels by class" card showing
   `human_instrument` counts vs. a target — the direct readout for tracking the 700-label effort.
   Backend: extend `/api/stats` (`api.py:stats`) with a `human_by_class` distribution (one cheap
   `GROUP BY human_instrument`).
10. **Active-learning queue modes** (`api.py:review_queue` + the `#modesel` dropdown): add
    "by class" (focus organ/piano/rhodes/…) and an "uncertain" mode (low `model_conf` once the
    model exists). This is the UI hook the training pipeline's `active.py` plugs into; ship the
    `by class` filter now (works off existing columns), wire "uncertain" when `model_conf` lands.

## Tier 3 — polish (low effort, do if time)

11. **Distinct label-button colors / grouping** (`review.js` `COLORS`, `app.css .ibtns`): today
    snare/tom share orange, hihat/cymbal share yellow, kick/808/vocal share pink — hurts at-a-glance
    scanning. Give each class a unique hue and/or group buttons (drums · keys · melodic · other).
12. **Map "label this" hotkey** (`map.js`): bind a key to label the selected point via the same
    flow, so quick corrections don't require the dropdown+button (`#labelSel`/`#btnLabel`).
13. **CSS cleanup** (`app.css`): the file concatenates four `:root`/`*`/`body` blocks (one per
    former page); the map block even drops some vars. Deduplicate into one `:root` and scope
    page-specific rules under a `body.page-*` class. Cosmetic, no visual change.
14. **Inline toasts instead of `alert()`** for label-add and projection-failure errors
    (`review.js:26`, `map.js:227`) — reuse the existing `showToast`.

## Verification

- Restart `sample-tagger-web`; open `/review`.
- **Keyboard flow**: type a label prefix + Enter saves and advances; number keys save; Space
  replays audio; Enter-on-empty accepts the highlighted suggestion; `z` undoes; `s` skips.
  Confirm each `save()` still POSTs `/api/label` and writes `human_instrument` (check via
  `sqlite3 samples.db "SELECT human_instrument,count(*) ... GROUP BY 1"`).
- **No regressions**: mouse clicking labels, ←/→ nav, the label-manager modal, and the mobile
  overlay (`<680px`) all still work; typing in the "new label" input does NOT trigger shortcuts.
- **Dashboard**: the new per-class card renders and counts match the DB; `/api/stats` still loads.
- **Queue modes**: "by class" returns only that class's candidates; existing "disagree"/"all"
  unchanged.
- Phone check: review label buttons stay large/tappable; map still stacks.

## Note

Repo is being actively refactored elsewhere; these are all in `web/static/*` plus two small
`api.py` additions — coordinate timing or hand the static-file diffs to the same implementer.
The boilerplate-reduction plan previously at this path is saved at
`docs/boilerplate_reduction_plan.md`.
