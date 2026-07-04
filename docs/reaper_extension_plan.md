# REAPER extension: CLAP text search inside the DAW

> **STATUS (2026-07-04): Phases 1–2 implemented & working** on the Windows DAW PC
> (`reaper/sampletagger_search.lua`). One deviation from the plan below: blocking
> `ExecProcess` waits proved broken on Windows (exit 259/STILL_ACTIVE), so ALL
> HTTP — search JSON included — uses fire-and-forget curl + output-file polling,
> not blocking calls. Phase 3 (ReaPack packaging, server-side filters,
> keyboard-only flow) remains open.

Goal: a dockable panel in REAPER — type a query ("dusty vinyl breakbeat"), hit the
sample-tagger server's CLAP search, preview results, insert the pick at the edit
cursor. No server-side changes needed for the MVP; the API already has everything.

## Tech choice

**ReaScript Lua + ReaImGui**, not a compiled C++ extension.

- ReaImGui gives a proper dockable GUI (same tech as many ReaTeam scripts), installable via ReaPack.
- No compile step, works on whatever OS the DAW machine runs.
- HTTP via `curl` through `reaper.ExecProcess()` — the standard ReaScript trick. curl ships with Windows 10+, macOS, and every Linux.
- Preview playback via the **SWS** `CF_Preview` API (`CF_CreatePreview` / `CF_Preview_Play`) — audition without touching the project.

Dependencies on the DAW machine: REAPER ≥ 6.x, SWS ≥ 2.13, ReaImGui (both via ReaPack), curl in PATH.

Repo layout: new `reaper/` dir in this repo:

```
reaper/
  sampletagger_search.lua   -- the script (single entry point)
  json.lua                  -- small custom JSON decode/encode module (no deps)
```

## Existing API surface used (no changes)

| Endpoint | Use |
|---|---|
| `GET /api/search_text?q=<urlenc>&k=<n>&offset=<n>` | CLAP text search. Hits carry `path` (absolute server path), `name`, `score`, plus meta (`instrument`, `human_labels`, `model_labels`, `sample_type`, `duration_s`, …) |
| `GET /api/audio?path=<urlenc>` | raw file bytes, correct Content-Type, Range supported |
| `GET /api/audio?path=<urlenc>&norm=1` | ffmpeg loudness-normalized mp3 — good for previews, not for insert |
| `GET /api/similar?path=<urlenc>&k=<n>` | acoustic (PANNs) similarity — "more like this" button |

Gotchas to design around:

- **First query after server start is slow** (CLAP model load, several seconds). Show a "warming up…" status instead of a spinner-of-death; use a generous curl timeout (`-m 30`) on the first request.
- Server reads files off the rclone FUSE mount → first-byte latency on `/api/audio`. Fine on LAN, but download must not block the UI thread forever (see async note below).
- `search_text` has **no server-side filters** — filter client-side on the meta in the hits (instrument / sample_type / duration), and over-fetch (`k=100`) so filters still leave a page of results. If that proves too lossy, adding `instrument=`/`type=` params to `search_text_api` is a small Phase-3 server change.

## Getting audio into REAPER: two modes

Configurable, auto-detected at startup:

1. **`direct`** — the DAW machine has the same pCloud library mounted. A single
   prefix rewrite maps server paths to local ones, e.g.
   `/home/phlp/pcloud/DAW/Samples` → `P:\DAW\Samples`. Insert references the
   mounted file directly (zero copy, but the project then depends on the mount).
2. **`download`** — fetch via `/api/audio` into a local cache
   `<GetResourcePath()>/SampleTagger/cache/<sha1(path)><ext>`, insert the cached
   copy. Cache hit = instant re-insert. This is the safe default.

Startup check: if the configured local prefix exists on disk → offer `direct`, else `download`.

Config file `<GetResourcePath()>/SampleTagger/config.json`:

```json
{
  "server": "http://192.168.178.56:8765",
  "mode": "download",
  "remote_prefix": "/home/phlp/pcloud/DAW/Samples",
  "local_prefix": "",
  "insert_mode": "cursor"
}
```

## UI sketch (ReaImGui, dockable)

```
┌ SampleTagger ──────────────────────────────┐
│ [ dusty vinyl breakbeat        ] [Search]  │
│ filters: [inst ▾] [oneshot|loop|all] [dur] │
│────────────────────────────────────────────│
│ ▸ AMEN_170_full.wav      loop  drums  0.41 │
│ ▸ vinyl_break_92.wav     loop  drums  0.39 │  ← click = preview
│ ▸ dusty_kick_03.wav      1shot kick   0.35 │  ← dbl-click/Enter = insert
│   …                              [more]    │
│────────────────────────────────────────────│
│ ▶ vinyl_break_92.wav  [Insert] [Similar]   │
│ status: 48 hits (226k indexed)             │
└────────────────────────────────────────────┘
```

- Click row → download (if needed) → `CF_CreatePreview` + play. Toggle for `norm=1` level-matched previews.
- Double-click / Enter → `reaper.InsertMedia(local_path, 0)` at edit cursor on selected track (mode `1` = new track, behind the `insert_mode` setting). Wrap in `Undo_BeginBlock`/`Undo_EndBlock`.
- `[Similar]` → `/api/similar` with the selected hit's path, replaces the result list.
- `[more]` → same query with `offset += k` (the API already paginates).
- Note: ReaImGui cannot start an OS-level drag onto the arrange view — buttons/keys are the insert path, that's a hard constraint.

## Implementation notes

**HTTP from Lua.** `reaper.ExecProcess(cmd, timeout_ms)` returns exit code on the
first line, stdout after. Search (small JSON) can be a blocking call with
`-m 30`; keep it on a `reaper.defer` tick so the panel repaints first
("searching…"). Audio download in Phase 1 is also blocking (LAN + small files);
Phase 2 makes it async: `ExecProcess(curl -o file.part …, -2)` (fire-and-forget),
then poll each defer tick until the file's size is stable for 2 ticks, rename to
final name, play. Always download to `.part` and rename so a killed curl never
leaves a truncated file that later cache-hits.

**URL encoding.** Sample paths contain spaces/`&`/unicode — encode every query
param (`string.gsub(s, "[^%w%-%._~]", function(c) return string.format("%%%02X", c:byte()) end)`).

**JSON.** ReaScript Lua has no JSON lib; a small custom `json.lua` lives next to
the script and is loaded with `dofile(script_dir .. "json.lua")`.

**Windows curl quoting.** Build the command as `curl.exe -s -m 30 "<url>"` —
ExecProcess doesn't go through a shell, so no PowerShell-alias or pipe issues,
but do quote the URL.

## Phases

**Phase 1 — MVP (single sitting):**
- [edit] `reaper/sampletagger_search.lua`: config load, curl+json helpers, search box, result list, blocking download, `InsertMedia` on double-click.
- [edit] vendor `reaper/json.lua`.
- [ui] REAPER: Actions → Load ReaScript → dock the window.
- [check] query "vinyl breakbeat" → hits match the webapp map-page search; insert lands at edit cursor; undo removes it.

**Phase 2 — audition quality of life:**
- [edit] CF_Preview playback on single click, stop on next click / Esc, `norm=1` toggle.
- [edit] async `.part` download + cache, cache-size cap (LRU delete over e.g. 2 GB).
- [edit] client-side filters (instrument dropdown fed from hit meta, type radio, max duration).
- [check] rapid-fire clicking rows never plays a truncated file.

**Phase 3 — nice-to-have:**
- [edit] `[Similar]` via `/api/similar`; pagination `[more]`.
- [edit] optional server change: `instrument=`/`type=` filter params on `search_text_api` (filter before top-k instead of after).
- [edit] ReaPack index in the repo so the DAW machine can install/update the script like any other package.
- [edit] keyboard-only flow: focus search on open, arrows navigate hits, Enter inserts (mirrors the map page's arrow-key navigation).

## Open questions (defaults chosen, flip in config if wrong)

- Which OS runs REAPER? Only affects the `local_prefix` example and nothing else — both modes are OS-agnostic.
- Insert at edit cursor on selected track (`InsertMedia` mode 0) chosen as default; media-explorer-style "insert on new track" is the config alternative.
