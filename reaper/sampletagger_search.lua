-- @description SampleTagger — CLAP text search for the sample library
-- @version 0.1
-- @author phlp
-- @about
--   Dockable panel that queries the sample-tagger server's CLAP text search
--   (/api/search_text), previews hits (SWS CF_Preview) and inserts them at the
--   edit cursor. Requires ReaImGui + SWS (via ReaPack) and curl in PATH.
--   Config lives in <resource path>/SampleTagger/config.json.

-- ---------------------------------------------------------------- deps

if not reaper.ImGui_CreateContext then
  reaper.MB("This script needs ReaImGui.\nInstall it via Extensions > ReaPack > Browse packages.",
            "SampleTagger", 0)
  return
end
local PREVIEW_OK = reaper.CF_CreatePreview ~= nil  -- SWS >= 2.13; insert works without it

local SCRIPT_PATH = ({reaper.get_action_context()})[2]
local SCRIPT_DIR = SCRIPT_PATH:match("^(.*[/\\])")
local json = dofile(SCRIPT_DIR .. "json.lua")

local IS_WIN = reaper.GetOS():match("Win") ~= nil
local SEP = IS_WIN and "\\" or "/"

-- Pin the stock Windows curl; PATH may resolve to an MSYS/Git curl that
-- misbehaves under ExecProcess (no console, different DLLs).
local CURL = "curl"
if IS_WIN then
  local c = (os.getenv("SystemRoot") or "C:\\Windows") .. "\\System32\\curl.exe"
  if reaper.file_exists(c) then CURL = '"' .. c .. '"' end
end

-- ---------------------------------------------------------------- config

local CFG_DIR = reaper.GetResourcePath() .. SEP .. "SampleTagger"
local CACHE_DIR = CFG_DIR .. SEP .. "cache"
local CFG_FILE = CFG_DIR .. SEP .. "config.json"
reaper.RecursiveCreateDirectory(CACHE_DIR, 0)

local DEFAULTS = {
  server = "http://192.168.178.56:8765",
  mode = "download",          -- "download" | "direct"
  remote_prefix = "/home/phlp/pcloud/DAW/Samples",
  local_prefix = "",          -- where the same library is mounted on THIS machine
  insert_mode = "cursor",     -- "cursor" | "new_track"
  norm_preview = true,        -- preview server-normalized mp3 instead of the raw file
  k = 48,                     -- results per page
}

local config = {}

local function load_config()
  for k, v in pairs(DEFAULTS) do config[k] = v end
  local f = io.open(CFG_FILE, "r")
  if not f then return end
  local ok, data = pcall(json.decode, f:read("*a"))
  f:close()
  if ok and type(data) == "table" then
    for k in pairs(DEFAULTS) do
      if data[k] ~= nil then config[k] = data[k] end
    end
  end
end

local function save_config()
  local f = io.open(CFG_FILE, "w")
  if not f then return end
  f:write(json.encode(config))
  f:close()
end

load_config()

-- ---------------------------------------------------------------- state

local state = {
  query = "",
  hits = {},            -- current result list (search or similar)
  total = nil,          -- library size reported by the server
  sel = nil,            -- selected hit index
  offset = 0,
  list_kind = "search", -- "search" | "similar" (for [more] pagination)
  status = PREVIEW_OK and "ready" or "ready (no SWS: preview disabled, insert only)",
  first_search_done = false,
  focus_query = true,
  f_inst = "all",
  f_type = "all",
  f_maxdur = 0.0,
  show_settings = false,
  preview_token = 0,    -- invalidates in-flight preview downloads on new clicks
}

local downloads = {}    -- final_path -> {part, url, started, deadline, cbs = {fn,...}}

-- ---------------------------------------------------------------- helpers

local function urlencode(s)
  return (s:gsub("[^%w%-%._~]", function(c)
    return string.format("%%%02X", c:byte())
  end))
end

local function trim(s) return s:match("^%s*(.-)%s*$") end

local function fnv1a(s)
  local h = 2166136261
  for i = 1, #s do
    h = (h ~ s:byte(i)) * 16777619 & 0xFFFFFFFF
  end
  return string.format("%08x", h)
end

local function best_label(hit)
  if hit.human_labels and hit.human_labels[1] then return hit.human_labels[1] end
  return hit.instrument or hit.model_instrument or hit.panns_instrument or "?"
end

local function read_small(path)
  local f = io.open(path, "rb")
  if not f then return nil end
  local s = f:read("*a")
  f:close()
  return s
end

-- ---------------------------------------------------------------- file access

-- Map a server-side sample path to this machine's mount (direct mode).
local function map_direct(remote)
  if config.local_prefix == "" then return nil end
  if remote:sub(1, #config.remote_prefix) ~= config.remote_prefix then return nil end
  local rel = remote:sub(#config.remote_prefix + 1)
  if config.local_prefix:find("\\") then rel = rel:gsub("/", "\\") end
  local p = config.local_prefix .. rel
  if reaper.file_exists(p) then return p end
  return nil
end

local function cache_path(remote, norm)
  local base = remote:match("([^/]+)$") or "sample"
  base = base:gsub('[<>:"/\\|%?%*]', "_")
  local ext = norm and ".norm.mp3" or ""
  return CACHE_DIR .. SEP .. fnv1a(remote) .. "_" .. base .. ext
end

-- Fire-and-forget download through a shell so ".part && rename" is atomic-ish:
-- the final file only ever appears complete, killed transfers leave only .part.
-- curl's stderr goes to <final>.err so failures can be reported with a reason.
local function start_download(url, final, timeout_s)
  local part, errf = final .. ".part", final .. ".err"
  local cmd
  if IS_WIN then
    -- /S /C "…" : cmd strips the outer quotes and runs the rest verbatim,
    -- which keeps the nested quoting around curl path + args intact
    cmd = ('cmd.exe /S /C "%s -sfS -m %d -o "%s" "%s" 2> "%s" && move /Y "%s" "%s""')
          :format(CURL, timeout_s, part, url, errf, part, final)
  else
    cmd = ("/bin/sh -c \"%s -sfS -m %d -o '%s' '%s' 2> '%s' && mv '%s' '%s'\"")
          :format(CURL, timeout_s, part, url, errf, part, final)
  end
  reaper.ExecProcess(cmd, -2)
  return part, errf
end

-- All HTTP goes through fire-and-forget curl + polling for the output file:
-- ExecProcess's blocking wait is broken on some Windows setups (returns
-- 259/STILL_ACTIVE while the same curl works fine in cmd), so we never wait
-- on a process — we wait for its file to appear. cb(final) on success,
-- fail(msg) on error (optional; defaults to a status-line message).
local function request(url, final, timeout_s, cb, fail)
  local dl = downloads[final]
  if dl then
    dl.cbs[#dl.cbs + 1] = cb
    return
  end
  local part, errf = start_download(url, final, timeout_s)
  downloads[final] = { part = part, err = errf, started = reaper.time_precise(),
                       deadline = reaper.time_precise() + timeout_s + 10,
                       cbs = { cb }, fail = fail }
end

local req_seq = 0
local function api_get_async(path_and_query, timeout_s, cb)
  req_seq = req_seq + 1
  local final = CACHE_DIR .. SEP .. ("req_%d_%d.json"):format(os.time(), req_seq)
  request(config.server .. path_and_query, final, timeout_s, function(f)
    local body = read_small(f)
    os.remove(f)
    if not body or body == "" then cb(nil, "empty response"); return end
    local ok, data = pcall(json.decode, body)
    if ok and data then cb(data) else cb(nil, "bad JSON from server") end
  end, function(msg) cb(nil, msg) end)
end

-- Get a playable/insertable local path for a hit, then call cb(localpath).
-- cb may fire immediately (direct mode / cache hit) or on a later defer tick.
local function ensure_local(remote, norm, cb)
  if config.mode == "direct" and not norm then
    local p = map_direct(remote)
    if p then cb(p); return end
    -- fall through to download if the mount doesn't have it
  end
  local final = cache_path(remote, norm)
  if reaper.file_exists(final) then cb(final); return end
  local url = config.server .. "/api/audio?path=" .. urlencode(remote) .. (norm and "&norm=1" or "")
  request(url, final, 180, cb)
  state.status = "downloading " .. (remote:match("([^/]+)$") or "")
end

local function poll_downloads()
  local now = reaper.time_precise()
  for final, dl in pairs(downloads) do
    local errtxt = read_small(dl.err)
    errtxt = errtxt and trim(errtxt) or ""
    if reaper.file_exists(final) then
      downloads[final] = nil
      os.remove(dl.err)
      for _, cb in ipairs(dl.cbs) do cb(final) end
    elseif (errtxt ~= "" and not reaper.file_exists(dl.part))
        or now > dl.deadline
        or (now > dl.started + 15 and not reaper.file_exists(dl.part)) then
      -- error text with no .part = curl failed outright (refused, 404, …);
      -- the two timeouts catch hangs and shells that never launched curl
      downloads[final] = nil
      os.remove(dl.err)
      os.remove(dl.part)
      local msg = errtxt ~= "" and errtxt:match("[^\r\n]+") or "timed out"
      if dl.fail then dl.fail(msg)
      else state.status = "download failed: " .. msg end
    end
  end
end

-- ---------------------------------------------------------------- preview

local preview = { handle = nil, src = nil }

local function stop_preview()
  if preview.handle then reaper.CF_Preview_Stop(preview.handle) end
  if preview.src then reaper.PCM_Source_Destroy(preview.src) end
  preview.handle, preview.src = nil, nil
end

local function play_file(path)
  if not PREVIEW_OK then return end
  stop_preview()
  local src = reaper.PCM_Source_CreateFromFile(path)
  if not src then state.status = "cannot open " .. path; return end
  local h = reaper.CF_CreatePreview(src)
  if reaper.CF_Preview_SetOutput then reaper.CF_Preview_SetOutput(h, 0) end
  reaper.CF_Preview_Play(h)
  preview.handle, preview.src = h, src
end

local function preview_hit(hit)
  if not PREVIEW_OK then return end
  state.preview_token = state.preview_token + 1
  local token = state.preview_token
  ensure_local(hit.path, config.norm_preview, function(localpath)
    if token ~= state.preview_token then return end  -- user clicked elsewhere meanwhile
    play_file(localpath)
    state.status = "▶ " .. hit.name
  end)
end

-- ---------------------------------------------------------------- insert

local function insert_hit(hit)
  ensure_local(hit.path, false, function(localpath)
    stop_preview()
    reaper.Undo_BeginBlock()
    reaper.InsertMedia(localpath, config.insert_mode == "new_track" and 1 or 0)
    reaper.Undo_EndBlock("Insert sample: " .. hit.name, -1)
    state.status = "inserted " .. hit.name
  end)
end

-- ---------------------------------------------------------------- queries

local function queue_search(append)
  local q = trim(state.query)
  if q == "" then return end
  local offset = append and state.offset + config.k or 0
  state.status = append and "loading more…" or "searching…"
  -- first query after a server restart loads the CLAP model → generous timeout
  api_get_async("/api/search_text?q=" .. urlencode(q)
                .. "&k=" .. config.k .. "&offset=" .. offset,
                state.first_search_done and 30 or 90,
    function(res, err)
      if not res then state.status = "search failed: " .. tostring(err); return end
      state.first_search_done = true
      if append then
        for _, h in ipairs(res.hits) do state.hits[#state.hits + 1] = h end
      else
        state.hits, state.sel = res.hits, nil
      end
      state.offset, state.total, state.list_kind = offset, res.n, "search"
      state.status = ("%d hits shown (%s files indexed)"):format(#state.hits, tostring(res.n))
    end)
end

local function queue_similar(hit)
  state.status = "finding similar…"
  api_get_async("/api/similar?path=" .. urlencode(hit.path) .. "&k=" .. config.k, 60,
    function(res, err)
      if not res then state.status = "similar failed: " .. tostring(err); return end
      state.hits, state.sel = res.hits, nil
      state.offset, state.list_kind = 0, "similar"
      state.status = ("%d similar to %s"):format(#state.hits, res.matched_name or hit.name)
    end)
end

-- ---------------------------------------------------------------- filters

local function visible_hits()
  local out = {}
  for i, h in ipairs(state.hits) do
    local ok = true
    if state.f_inst ~= "all" and best_label(h) ~= state.f_inst then ok = false end
    if ok and state.f_type ~= "all" then
      local t = h.sample_type
      if state.f_type == "other" then
        if t == "oneshot" or t == "loop" then ok = false end
      elseif t ~= state.f_type then ok = false end
    end
    if ok and state.f_maxdur > 0 and (h.duration_s or 0) > state.f_maxdur then ok = false end
    if ok then out[#out + 1] = i end
  end
  return out
end

local function instrument_options()
  local seen, opts = {}, { "all" }
  for _, h in ipairs(state.hits) do
    local l = best_label(h)
    if not seen[l] then seen[l] = true; opts[#opts + 1] = l end
  end
  table.sort(opts, function(a, b)
    if a == "all" then return true end
    if b == "all" then return false end
    return a < b
  end)
  return opts
end

-- ---------------------------------------------------------------- UI

local ctx = reaper.ImGui_CreateContext("SampleTagger")
if reaper.ImGui_SetConfigVar and reaper.ImGui_ConfigVar_Flags and reaper.ImGui_ConfigFlags_DockingEnable then
  reaper.ImGui_SetConfigVar(ctx, reaper.ImGui_ConfigVar_Flags(), reaper.ImGui_ConfigFlags_DockingEnable())
end

local function combo(label, current, options, width)
  local changed = nil
  reaper.ImGui_SetNextItemWidth(ctx, width or 110)
  if reaper.ImGui_BeginCombo(ctx, label, current) then
    for _, opt in ipairs(options) do
      if reaper.ImGui_Selectable(ctx, opt, opt == current) then changed = opt end
    end
    reaper.ImGui_EndCombo(ctx)
  end
  return changed
end

local function draw_settings()
  local rv
  rv, config.server = reaper.ImGui_InputText(ctx, "server", config.server)
  local m = combo("file access##mode", config.mode, { "download", "direct" }, 120)
  if m then config.mode = m end
  if config.mode == "direct" then
    rv, config.remote_prefix = reaper.ImGui_InputText(ctx, "remote prefix", config.remote_prefix)
    rv, config.local_prefix = reaper.ImGui_InputText(ctx, "local prefix", config.local_prefix)
  end
  local im = combo("insert##imode", config.insert_mode, { "cursor", "new_track" }, 120)
  if im then config.insert_mode = im end
  reaper.ImGui_SetNextItemWidth(ctx, 80)
  rv, config.k = reaper.ImGui_InputInt(ctx, "results per page", config.k)
  if config.k < 1 then config.k = 1 end
  if reaper.ImGui_Button(ctx, "Save settings") then
    save_config()
    state.status = "settings saved"
    state.show_settings = false
  end
end

local function draw_results()
  local footer = reaper.ImGui_GetFrameHeightWithSpacing(ctx) * 2 + 6
  local flags = reaper.ImGui_TableFlags_ScrollY()
              | reaper.ImGui_TableFlags_RowBg()
              | reaper.ImGui_TableFlags_Resizable()
              | reaper.ImGui_TableFlags_BordersInnerV()
  if not reaper.ImGui_BeginTable(ctx, "results", 5, flags, 0, -footer) then return end
  reaper.ImGui_TableSetupScrollFreeze(ctx, 0, 1)
  reaper.ImGui_TableSetupColumn(ctx, "name", reaper.ImGui_TableColumnFlags_WidthStretch())
  reaper.ImGui_TableSetupColumn(ctx, "type", reaper.ImGui_TableColumnFlags_WidthFixed(), 55)
  reaper.ImGui_TableSetupColumn(ctx, "label", reaper.ImGui_TableColumnFlags_WidthFixed(), 80)
  reaper.ImGui_TableSetupColumn(ctx, "dur", reaper.ImGui_TableColumnFlags_WidthFixed(), 48)
  reaper.ImGui_TableSetupColumn(ctx, "score", reaper.ImGui_TableColumnFlags_WidthFixed(), 45)
  reaper.ImGui_TableHeadersRow(ctx)

  local sel_flags = reaper.ImGui_SelectableFlags_SpanAllColumns()
                  | reaper.ImGui_SelectableFlags_AllowDoubleClick()
  for _, i in ipairs(visible_hits()) do
    local h = state.hits[i]
    reaper.ImGui_TableNextRow(ctx)
    reaper.ImGui_TableNextColumn(ctx)
    if reaper.ImGui_Selectable(ctx, ("%s##%d"):format(h.name, i), i == state.sel, sel_flags) then
      state.sel = i
      if reaper.ImGui_IsMouseDoubleClicked(ctx, 0) then insert_hit(h) else preview_hit(h) end
    end
    if reaper.ImGui_IsItemHovered(ctx) then
      reaper.ImGui_SetTooltip(ctx, h.path .. (h.bpm and ("\nbpm: " .. h.bpm) or "")
                                   .. (h.key and ("  key: " .. h.key) or ""))
    end
    reaper.ImGui_TableNextColumn(ctx); reaper.ImGui_Text(ctx, h.sample_type or "?")
    reaper.ImGui_TableNextColumn(ctx); reaper.ImGui_Text(ctx, best_label(h))
    reaper.ImGui_TableNextColumn(ctx)
    reaper.ImGui_Text(ctx, h.duration_s and ("%.2f"):format(h.duration_s) or "")
    reaper.ImGui_TableNextColumn(ctx)
    reaper.ImGui_Text(ctx, h.score and ("%.3f"):format(h.score) or "")
  end
  reaper.ImGui_EndTable(ctx)
end

local function draw()
  local rv

  -- search row
  if state.focus_query then reaper.ImGui_SetKeyboardFocusHere(ctx); state.focus_query = false end
  reaper.ImGui_SetNextItemWidth(ctx, -132)
  rv, state.query = reaper.ImGui_InputText(ctx, "##query", state.query,
                                           reaper.ImGui_InputTextFlags_EnterReturnsTrue())
  if rv then queue_search(false) end
  reaper.ImGui_SameLine(ctx)
  if reaper.ImGui_Button(ctx, "Search") then queue_search(false) end
  reaper.ImGui_SameLine(ctx)
  if reaper.ImGui_Button(ctx, "⚙") then state.show_settings = not state.show_settings end

  if state.show_settings then
    draw_settings()
    reaper.ImGui_Separator(ctx)
  end

  -- filter row
  if #state.hits > 0 then
    local ni = combo("##finst", state.f_inst, instrument_options(), 100)
    if ni then state.f_inst = ni end
    reaper.ImGui_SameLine(ctx)
    local nt = combo("##ftype", state.f_type, { "all", "oneshot", "loop", "other" }, 85)
    if nt then state.f_type = nt end
    reaper.ImGui_SameLine(ctx)
    reaper.ImGui_SetNextItemWidth(ctx, 70)
    rv, state.f_maxdur = reaper.ImGui_InputDouble(ctx, "max s", state.f_maxdur, 0, 0, "%.1f")
    if state.f_maxdur < 0 then state.f_maxdur = 0 end
    if PREVIEW_OK then
      reaper.ImGui_SameLine(ctx)
      rv, config.norm_preview = reaper.ImGui_Checkbox(ctx, "norm", config.norm_preview)
    end
  end

  draw_results()

  -- footer
  local sel = state.sel and state.hits[state.sel]
  if sel then
    if reaper.ImGui_Button(ctx, "Insert") then insert_hit(sel) end
    reaper.ImGui_SameLine(ctx)
    if reaper.ImGui_Button(ctx, "Similar") then queue_similar(sel) end
    reaper.ImGui_SameLine(ctx)
    if PREVIEW_OK and reaper.ImGui_Button(ctx, "Stop") then stop_preview() end
    reaper.ImGui_SameLine(ctx)
  end
  if #state.hits > 0 and state.list_kind == "search" then
    if reaper.ImGui_Button(ctx, "more") then queue_search(true) end
    reaper.ImGui_SameLine(ctx)
  end
  reaper.ImGui_Text(ctx, state.status)
end

-- ---------------------------------------------------------------- main loop

local function loop()
  poll_downloads()

  reaper.ImGui_SetNextWindowSize(ctx, 460, 560, reaper.ImGui_Cond_FirstUseEver())
  local visible, open = reaper.ImGui_Begin(ctx, "SampleTagger", true)
  if visible then
    draw()
    reaper.ImGui_End(ctx)
  end
  if open then
    reaper.defer(loop)
  else
    stop_preview()
  end
end

reaper.defer(loop)
