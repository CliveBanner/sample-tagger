-- Minimal pure-Lua JSON decode/encode for sampletagger_search.lua.
-- Decode covers the full spec (incl. \uXXXX + surrogate pairs); null becomes nil.
-- Encode is only what config saving needs (flat tables of string/number/bool).

local json = {}

-- ---------------------------------------------------------------- decode

local str, pos

local function decode_error(msg)
  error(("json: %s at byte %d"):format(msg, pos), 0)
end

local function skip_ws()
  pos = str:find("[^ \t\r\n]", pos) or #str + 1
end

local function utf8_char(cp)
  if cp < 0x80 then
    return string.char(cp)
  elseif cp < 0x800 then
    return string.char(0xC0 | (cp >> 6), 0x80 | (cp & 0x3F))
  elseif cp < 0x10000 then
    return string.char(0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F))
  else
    return string.char(0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3F),
                       0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F))
  end
end

local escapes = { ['"'] = '"', ['\\'] = '\\', ['/'] = '/',
                  b = '\b', f = '\f', n = '\n', r = '\r', t = '\t' }

local function parse_string()
  pos = pos + 1  -- opening quote
  local out = {}
  while true do
    local c = str:sub(pos, pos)
    if c == '' then decode_error("unterminated string") end
    if c == '"' then pos = pos + 1; break end
    if c == '\\' then
      local e = str:sub(pos + 1, pos + 1)
      if escapes[e] then
        out[#out + 1] = escapes[e]; pos = pos + 2
      elseif e == 'u' then
        local hex = str:sub(pos + 2, pos + 5)
        local cp = tonumber(hex, 16) or decode_error("bad \\u escape")
        pos = pos + 6
        if cp >= 0xD800 and cp <= 0xDBFF and str:sub(pos, pos + 1) == '\\u' then
          local lo = tonumber(str:sub(pos + 2, pos + 5), 16)
          if lo and lo >= 0xDC00 and lo <= 0xDFFF then
            cp = 0x10000 + (cp - 0xD800) * 0x400 + (lo - 0xDC00)
            pos = pos + 6
          end
        end
        out[#out + 1] = utf8_char(cp)
      else
        decode_error("bad escape \\" .. e)
      end
    else
      local nxt = str:find('["\\]', pos) or decode_error("unterminated string")
      out[#out + 1] = str:sub(pos, nxt - 1)
      pos = nxt
    end
  end
  return table.concat(out)
end

local function parse_number()
  local num = str:match("^-?%d+%.?%d*[eE]?[%+%-]?%d*", pos)
  local v = tonumber(num)
  if not v then decode_error("bad number") end
  pos = pos + #num
  return v
end

local parse_value

local function parse_array()
  pos = pos + 1
  local arr, n = {}, 0
  skip_ws()
  if str:sub(pos, pos) == ']' then pos = pos + 1; return arr end
  while true do
    n = n + 1
    arr[n] = parse_value()
    skip_ws()
    local c = str:sub(pos, pos)
    pos = pos + 1
    if c == ']' then break end
    if c ~= ',' then decode_error("expected ',' or ']'") end
    skip_ws()
  end
  return arr
end

local function parse_object()
  pos = pos + 1
  local obj = {}
  skip_ws()
  if str:sub(pos, pos) == '}' then pos = pos + 1; return obj end
  while true do
    if str:sub(pos, pos) ~= '"' then decode_error("expected object key") end
    local key = parse_string()
    skip_ws()
    if str:sub(pos, pos) ~= ':' then decode_error("expected ':'") end
    pos = pos + 1
    skip_ws()
    obj[key] = parse_value()
    skip_ws()
    local c = str:sub(pos, pos)
    pos = pos + 1
    if c == '}' then break end
    if c ~= ',' then decode_error("expected ',' or '}'") end
    skip_ws()
  end
  return obj
end

parse_value = function()
  local c = str:sub(pos, pos)
  if c == '{' then return parse_object() end
  if c == '[' then return parse_array() end
  if c == '"' then return parse_string() end
  if str:sub(pos, pos + 3) == 'true' then pos = pos + 4; return true end
  if str:sub(pos, pos + 4) == 'false' then pos = pos + 5; return false end
  if str:sub(pos, pos + 3) == 'null' then pos = pos + 4; return nil end
  if c:match('[%-%d]') then return parse_number() end
  decode_error("unexpected character " .. (c == '' and "<eof>" or c))
end

function json.decode(s)
  str, pos = s, 1
  skip_ws()
  local v = parse_value()
  str = nil
  return v
end

-- ---------------------------------------------------------------- encode

local esc_map = { ['"'] = '\\"', ['\\'] = '\\\\', ['\b'] = '\\b',
                  ['\f'] = '\\f', ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t' }

local function encode_string(s)
  return '"' .. s:gsub('[%c"\\]', function(c)
    return esc_map[c] or string.format('\\u%04x', c:byte())
  end) .. '"'
end

local function encode_value(v, indent)
  local t = type(v)
  if t == 'string' then return encode_string(v) end
  if t == 'number' or t == 'boolean' then return tostring(v) end
  if t == 'nil' then return 'null' end
  if t == 'table' then
    if v[1] ~= nil or next(v) == nil then  -- array (or empty)
      local parts = {}
      for i = 1, #v do parts[i] = encode_value(v[i], indent) end
      return '[' .. table.concat(parts, ', ') .. ']'
    end
    local keys = {}
    for k in pairs(v) do keys[#keys + 1] = k end
    table.sort(keys)
    local parts = {}
    for _, k in ipairs(keys) do
      parts[#parts + 1] = indent .. '  ' .. encode_string(k) .. ': '
                          .. encode_value(v[k], indent .. '  ')
    end
    return '{\n' .. table.concat(parts, ',\n') .. '\n' .. indent .. '}'
  end
  error('json: cannot encode ' .. t)
end

function json.encode(v)
  return encode_value(v, '')
end

return json
