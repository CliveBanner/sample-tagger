import os
import re
import json
import urllib.parse
from . import state
from . import runs
from . import labeling
from . import sonic
from . import mapview
from . import stats
from . import gold

def _qs(req):
    return urllib.parse.parse_qs(urllib.parse.urlparse(req.path).query)

def q_str(req, key, default=""):
    return (_qs(req).get(key) or [default])[0]

def q_int(req, key, default=0):
    return int((_qs(req).get(key) or [default])[0])

def serve_audio(req, path):
    if not path or not state.valid_sample(path) or not os.path.isfile(path):
        req._send(404, "not found", "text/plain"); return
    ct = state.AUDIO_CT.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
    size = os.path.getsize(path)
    start, end, status = 0, size - 1, 200
    rng = req.headers.get("Range")
    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if m:
            if m.group(1):
                start = int(m.group(1))
            if m.group(2):
                end = int(m.group(2))
            end = min(end, size - 1); start = min(start, end); status = 206
    length = end - start + 1
    req.send_response(status)
    req.send_header("Content-Type", ct)
    req.send_header("Accept-Ranges", "bytes")
    req.send_header("Content-Length", str(length))
    if status == 206:
        req.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    req.end_headers()
    try:
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                req.wfile.write(chunk)
                remaining -= len(chunk)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass

def serve_audio_normalized(req, path):
    if not path or not state.valid_sample(path) or not os.path.isfile(path):
        req._send(404, "not found", "text/plain"); return
    import subprocess
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path,
         "-af", "volumedetect", "-vn", "-sn", "-dn", "-f", "null", "/dev/null"],
        capture_output=True, text=True)
    peak_db = 0.0
    for line in probe.stderr.splitlines():
        m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", line)
        if m:
            peak_db = float(m.group(1)); break
    gain_db = min(-1.0 - peak_db, 30.0)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", path,
           "-af", f"volume={gain_db:.2f}dB",
           "-ar", "44100", "-ac", "2",
           "-c:a", "libmp3lame", "-q:a", "4",
           "-f", "mp3", "pipe:1"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        serve_audio(req, path); return
    req.send_response(200)
    req.send_header("Content-Type", "audio/mpeg")
    req.send_header("Cache-Control", "no-store")
    req.end_headers()
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            req.wfile.write(chunk)
    except (BrokenPipeError, ConnectionResetError, OSError):
        proc.kill()
    finally:
        proc.wait()

def _json(req, obj, code=200):
    req._send(code, json.dumps(obj), "application/json")

GET_ROUTES = {
    "/api/labels": state.get_labels,
    "/api/colors": state.get_colors,
    "/api/weakmap": state.get_weakmap,
    "/api/review/queue": lambda req: labeling.review_queue(q_str(req, "mode", "disagree")),
    "/api/config": state.load_config,
    "/api/run/status": runs.run_status,
    "/api/run/ml/status": runs.ml_run_status,
    "/api/stats": stats.stats,
    "/api/log": runs.log_tail,
    "/api/errors": stats.recent_errors,
    "/api/similar": lambda req: mapview.similar_api(q_str(req, "path") or q_str(req, "q"), q_int(req, "k", 24)),
    "/api/search_text": lambda req: mapview.search_text_api(q_str(req, "q"), q_int(req, "k", 24), q_int(req, "offset", 0)),
    "/api/propagate": lambda req: mapview.propagate_candidates(q_str(req, "path"), q_int(req, "k", 24)),
    "/api/sonic/families": sonic.sonic_families,
    "/api/sonic/grains": lambda req: sonic.sonic_grains(q_int(req, "family", -1)),
    "/api/sonic/members": lambda req: sonic.sonic_members(q_int(req, "grain", -1)),
    "/api/map": mapview.map_api,
    "/api/point": lambda req: mapview.point_api(q_int(req, "i", -1)),
    "/api/reproject": mapview.reproject_start,
    "/api/reproject_status": lambda req: state._REPROJ,
    "/api/gold/status": gold.gold_status,
    "/api/ml/metrics": gold.ml_metrics,
}

def p_str(data, key, default=""):
    return data.get(key, default)

def p_int(data, key, default=0):
    try: return int(data.get(key, default))
    except (ValueError, TypeError): return default

POST_ROUTES = {
    "/api/config": lambda data: state.save_config(data),
    "/api/run/start": lambda data: {"ok": False, "msg": "use /api/run/discover or /api/run/label"},
    "/api/run/stop": lambda data: runs.run_stop(),
    "/api/run/discover": lambda data: runs.run_start("discover"),
    "/api/run/label": lambda data: runs.run_start("label"),
    "/api/run/ml": lambda data: runs.ml_run_start(),
    "/api/run/ml/stop": lambda data: runs.ml_run_stop(),
    "/api/label": lambda data: labeling.label_api(
        p_str(data, "path"),
        data.get("labels") if data.get("labels") is not None
        else [l for l in [p_str(data, "instrument"), p_str(data, "instrument2")] if l]),
    "/api/label_type": lambda data: labeling.label_type_api(p_str(data, "path"), p_str(data, "sample_type")),
    "/api/rate": lambda data: labeling.rate_api(p_str(data, "path"), p_int(data, "rating")),
    "/api/label_propagate": lambda data: labeling.label_propagate(
        data.get("paths", []), data.get("labels") or p_str(data, "instrument")),
    "/api/labels/add": lambda data: labeling.add_label(p_str(data, "name")),
    "/api/labels/delete": lambda data: labeling.delete_label(p_str(data, "name")),
    "/api/label_map": lambda data: labeling.label_map(data),
    "/api/gold/sample": gold.gold_sample,
    "/api/gold/freeze": gold.gold_freeze,
}

def _read_body(req):
    try:
        length = int(req.headers.get("Content-Length", 0))
        if length > 0:
            return req.rfile.read(length).decode("utf-8")
    except (ValueError, OSError):
        pass
    return ""

def handle_post(req, route):
    body = _read_body(req)
    if route in POST_ROUTES:
        try:
            data = json.loads(body) if body else {}
            _json(req, POST_ROUTES[route](data))
        except Exception as e:
            req._send(400, str(e), "text/plain")
    else:
        req._send(404, "not found", "text/plain")

def handle_get(req, route):
    if route == "/api/audio":
        qs = _qs(req)
        path = (qs.get("path") or [""])[0]
        if (qs.get("norm") or [""])[0] == "1":
            serve_audio_normalized(req, path)
        else:
            serve_audio(req, path)
        return

    if route in GET_ROUTES:
        try:
            handler = GET_ROUTES[route]
            if route in ["/api/review/queue", "/api/similar", "/api/search_text", "/api/propagate", "/api/point", "/api/reproject_status", "/api/sonic/grains", "/api/sonic/members"]:
                _json(req, handler(req))
            else:
                _json(req, handler())
        except Exception as e:
            _json(req, {"ready": False, "msg": str(e)}, code=500)
    else:
        req._send(404, "not found", "text/plain")
