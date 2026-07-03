import os
import re
import json
import urllib.parse
from . import state
from . import runs
from . import labeling
from . import clusters
from . import mapview
from . import stats
from . import gold

def _qs(req):
    return urllib.parse.parse_qs(urllib.parse.urlparse(req.path).query)

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
    "/api/review/queue": lambda req: labeling.review_queue((_qs(req).get("mode") or ["disagree"])[0]),
    "/api/config": state.load_config,
    "/api/run/status": runs.run_status,
    "/api/run/ml/status": runs.ml_run_status,
    "/api/stats": stats.stats,
    "/api/log": runs.log_tail,
    "/api/errors": stats.recent_errors,
    "/api/similar": lambda req: mapview.similar_api((_qs(req).get("path") or _qs(req).get("q") or [""])[0], int((_qs(req).get("k") or ["24"])[0])),
    "/api/search_text": lambda req: mapview.search_text_api((_qs(req).get("q") or [""])[0], int((_qs(req).get("k") or ["24"])[0])),
    "/api/propagate": lambda req: mapview.propagate_candidates((_qs(req).get("path") or [""])[0], int((_qs(req).get("k") or ["24"])[0])),
    "/api/clusters": lambda req: clusters.clusters_list((_qs(req).get("mode") or ["value"])[0], int((_qs(req).get("limit") or ["300"])[0])),
    "/api/sonic/families": clusters.sonic_families,
    "/api/sonic/grains": lambda req: clusters.sonic_grains(int((_qs(req).get("family") or ["-1"])[0])),
    "/api/sonic/members": lambda req: clusters.sonic_members(int((_qs(req).get("grain") or ["-1"])[0])),
    "/api/cluster": lambda req: clusters.cluster_detail(int((_qs(req).get("id") or ["-1"])[0])),
    "/api/map": mapview.map_api,
    "/api/point": lambda req: mapview.point_api(int((_qs(req).get("i") or ["-1"])[0])),
    "/api/reproject": mapview.reproject_start,
    "/api/reproject_status": lambda req: state._REPROJ,
    "/api/gold/status": gold.gold_status,
    "/api/ml/metrics": gold.ml_metrics,
}

POST_ROUTES = {
    "/api/config": lambda data: state.save_config(data),
    "/api/run/start": lambda data: {"ok": False, "msg": "use /api/run/discover or /api/run/label"},
    "/api/run/stop": lambda data: runs.run_stop(),
    "/api/run/discover": lambda data: runs.run_start("discover"),
    "/api/run/label": lambda data: runs.run_start("label"),
    "/api/run/ml": lambda data: runs.ml_run_start(),
    "/api/run/ml/stop": lambda data: runs.ml_run_stop(),
    "/api/label": lambda data: labeling.label_api(
        data.get("path",""),
        data.get("labels") if data.get("labels") is not None
        else [l for l in [data.get("instrument",""), data.get("instrument2")] if l]),
    "/api/label_type": lambda data: labeling.label_type_api(data.get("path",""), data.get("sample_type","")),
    "/api/rate": lambda data: labeling.rate_api(data.get("path",""), data.get("rating", 0)),
    "/api/label_propagate": lambda data: labeling.label_propagate(
        data.get("paths", []), data.get("labels") or data.get("instrument","")),
    "/api/label_cluster": lambda data: labeling.label_cluster(int(data.get("cluster_id", -1)), data.get("instrument",""), data.get("exclude", [])),
    "/api/labels/add": lambda data: labeling.add_label(data.get("name","")),
    "/api/labels/delete": lambda data: labeling.delete_label(data.get("name","")),
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
            if route in ["/api/review/queue", "/api/similar", "/api/search_text", "/api/propagate", "/api/clusters", "/api/cluster", "/api/point", "/api/reproject_status", "/api/sonic/grains", "/api/sonic/members"]:
                _json(req, handler(req))
            else:
                _json(req, handler())
        except Exception as e:
            _json(req, {"ready": False, "msg": str(e)}, code=500)
    else:
        req._send(404, "not found", "text/plain")
