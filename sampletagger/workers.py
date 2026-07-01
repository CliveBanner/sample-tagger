import os
from .paths import parse_path_hints
from .audio import _true_duration, load_audio, harmonic_ratio, classify_instrument_audio
from .constants import SR, DIM, AUDIO_EXTS
from .config import cfg
from .panns import _panns_forward

_USE_PANNS = False
_EMBED = False
_DO_AUDIO = False
_DO_PANNS = False

def discover_one(path):
    """Stat one file and extract path hints. No audio I/O."""
    try:
        st = os.stat(path)
        hints = parse_path_hints(path)
        return {"path": path, "ok": True,
                "mtime": st.st_mtime, "size": st.st_size,
                "path_instrument": hints.get("instrument")}
    except OSError as e:
        return {"path": path, "ok": False, "error": str(e)}


def _init_label_worker(do_audio, do_panns):
    global _DO_AUDIO, _DO_PANNS, _USE_PANNS, _EMBED
    _DO_AUDIO = do_audio
    _DO_PANNS = do_panns
    _USE_PANNS = do_panns
    _EMBED = do_panns


def label_one(path):
    """Decode audio once and run requested classifiers. Returns a dict of
    column → value pairs; only the requested classifiers are included."""
    result = {"path": path, "status": "ok", "error": ""}
    if not (_DO_AUDIO or _DO_PANNS):
        # path-only: re-derive from filename (no decode needed)
        result["path_instrument"] = parse_path_hints(path).get("instrument")
        result["_missing"] = True   # signal to always write (redo guard handled above)
        return result
    try:
        duration_s = _true_duration(path)
        y, sr = load_audio(path, duration=cfg.analyze_seconds)
        if duration_s is None:
            import librosa
            duration_s = float(librosa.get_duration(y=y, sr=sr))
        if y.size == 0:
            raise ValueError("empty audio")

        if _DO_PANNS and duration_s >= cfg.panns_min_duration:
            inst, conf, emb = _panns_forward(y, sr)
            result["panns_instrument"] = inst
            result["panns_conf"] = conf
            result["emb"] = emb.tobytes() if emb is not None else None

        if _DO_AUDIO:
            import librosa
            h, p = librosa.effects.hpss(y)
            hr = harmonic_ratio(h, p)
            is_tonal = hr >= cfg.harmonic_ratio_tonal
            result["audio_instrument"] = classify_instrument_audio(
                y, sr, is_tonal, duration_s)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
    return result

