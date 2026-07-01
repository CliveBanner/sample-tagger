import os
import time
from dataclasses import dataclass
from .constants import ANALYZE_SECONDS, PANNS_MIN_DURATION
from .paths import parse_path_hints, is_sfx, FAST_NO_AUDIO, ONESHOT_DRUMS, DRUM_NONTONAL
from .audio import _true_duration, load_audio, harmonic_ratio, count_onsets, detect_bpm, classify_loop, detect_key, classify_instrument_audio
from .panns import _panns_forward

@dataclass
class Analysis:
    path: str
    mtime: float
    size: int
    duration_s: float
    sample_type: str
    bpm: "int | None"
    key: "str | None"
    tonal: str
    instrument: str
    source: str                    # who decided instrument: "path"|"audio"|"panns"|"human"|"none"
    status: str = "ok"
    error: str = ""
    emb: "bytes | None" = None     # 2048-d PANNs embedding (float32 bytes) for similarity
    path_instrument: "str | None" = None   # what folder/filename heuristics said
    panns_instrument: "str | None" = None  # what CNN14 said (even if path won)
    panns_conf: "float | None" = None      # CNN14 confidence for its top mapped label
    audio_instrument: "str | None" = None  # what spectral heuristics said



def analyze_file(path):
    """Full pipeline for one file. Never raises; returns Analysis(status=...)."""
    from . import workers
    try:
        st = os.stat(path)
        mtime, size = st.st_mtime, st.st_size
    except OSError as e:
        return Analysis(path, 0, 0, 0.0, "oneshot", None, None, "percussive",
                        "fx", "path", status="error", error=f"stat: {e}")

    hints = parse_path_hints(path)
    sfx = is_sfx(path)
    path_drum = (hints.get("instrument") in FAST_NO_AUDIO
                 and hints.get("sample_type") != "loop")

    # No-decode shortcuts — but only when we're NOT capturing embeddings
    # and NOT in panns-only mode (which needs to decode to run PANNs).
    if not workers._EMBED and not workers._PANNS_ONLY:
        # SFX/speech/field-recording: bpm & key meaningless, spectral is noise.
        if sfx:
            return Analysis(path, mtime, size, 0.0,
                hints.get("sample_type", "oneshot"), None, None,
                "percussive", "fx", "path")
        # non-pitched drum one-shot named by folder/file: path says it all.
        if path_drum:
            return Analysis(path, mtime, size, 0.0,
                "oneshot", None, hints.get("key"), "percussive",
                hints["instrument"], "path")

    try:
        duration_s = _true_duration(path)
        y, sr = load_audio(path, duration=ANALYZE_SECONDS)
        if duration_s is None:
            import librosa
            duration_s = float(librosa.get_duration(y=y, sr=sr))
        if y.size == 0:
            raise ValueError("empty audio")

        # One CNN14 forward pass, reused for both the embedding and (when the
        # file is unlabeled) the instrument prediction.
        want_class = (workers._USE_PANNS and not sfx and duration_s >= PANNS_MIN_DURATION)
        inst_panns, panns_conf, emb_arr = None, None, None
        if workers._EMBED or want_class:
            inst_panns, panns_conf, emb_arr = _panns_forward(y, sr)
        emb_bytes = emb_arr.tobytes() if emb_arr is not None else None
        dur = round(duration_s, 3)
        path_instr = hints.get("instrument")   # always record what the path said

        if sfx and not workers._PANNS_ONLY:              # cheap class, but we decoded to embed
            a = Analysis(path, mtime, size, dur, hints.get("sample_type", "oneshot"),
                         None, None, "percussive", "fx", "path",
                         path_instrument=path_instr, panns_instrument=inst_panns,
                         panns_conf=panns_conf)
        elif path_drum and not workers._PANNS_ONLY:
            instr = (inst_panns or hints["instrument"]) if inst_panns is not None else hints["instrument"]
            src   = "panns" if inst_panns is not None else "path"
            a = Analysis(path, mtime, size, dur, "oneshot", None, hints.get("key"),
                         "percussive", instr, src,
                         path_instrument=path_instr, panns_instrument=inst_panns,
                         panns_conf=panns_conf)
        else:
            import librosa
            h, p = librosa.effects.hpss(y)
            hr = harmonic_ratio(h, p)
            is_tonal = hr >= HARMONIC_RATIO_TONAL
            n_onsets = count_onsets(p, sr)
            bpm_raw = detect_bpm(y, sr)
            audio_instr = classify_instrument_audio(y, sr, is_tonal, duration_s)

            if inst_panns is not None:
                instrument, source = inst_panns, "panns"
            elif workers._PANNS_ONLY:
                instrument, source = None, "none"
            elif "instrument" in hints:
                instrument, source = hints["instrument"], "path"
            else:
                instrument, source = audio_instr, "audio"

            if hints.get("sample_type"):
                sample_type = hints["sample_type"]
            elif source == "path" and instrument in ONESHOT_DRUMS and duration_s < 4.0:
                sample_type = "oneshot"
            else:
                sample_type = classify_loop(duration_s, n_onsets, bpm_raw, hr)

            if hints.get("bpm"):
                bpm = hints["bpm"]
            elif sample_type == "loop" and bpm_raw and BPM_MIN <= bpm_raw <= BPM_MAX:
                bpm = int(round(bpm_raw))
            else:
                bpm = None

            if hints.get("key"):
                key = hints["key"]
            elif instrument in DRUM_NONTONAL:
                key = None
            else:
                key = detect_key(y, sr, is_tonal)

            a = Analysis(path, mtime, size, dur, sample_type, bpm, key,
                         "tonal" if is_tonal else "percussive", instrument, source,
                         path_instrument=path_instr, panns_instrument=inst_panns,
                         panns_conf=panns_conf, audio_instrument=audio_instr)

        a.emb = emb_bytes
        return a
    except Exception as e:
        # Couldn't decode/analyze.
        # In panns-only mode don't fall back to path hints — leave unclassified.
        instr = None if workers._PANNS_ONLY else hints.get("instrument", "fx")
        src   = "none" if workers._PANNS_ONLY else ("path" if hints else "none")
        return Analysis(
            path, mtime, size, 0.0,
            hints.get("sample_type", "oneshot"), hints.get("bpm"), hints.get("key"),
            "percussive", instr, src,
            status="error", error=f"{type(e).__name__}: {e}",
            path_instrument=hints.get("instrument"))

