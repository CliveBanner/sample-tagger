#!/usr/bin/env python3
"""
sample_tagger.py — Two-stage audio sample classifier.

Stage 1 — Discover (--stage discover):
  Walk the filesystem, register new files (status='new'), mark deleted files
  (status='missing'). Path hints are extracted for free during this step.
  No audio decoding.

Stage 2 — Label (--stage label):
  Compute one or more of:
    path   — folder/filename heuristics (instant, re-run of discovery hints)
    audio  — spectral analysis via librosa (medium speed, no model)
    panns  — CNN14 neural net via PANNs (slow, ~1 GB RAM/worker, also produces
              the 2048-d embedding for the similarity map)
  Results are stored in separate columns (path_instrument, audio_instrument,
  panns_instrument). The final instrument/source label is set only by the human
  or the trained classifier — never automatically by this script.

  --redo {path|audio|panns|all}  Force-overwrite existing labels for that
  classifier (use when updating a model or fixing a bug in the logic).

Usage:
  sample_tagger.py /path --stage discover              # find new/missing files
  sample_tagger.py /path --stage discover --no-cache  # force fresh walk
  sample_tagger.py /path --stage label --classifiers panns,audio -j 4
  sample_tagger.py /path --stage label --classifiers panns --redo panns
"""

import argparse
import os
import re
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
SR = 22050                  # analysis samplerate
DIM = 2048                  # PANNs CNN14 embedding dimensionality
ANALYZE_SECONDS = 30.0      # only decode the first N seconds for spectral work
LOOP_MIN_SEC = 0.8
LOOP_BAR_TOLERANCE = 0.12
HARMONIC_RATIO_TONAL = 0.35
BPM_MIN = 60
BPM_MAX = 200
PANNS_MIN_DURATION = 1.0    # skip PANNs on clips shorter than this (unreliable)
AUDIO_EXTS = {".wav", ".mp3", ".aif", ".aiff", ".flac"}

# ---- load config.json overrides (set via the web settings page) ----
try:
    import json as _json
    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(_cfg_path) as _f:
        _cfg = _json.load(_f)
    ANALYZE_SECONDS    = float(_cfg.get("analyze_seconds",       ANALYZE_SECONDS))
    LOOP_MIN_SEC       = float(_cfg.get("loop_min_sec",          LOOP_MIN_SEC))
    LOOP_BAR_TOLERANCE = float(_cfg.get("loop_bar_tolerance",    LOOP_BAR_TOLERANCE))
    HARMONIC_RATIO_TONAL = float(_cfg.get("harmonic_ratio_tonal", HARMONIC_RATIO_TONAL))
    BPM_MIN            = int(_cfg.get("bpm_min",                 BPM_MIN))
    BPM_MAX            = int(_cfg.get("bpm_max",                 BPM_MAX))
    PANNS_MIN_DURATION = float(_cfg.get("panns_min_duration",    PANNS_MIN_DURATION))
    del _json, _cfg, _cfg_path, _f
except Exception:
    pass

KRUMHANSL_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KRUMHANSL_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


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


# ----------------------------------------------------------------------------
# Path / filename priors  — cheap, often more reliable than spectral guesses
# ----------------------------------------------------------------------------
# order matters: more specific first
_INSTRUMENT_PATTERNS = [
    ("808",    r"\b808s?\b|sub ?bass"),
    ("kick",   r"\bkick|\bbd\b|bass ?drum|\bkik\b"),
    ("snare",  r"\bsnare|\bsd\b|\brim|rimshot"),
    ("clap",   r"\bclap|\bclp\b"),
    ("hihat",  r"\bhi ?hats?|\bhats?\b|\bhh\b|open ?hat|closed ?hat|\bohh\b|\bchh\b"),
    ("cymbal", r"\bcymbal|\bcrash|\bride\b|\bsplash"),
    ("tom",    r"\btoms?\b"),
    ("perc",   r"\bperc|percussion|conga|bongo|shaker|tambourine|\bclave|cowbell|woodblock"),
    ("vocal",  r"\bvocal|\bvox\b|\bacapella|acappella|\bchant|\bchoir|\bphrase\b|\badlib"),
    ("bass",   r"\bbass\b|reese|\bsub\b"),
    ("fx",     r"\bfx\b|\briser|\bdownlifter|\bimpact|\bsweep|\bwhoosh|foley|ambien|\bnoise\b|\bsfx\b|texture|drone"),
    ("synth",  r"\bsynth|\blead\b|\bpluck|\barp\b|\bpad\b|\bkeys?\b|\bpiano|\bguitar|\bbrass|\bstring|\bbell|\bchord"),
    ("drums",  r"\bdrums?\b|\bbreak(beat)?s?\b|\bgroove\b"),   # full drum loops/breaks
]

# one-shot drum types: in instrument-named folders these are hits, not loops
ONESHOT_DRUMS = {"kick", "snare", "hihat", "clap", "tom", "cymbal", "808", "perc"}
# non-pitched types: never attach a detected key (only a filename key, if any)
DRUM_NONTONAL = {"kick", "snare", "hihat", "clap", "tom", "cymbal", "perc", "drums"}
# non-pitched one-shots identified by path: nothing useful comes from decoding
# them (no key, no bpm, percussive) — skip librosa entirely for big speedups.
FAST_NO_AUDIO = {"kick", "snare", "hihat", "clap", "tom", "cymbal", "perc"}

# Sound-effect / speech / field-recording material. BPM and key are meaningless
# here, so we force instrument=fx and suppress bpm/key (and skip decoding).
# Matched against the whole path: explicit pack folders + generic keywords.
_SFX_PACKS = {
    "99sounds halloween sound effects", "aversion - horror sound effects free pack",
    "bbc sound effects", "ghosthack - advent calendar 2019 day 5 - horror sounds",
    "ghosthack - free metal screams", "gun sounds", "national library of medicine",
    "ocean swift - 20th century public domain speeches", "creepy_audio_recordings",
    "hl2", "system32exe", "raw data", "_random",
}
_SFX_RE = re.compile(
    r"sound ?effects?|\bsfx\b|foley|field.?recording|\bspeeche?s?\b|"
    r"\bscream|\bgun ?sound|horror sound|\bsiren\b|\binterview\b", re.I)


def is_sfx(path):
    parts = path.lower().split(os.sep)
    if any(p in _SFX_PACKS for p in parts):
        return True
    return bool(_SFX_RE.search(path))
_LOOP_RE   = re.compile(r"\bloops?\b|\bbreak(beat)?s?\b|groove", re.I)
_ONESHOT_RE= re.compile(r"\bone[ _-]?shots?\b|\bhits?\b|\bstabs?\b", re.I)
_BPM_RE    = re.compile(r"(?<!\d)(\d{2,3})\s?bpm\b", re.I)
# note stays uppercase (so "AV"/"Free" don't match); accidental lowercase 'b' only
# (avoids B-note vs b-flat clash); qualifier is case-insensitive ("Min"/"min"/"m").
_KEY_RE    = re.compile(r"(?<![A-Za-z])([A-G])[ _]?(#|b)?[ _]?((?i:maj(?:or)?|min(?:or)?|m))?\b")

_KEY_WORD_TONAL = re.compile(r"maj|min|chord|melod|key|scale", re.I)


def parse_path_hints(path):
    """Return dict of strong hints derived from folder + filename text."""
    text = path.lower()
    base = os.path.basename(path).lower()
    hints = {}

    inst = None
    for name, pat in _INSTRUMENT_PATTERNS:
        if re.search(pat, text, re.I):
            inst = name
            break
    if inst:
        hints["instrument"] = inst

    if _ONESHOT_RE.search(text):
        hints["sample_type"] = "oneshot"
    elif _LOOP_RE.search(text):
        hints["sample_type"] = "loop"

    m = _BPM_RE.search(base) or _BPM_RE.search(text)
    if m:
        b = int(m.group(1))
        if 40 <= b <= 300:
            hints["bpm"] = b

    # key only trusted from the filename, and only when it looks musical
    if _KEY_WORD_TONAL.search(base) or re.search(r"[A-G](#|b)?m?\b", os.path.basename(path)):
        km = _KEY_RE.search(os.path.basename(path))
        if km:
            note = km.group(1).upper()
            acc = km.group(2)
            if acc in ("#", "sharp"):
                note += "#"
            elif acc in ("b", "flat"):
                # normalize flats to sharps
                flat_to_sharp = {"Cb":"B","Db":"C#","Eb":"D#","Fb":"E","Gb":"F#","Ab":"G#","Bb":"A#"}
                note = flat_to_sharp.get(note + "b", note)
            qual = km.group(3) or ""
            minor = qual.lower().startswith("m") and not qual.lower().startswith("maj")
            hints["key"] = note + ("m" if minor else "")
    return hints


# ----------------------------------------------------------------------------
# Audio analysis
# ----------------------------------------------------------------------------
def _true_duration(path):
    """Cheap full duration without decoding the whole file."""
    import soundfile as sf
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def load_audio(path, duration=None):
    import librosa
    try:
        y, sr = librosa.load(path, sr=SR, mono=True, duration=duration)
    except Exception as e:
        if "not finite" not in str(e):
            raise
        # Librosa validates after read and raises if the buffer has NaN/inf (common
        # in old float-format WAVs from E-MU Emulator, Emax, Amiga trackers, etc.).
        # Bypass by reading with soundfile (no validation) then sanitize + resample.
        import soundfile as sf
        data, sr_raw = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if duration is not None:
            data = data[:int(duration * sr_raw)]
        y = librosa.resample(data, orig_sr=float(sr_raw), target_sr=SR)
        sr = SR
    else:
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, sr


def harmonic_ratio(h, p):
    eh = float(np.sum(h ** 2)); ep = float(np.sum(p ** 2))
    return eh / (eh + ep) if (eh + ep) else 0.0


def detect_bpm(y, sr):
    import librosa
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
    except AttributeError:
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    if tempo is None or len(tempo) == 0:
        return None
    return float(tempo[0])


def count_onsets(perc, sr):
    import librosa
    return int(len(librosa.onset.onset_detect(y=perc, sr=sr, units="frames", delta=0.07)))


def classify_loop(duration_s, n_onsets, bpm, hr):
    if duration_s < LOOP_MIN_SEC:
        return "oneshot"
    bar_aligned = False
    if bpm and bpm > 0:
        bars = (duration_s * bpm / 60.0) / 4.0
        nearest = round(bars)
        if nearest >= 1:
            bar_aligned = abs(bars - nearest) / nearest <= LOOP_BAR_TOLERANCE
    if (1.0 - hr) < 0.12 and not bar_aligned:
        return "oneshot"
    if bar_aligned or n_onsets >= 4:
        return "loop"
    if duration_s >= 2.0 and n_onsets >= 3:
        return "loop"
    return "oneshot"


def detect_key(y, sr, is_tonal):
    if not is_tonal:
        return None
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = np.mean(chroma, axis=1)
    if profile.sum() == 0:
        return None
    best_score, best_key = -np.inf, None
    for i in range(12):
        maj = np.corrcoef(np.roll(KRUMHANSL_MAJOR, i), profile)[0, 1]
        minr = np.corrcoef(np.roll(KRUMHANSL_MINOR, i), profile)[0, 1]
        if maj > best_score: best_score, best_key = maj, NOTE_NAMES[i]
        if minr > best_score: best_score, best_key = minr, NOTE_NAMES[i] + "m"
    return best_key


def classify_instrument_audio(y, sr, is_tonal, duration_s):
    import librosa
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    S = np.abs(librosa.stft(y)); freqs = librosa.fft_frequencies(sr=sr)
    low_ratio = float(S[freqs < 150].sum() / (S.sum() + 1e-9))
    short = duration_s < 1.5
    if is_tonal and not short:
        if low_ratio > 0.45 and centroid < 800: return "bass"
        return "synth" if centroid > 1500 else "tonal"
    if low_ratio > 0.45 and centroid < 600: return "kick"
    if zcr > 0.15 and centroid > 5000:
        return "cymbal" if duration_s > 0.6 else "hihat"
    if flatness > 0.2 and 1500 < centroid < 5000:
        return "clap" if zcr > 0.08 else "snare"
    if centroid < 1500 and low_ratio > 0.25: return "tom"
    return "tonal" if is_tonal else "perc"


# ----------------------------------------------------------------------------
# PANNs (AudioSet CNN14) instrument recognition — optional, --panns
# ----------------------------------------------------------------------------
_USE_PANNS = False
_PANNS_ONLY = False  # when True: skip path/audio fallbacks; only classify via PANNs
_EMBED = False      # also capture & store the 2048-d embedding (for similarity)
_PANNS = None       # lazily-loaded per worker process
_PANNS_LABELS = None

# AudioSet label substring -> our taxonomy, in priority order (specific first).
_PANNS_MAP = [
    ("snare",  ("snare",)),
    ("kick",   ("bass drum",)),
    ("hihat",  ("hi-hat",)),
    ("cymbal", ("cymbal", "crash")),
    ("drums",  ("drum kit", "drum machine", "drum roll", "drumming", "drums")),
    ("clap",   ("clapping", "hands", "finger snapping")),
    ("perc",   ("cowbell", "tambourine", "maraca", "wood block", "rattle",
                "percussion", "tabla", "conga", "bongo", "gong")),
    ("bass",   ("bass guitar", "double bass", "bass (instrument)")),
    ("synth",  ("synthesizer", "sawtooth", "electronic music")),
    ("vocal",  ("singing", "choir", "vocal", "chant", "rapping", "yodeling",
                "speech", "humming")),
    ("tonal",  ("piano", "keyboard", "organ", "guitar", "banjo", "mandolin",
                "ukulele", "violin", "cello", "string", "orchestra", "trumpet",
                "brass", "trombone", "saxophone", "horn", "flute", "clarinet",
                "bell", "chime", "vibraphone", "marimba", "xylophone",
                "glockenspiel", "harp", "accordion")),
    ("fx",     ("gunshot", "explosion", "glass", "shatter", "siren", "noise",
                "wind", "rain", "thunder", "engine", "vehicle", "whoosh")),
]


def _map_scores(scores, labels, topk=8):
    """Map a 527-d AudioSet score vector to (instrument, confidence) via
    _PANNS_MAP. The first mappable label in top-k confidence order wins."""
    order = np.argsort(scores)[::-1][:topk]
    for i in order:
        lab = labels[i].lower()
        for cat, keys in _PANNS_MAP:
            if any(k in lab for k in keys):
                return cat, float(scores[i])
    return None, None


def _get_panns():
    global _PANNS, _PANNS_LABELS
    if _PANNS is None:
        import torch
        torch.set_num_threads(1)            # avoid thread oversubscription across workers
        from panns_inference import AudioTagging, labels
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _PANNS = AudioTagging(checkpoint_path=None, device=device)
        _PANNS_LABELS = labels
    return _PANNS, _PANNS_LABELS


def _panns_forward(y, sr, topk=8):
    """One CNN14 forward pass.
    Returns (instrument_or_None, confidence_or_None, embedding_or_None).
    instrument = top mapped AudioSet prediction; confidence = softmax score [0-1]."""
    try:
        at, labels = _get_panns()
        import librosa
        y32 = librosa.resample(y, orig_sr=sr, target_sr=32000).astype(np.float32)
        if 0 < y32.shape[0] < 32000:
            # CNN14 needs ~1s. Tile (not zero-pad) so short one-shots carry real
            # signal throughout — silence-padding collapses them to one vector and
            # destroys similarity discrimination between e.g. kicks and snares.
            y32 = np.tile(y32, int(np.ceil(32000 / y32.shape[0])))[:32000]
        clipwise, emb = at.inference(y32[None, :])
        inst, conf = _map_scores(clipwise[0], labels, topk)
        return inst, conf, np.asarray(emb[0], dtype=np.float32)
    except Exception:
        return None, None, None


def classify_instrument_panns(y, sr):
    return _panns_forward(y, sr)[0]


def analyze_file(path):
    """Full pipeline for one file. Never raises; returns Analysis(status=...)."""
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
    if not _EMBED and not _PANNS_ONLY:
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
        want_class = (_USE_PANNS and not sfx and duration_s >= PANNS_MIN_DURATION)
        inst_panns, panns_conf, emb_arr = None, None, None
        if _EMBED or want_class:
            inst_panns, panns_conf, emb_arr = _panns_forward(y, sr)
        emb_bytes = emb_arr.tobytes() if emb_arr is not None else None
        dur = round(duration_s, 3)
        path_instr = hints.get("instrument")   # always record what the path said

        if sfx and not _PANNS_ONLY:              # cheap class, but we decoded to embed
            a = Analysis(path, mtime, size, dur, hints.get("sample_type", "oneshot"),
                         None, None, "percussive", "fx", "path",
                         path_instrument=path_instr, panns_instrument=inst_panns,
                         panns_conf=panns_conf)
        elif path_drum and not _PANNS_ONLY:
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
            elif _PANNS_ONLY:
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
        instr = None if _PANNS_ONLY else hints.get("instrument", "fx")
        src   = "none" if _PANNS_ONLY else ("path" if hints else "none")
        return Analysis(
            path, mtime, size, 0.0,
            hints.get("sample_type", "oneshot"), hints.get("bpm"), hints.get("key"),
            "percussive", instr, src,
            status="error", error=f"{type(e).__name__}: {e}",
            path_instrument=hints.get("instrument"))


# ----------------------------------------------------------------------------
# Tag writing  (in-file)  — only called for status == ok
# ----------------------------------------------------------------------------
def write_tags(a: Analysis):
    ext = os.path.splitext(a.path)[1].lower()
    if ext == ".flac":
        from mutagen.flac import FLAC
        au = FLAC(a.path)
        au["GENRE"] = a.instrument
        au["INSTRUMENT"] = a.instrument
        au["SAMPLE_TYPE"] = a.sample_type
        au["TONAL"] = a.tonal
        if a.duration_s: au["DURATION_S"] = str(a.duration_s)
        if a.bpm: au["BPM"] = str(a.bpm)
        if a.key: au["KEY"] = a.key
        au.save()
        return True

    from mutagen.id3 import TBPM, TKEY, TCON, TXXX
    if ext == ".mp3":
        from mutagen.mp3 import MP3 as Box
    elif ext == ".wav":
        from mutagen.wave import WAVE as Box
    elif ext in (".aif", ".aiff"):
        from mutagen.aiff import AIFF as Box
    else:
        return False
    au = Box(a.path)
    if au.tags is None:
        au.add_tags()
    t = au.tags
    t.setall("TCON", [TCON(encoding=3, text=[a.instrument])])
    pairs = [("SAMPLE_TYPE", a.sample_type), ("INSTRUMENT", a.instrument), ("TONAL", a.tonal)]
    if a.duration_s:
        pairs.append(("DURATION_S", str(a.duration_s)))
    for desc, val in pairs:
        t.setall(f"TXXX:{desc}", [TXXX(encoding=3, desc=desc, text=[val])])
    if a.bpm: t.setall("TBPM", [TBPM(encoding=3, text=[str(a.bpm)])])
    if a.key: t.setall("TKEY", [TKEY(encoding=3, text=[a.key])])
    au.save()
    return True


# ----------------------------------------------------------------------------
# SQLite index / checkpoint
# ----------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  path TEXT PRIMARY KEY, mtime REAL, size INTEGER, duration_s REAL,
  sample_type TEXT, instrument TEXT, tonal TEXT, bpm INTEGER, key TEXT,
  source TEXT, status TEXT, error TEXT, tagged INTEGER DEFAULT 0, ts REAL,
  path_instrument TEXT, panns_instrument TEXT, panns_conf REAL, audio_instrument TEXT,
  panns_label TEXT, panns_label_conf REAL, panns_topk TEXT
);
CREATE INDEX IF NOT EXISTS idx_instr ON samples(instrument);
CREATE INDEX IF NOT EXISTS idx_type  ON samples(sample_type);
CREATE TABLE IF NOT EXISTS embeddings (
  path TEXT PRIMARY KEY, dim INTEGER, vec BLOB
);
"""

MIGRATIONS = [
    "ALTER TABLE samples ADD COLUMN path_instrument TEXT",
    "ALTER TABLE samples ADD COLUMN panns_instrument TEXT",
    "ALTER TABLE samples ADD COLUMN panns_conf REAL",
    "ALTER TABLE samples ADD COLUMN audio_instrument TEXT",
    "ALTER TABLE samples ADD COLUMN panns_label TEXT",        # raw top-1 AudioSet label
    "ALTER TABLE samples ADD COLUMN panns_label_conf REAL",   # its sigmoid score
    "ALTER TABLE samples ADD COLUMN panns_topk TEXT",         # JSON [[label,score],...] top-5
]

def db_connect(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # apply any missing columns to existing databases
    existing = {r[1] for r in con.execute("PRAGMA table_info(samples)")}
    for sql in MIGRATIONS:
        col = sql.split()[-2]  # column name is second-to-last token
        if col not in existing:
            con.execute(sql)
    con.execute("PRAGMA journal_mode=WAL")
    return con

def db_known_set(con):
    """All indexed paths → (mtime, size, status)."""
    return {r[0]: (r[1], r[2], r[3]) for r in
            con.execute("SELECT path, mtime, size, status FROM samples")}

def db_upsert(con, a: Analysis, tagged: bool):
    con.execute("""INSERT INTO samples
        (path,mtime,size,duration_s,sample_type,instrument,tonal,bpm,key,source,status,error,tagged,ts,
         path_instrument,panns_instrument,panns_conf,audio_instrument)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
          mtime=excluded.mtime,size=excluded.size,duration_s=excluded.duration_s,
          sample_type=excluded.sample_type,instrument=excluded.instrument,tonal=excluded.tonal,
          bpm=excluded.bpm,key=excluded.key,source=excluded.source,status=excluded.status,
          error=excluded.error,tagged=excluded.tagged,ts=excluded.ts,
          path_instrument=excluded.path_instrument,panns_instrument=excluded.panns_instrument,
          panns_conf=excluded.panns_conf,audio_instrument=excluded.audio_instrument""",
        (a.path, a.mtime, a.size, a.duration_s, a.sample_type, a.instrument, a.tonal,
         a.bpm, a.key, a.source, a.status, a.error, int(tagged), time.time(),
         a.path_instrument, a.panns_instrument, a.panns_conf, a.audio_instrument))
    if a.emb is not None:
        con.execute("INSERT INTO embeddings(path,dim,vec) VALUES (?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
                    (a.path, len(a.emb) // 4, a.emb))


def db_discover_upsert(con, path, mtime, size, path_instr):
    """Insert new file or refresh mtime/size. Never overwrites existing labels."""
    con.execute("""
        INSERT INTO samples (path, mtime, size, status, ts, path_instrument)
        VALUES (?, ?, ?, 'new', ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          mtime=excluded.mtime, size=excluded.size, ts=excluded.ts,
          path_instrument=COALESCE(samples.path_instrument, excluded.path_instrument),
          status=CASE WHEN samples.status='missing' THEN 'new' ELSE samples.status END
    """, (path, mtime, size, time.time(), path_instr))


def db_label_update(con, path, result, redo_set):
    """Write classifier results into their dedicated columns. Never touches
    instrument/source (those are reserved for human + trained classifier)."""
    ts = time.time()
    if "path_instrument" in result and ("path" in redo_set or result.get("_missing")):
        con.execute("UPDATE samples SET path_instrument=?, ts=? WHERE path=?",
                    (result["path_instrument"], ts, path))
    if "audio_instrument" in result:
        if "audio" in redo_set:
            con.execute("UPDATE samples SET audio_instrument=?, ts=? WHERE path=?",
                        (result["audio_instrument"], ts, path))
        else:
            con.execute("UPDATE samples SET audio_instrument=COALESCE(audio_instrument,?), ts=? WHERE path=?",
                        (result["audio_instrument"], ts, path))
    if "panns_instrument" in result:
        if "panns" in redo_set:
            con.execute("UPDATE samples SET panns_instrument=?, panns_conf=?, ts=? WHERE path=?",
                        (result["panns_instrument"], result.get("panns_conf"), ts, path))
        else:
            con.execute("""UPDATE samples SET
                panns_instrument=COALESCE(panns_instrument,?),
                panns_conf=COALESCE(panns_conf,?), ts=? WHERE path=?""",
                        (result["panns_instrument"], result.get("panns_conf"), ts, path))
        if result.get("emb"):
            con.execute("INSERT INTO embeddings(path,dim,vec) VALUES (?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
                        (path, len(result["emb"]) // 4, result["emb"]))
    # update status/error from decode attempt
    if "status" in result:
        con.execute("UPDATE samples SET status=?, error=? WHERE path=?",
                    (result["status"], result.get("error", ""), path))


# ----------------------------------------------------------------------------
# Workers — discover stage
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Workers — label stage
# ----------------------------------------------------------------------------
_DO_AUDIO = False
_DO_PANNS = False


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
        y, sr = load_audio(path, duration=ANALYZE_SECONDS)
        if duration_s is None:
            import librosa
            duration_s = float(librosa.get_duration(y=y, sr=sr))
        if y.size == 0:
            raise ValueError("empty audio")

        if _DO_PANNS and duration_s >= PANNS_MIN_DURATION:
            inst, conf, emb = _panns_forward(y, sr)
            result["panns_instrument"] = inst
            result["panns_conf"] = conf
            result["emb"] = emb.tobytes() if emb is not None else None

        if _DO_AUDIO:
            import librosa
            h, p = librosa.effects.hpss(y)
            hr = harmonic_ratio(h, p)
            is_tonal = hr >= HARMONIC_RATIO_TONAL
            result["audio_instrument"] = classify_instrument_audio(
                y, sr, is_tonal, duration_s)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
    return result


# ----------------------------------------------------------------------------
# Legacy combined worker (kept for write_tags support)
# ----------------------------------------------------------------------------
_WRITE_TAGS = True

def _init_worker(write_tags_flag, use_panns, embed, panns_only=False):
    global _WRITE_TAGS, _USE_PANNS, _EMBED, _PANNS_ONLY
    _WRITE_TAGS = write_tags_flag
    _USE_PANNS = use_panns
    _EMBED = embed
    _PANNS_ONLY = panns_only

def process_one(path):
    a = analyze_file(path)
    tagged = False
    if _WRITE_TAGS and a.status == "ok":
        try:
            tagged = write_tags(a)
        except Exception as e:
            a.error = f"tagwrite: {type(e).__name__}: {e}"
    return a, tagged


# ----------------------------------------------------------------------------
# File gathering
# ----------------------------------------------------------------------------
def gather(root):
    if os.path.isfile(root):
        yield root; return
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                yield os.path.join(dp, f)


# ----------------------------------------------------------------------------
# relabel-panns stage — recompute PANNs labels from stored embeddings only
# ----------------------------------------------------------------------------
def stage_relabel_panns(con, args):
    """Reconstruct the raw PANNs label (panns_label/panns_label_conf) for every
    stored embedding WITHOUT decoding any audio.

    CNN14's clipwise output is exactly sigmoid(fc_audioset(embedding)), and the
    2048-d embedding is already in the DB, so the whole library is relabeled with
    a single matrix multiply — no filesystem reads, no librosa, seconds not hours.

    Stores the model's raw output verbatim (527-way AudioSet vocabulary, e.g.
    "Bass drum", "Music", "Water") — no taxonomy mapping, no thresholding:
      panns_label / panns_label_conf  top-1 class + sigmoid score
      panns_topk                      JSON [[label, score], ...] of the top 5
    Top-1 is often the generic "Music" tag, so the top-5 is what carries the
    instrument signal. Collapsing to the instrument taxonomy is a later,
    separately-runnable step.
    """
    import json
    import torch
    from panns_inference import AudioTagging, labels as panns_labels
    TOPK = 5

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    print(f"Loading CNN14 classifier head (fc_audioset) on {device} ...", flush=True)
    at = AudioTagging(checkpoint_path=None, device=device)
    # On CUDA panns_inference wraps the net in DataParallel; the real module
    # (and fc_audioset) then lives under .module.
    model = getattr(at.model, "module", at.model)
    head = model.fc_audioset.eval()

    # Stream embeddings on a separate read-only connection so the cursor isn't
    # disturbed by our batched UPDATEs on `con`.
    rcon = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    total = rcon.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    if args.limit:
        total = min(total, args.limit)
    con.execute("PRAGMA busy_timeout=30000")
    print(f"{total} embeddings -> reconstructing PANNs labels (no audio decode)", flush=True)

    BATCH = args.batch
    n = 0
    pend_paths, pend_vecs = [], []
    t0 = time.time()

    def flush(paths, vecs):
        arr = np.frombuffer(b"".join(vecs), dtype=np.float32).reshape(len(vecs), -1)
        with torch.no_grad():
            clip = torch.sigmoid(head(torch.from_numpy(arr).to(device))).cpu().numpy()
        # top-K class indices per row (unsorted from argpartition, then sorted desc)
        part = np.argpartition(-clip, TOPK, axis=1)[:, :TOPK]
        ts = time.time()
        rows = []
        for r, p in enumerate(paths):
            idx = part[r][np.argsort(-clip[r, part[r]])]
            pairs = [[panns_labels[i], round(float(clip[r, i]), 4)] for i in idx]
            rows.append((pairs[0][0], pairs[0][1], json.dumps(pairs), ts, p))
        if not args.dry_run:
            con.executemany("UPDATE samples SET panns_label=?, panns_label_conf=?, "
                            "panns_topk=?, ts=? WHERE path=?", rows)
            con.commit()

    cur = rcon.execute("SELECT path, vec FROM embeddings")
    for p, v in cur:
        if v is None or len(v) != DIM * 4:
            continue
        pend_paths.append(p)
        pend_vecs.append(v)
        n += 1
        if len(pend_paths) >= BATCH:
            flush(pend_paths, pend_vecs)
            pend_paths, pend_vecs = [], []
            rate = n / (time.time() - t0)
            print(f"  {n}/{total}  {rate:6.0f}/s", flush=True)
        if args.limit and n >= args.limit:
            break
    if pend_paths:
        flush(pend_paths, pend_vecs)
    rcon.close()
    print(f"\nRelabel done: {n} files in {time.time()-t0:.1f}s "
          f"({'dry-run, nothing written' if args.dry_run else 'panns_label updated'})",
          flush=True)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Two-stage audio sample classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # find new files and extract path hints (no audio decode)
  sample_tagger.py /path --stage discover

  # force fresh filesystem walk (after adding files)
  sample_tagger.py /path --stage discover --no-cache

  # compute PANNs + audio labels for unlabeled files
  sample_tagger.py /path --stage label --classifiers panns,audio -j 4

  # force-refresh PANNs labels after a model update
  sample_tagger.py /path --stage label --classifiers panns --redo panns
""")
    ap.add_argument("path", help="root of the sample library")
    ap.add_argument("--db",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.db"),
                    help="SQLite database path (default: samples.db next to this script)")
    ap.add_argument("-j", "--workers", type=int, default=max(1, cpu_count() - 1),
                    help="parallel worker processes; PANNs uses ~1 GB RAM per worker")
    ap.add_argument("--stage", choices=["discover", "label", "relabel-panns"], required=True,
                    help="discover: walk filesystem, register new/missing files, extract path hints. "
                         "label: compute classifier columns for already-indexed files. "
                         "relabel-panns: recompute panns_instrument from stored embeddings only "
                         "(no audio decode) — seconds, not hours.")
    ap.add_argument("--classifiers", default="panns",
                    help="comma-separated classifiers for label stage: path, audio, panns "
                         "(default: panns)")
    ap.add_argument("--redo", default="",
                    help="comma-separated classifiers to force-overwrite (e.g. panns after a model "
                         "update); \'all\' overwrites every requested classifier")
    ap.add_argument("--no-cache", action="store_true",
                    help="(discover) force a fresh filesystem walk instead of the cached file list")
    ap.add_argument("--trust-db", action="store_true",
                    help="(discover) skip mtime/size checks on known files; faster on network mounts")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (0 = no limit); useful for test runs")
    ap.add_argument("--gpu", action="store_true",
                    help="(relabel-panns) run the classifier head on CUDA if available "
                         "(default CPU — the matmul is trivial and avoids GPU contention)")
    ap.add_argument("--batch", type=int, default=4096,
                    help="(relabel-panns) embeddings per matmul/commit batch")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="analyze but write nothing")
    args = ap.parse_args()

    con = None if (args.dry_run and args.stage != "relabel-panns") else db_connect(args.db)
    t0 = time.time()

    if args.stage == "relabel-panns":
        stage_relabel_panns(con, args)
        return

    # ------------------------------------------------------------------ discover
    if args.stage == "discover":
        filelist_cache = args.db + ".filelist"
        if not args.no_cache and os.path.exists(filelist_cache):
            print(f"Loading file list from cache ({filelist_cache}) ...", flush=True)
            with open(filelist_cache) as f:
                all_files = [l.rstrip("\n") for l in f if l.strip()]
        else:
            print(f"Walking {args.path} ...", flush=True)
            all_files = list(gather(args.path))
            if not args.dry_run:
                with open(filelist_cache, "w") as f:
                    f.write("\n".join(all_files) + "\n")

        known = db_known_set(con) if con else {}
        found_set = set(all_files)

        # mark files that disappeared from disk
        missing = [p for p, (_, _, st) in known.items()
                   if p not in found_set and st != "missing"]
        if missing and not args.dry_run:
            con.executemany("UPDATE samples SET status=\'missing\', ts=? WHERE path=?",
                            [(time.time(), p) for p in missing])
            con.commit()
            print(f"Marked {len(missing)} files as missing.", flush=True)

        # discover new / changed files
        if args.trust_db:
            todo = [p for p in all_files if p not in known]
        else:
            todo = []
            for p in all_files:
                if p not in known:
                    todo.append(p)
                else:
                    mtime, size, _ = known[p]
                    try:
                        st = os.stat(p)
                        if (round(st.st_mtime, 3), st.st_size) != (round(mtime, 3), size):
                            todo.append(p)
                    except OSError:
                        pass

        print(f"{len(all_files)} files on disk, {len(missing)} missing, "
              f"{len(todo)} new/changed to register.", flush=True)

        if args.limit:
            todo = todo[:args.limit]
        if not todo:
            print("Nothing to do."); return

        n, errors = 0, 0
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(discover_one, todo, chunksize=64):
                n += 1
                if r["ok"] and not args.dry_run:
                    db_discover_upsert(con, r["path"], r["mtime"], r["size"],
                                       r.get("path_instrument"))
                else:
                    errors += 1
                if con and n % 500 == 0:
                    con.commit()
                if n % 2000 == 0 or n == len(todo):
                    rate = n / (time.time() - t0)
                    print(f"  {n}/{len(todo)}  {rate:.0f}/s  err={errors}", flush=True)
        if con:
            con.commit()
        print(f"\nDiscovery done: {n} registered, {errors} errors, "
              f"{(time.time()-t0):.1f}s", flush=True)

    # ------------------------------------------------------------------ label
    elif args.stage == "label":
        classifiers = {c.strip().lower() for c in args.classifiers.split(",") if c.strip()}
        redo_set = ({c.strip().lower() for c in args.redo.split(",") if c.strip()}
                    if args.redo else set())
        if "all" in redo_set:
            redo_set = set(classifiers)

        do_audio = "audio" in classifiers
        do_panns = "panns" in classifiers

        # build todo: files missing at least one requested classifier result
        where_parts = []
        if "panns" in classifiers and "panns" not in redo_set:
            where_parts.append("panns_instrument IS NULL")
        if "audio" in classifiers and "audio" not in redo_set:
            where_parts.append("audio_instrument IS NULL")
        if "path" in classifiers and "path" not in redo_set:
            where_parts.append("path_instrument IS NULL")

        if where_parts or redo_set:
            where = ("(" + " OR ".join(where_parts) + ")" if where_parts else "1=1")
            todo = [r[0] for r in con.execute(
                f"SELECT path FROM samples WHERE status != \'missing\' AND ({where})")]
        else:
            todo = []

        print(f"Label stage  classifiers={sorted(classifiers)}  redo={sorted(redo_set)}", flush=True)
        print(f"{len(todo)} files to process on {args.workers} workers.", flush=True)

        if args.limit:
            todo = todo[:args.limit]
        if not todo:
            print("Nothing to do."); return

        n, errors = 0, 0
        with Pool(args.workers, initializer=_init_label_worker,
                  initargs=(do_audio, do_panns)) as pool:
            for result in pool.imap_unordered(label_one, todo, chunksize=8):
                n += 1
                if result.get("status") == "error":
                    errors += 1
                if not args.dry_run:
                    db_label_update(con, result["path"], result, redo_set)
                    if n % 200 == 0:
                        con.commit()
                if n % 200 == 0 or n == len(todo):
                    rate = n / (time.time() - t0)
                    eta = (len(todo) - n) / rate if rate else 0
                    print(f"  {n}/{len(todo)}  {rate:5.1f}/s  eta {eta/60:5.1f}m  err={errors}",
                          flush=True)
        if con:
            con.commit()
        print(f"\nLabeling done: {n} processed, {errors} errors, "
              f"{(time.time()-t0)/60:.1f}m", flush=True)
        if con:
            print(f"index: {args.db}", flush=True)


if __name__ == "__main__":
    main()
