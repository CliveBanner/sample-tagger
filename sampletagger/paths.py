import os
import re

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
    ("drums",  r"\bdrums?\b|\bbreak(beat)?s?\b|\bgroove\b"),
]

_LOOP_RE   = re.compile(r"\bloops?\b|\bbreak(beat)?s?\b|groove", re.I)
_ONESHOT_RE= re.compile(r"\bone[ _-]?shots?\b|\bhits?\b|\bstabs?\b", re.I)
_BPM_RE    = re.compile(r"(?<!\d)(\d{2,3})\s?bpm\b", re.I)
_KEY_RE    = re.compile(r"(?<![A-Za-z])([A-G])[ _]?(#|b)?[ _]?((?i:maj(?:or)?|min(?:or)?|m))?\b")
_KEY_WORD_TONAL = re.compile(r"maj|min|chord|melod|key|scale", re.I)

def parse_path_hints(path):
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

    if _KEY_WORD_TONAL.search(base) or re.search(r"[A-G](#|b)?m?\b", os.path.basename(path)):
        km = _KEY_RE.search(os.path.basename(path))
        if km:
            note = km.group(1).upper()
            acc = km.group(2)
            if acc in ("#", "sharp"):
                note += "#"
            elif acc in ("b", "flat"):
                flat_to_sharp = {"Cb":"B","Db":"C#","Eb":"D#","Fb":"E","Gb":"F#","Ab":"G#","Bb":"A#"}
                note = flat_to_sharp.get(note + "b", note)
            qual = km.group(3) or ""
            minor = qual.lower().startswith("m") and not qual.lower().startswith("maj")
            hints["key"] = note + ("m" if minor else "")
    return hints
