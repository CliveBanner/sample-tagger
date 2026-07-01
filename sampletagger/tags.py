import os
from .analyze import Analysis

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
