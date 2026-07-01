import os
from .config import cfg

SR = 22050
DIM = 2048
AUDIO_EXTS = {".wav", ".mp3", ".aif", ".aiff", ".flac"}

ANALYZE_SECONDS = cfg.analyze_seconds
LOOP_MIN_SEC = cfg.loop_min_sec
LOOP_BAR_TOLERANCE = cfg.loop_bar_tolerance
HARMONIC_RATIO_TONAL = cfg.harmonic_ratio_tonal
BPM_MIN, BPM_MAX = cfg.bpm_min, cfg.bpm_max
PANNS_MIN_DURATION = cfg.panns_min_duration
