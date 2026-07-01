import os
import json
from dataclasses import dataclass, fields, asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Config:
    library_path: str = "/home/phlp/pcloud/DAW/Samples"
    workers: int = 5
    trust_db: bool = True
    no_cache: bool = False
    label_path: bool = False
    label_audio: bool = False
    label_panns: bool = True
    gpu_python: str = ""
    redo: str = ""
    limit: int = 0
    analyze_seconds: float = 30.0
    loop_min_sec: float = 0.8
    loop_bar_tolerance: float = 0.12
    harmonic_ratio_tonal: float = 0.35
    bpm_min: int = 60
    bpm_max: int = 200
    panns_min_duration: float = 1.0
    proj_method: str = "auto"
    proj_n_neighbors: int = 25
    proj_min_dist: float = 0.12

def load_config(path=None):
    cfg = Config()
    try:
        with open(path or os.path.join(ROOT, "config.json")) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return cfg
    for fld in fields(cfg):
        if fld.name in data:
            val = data[fld.name]
            # Handle types properly (bool parsing from str can be tricky, but from json it's fine)
            setattr(cfg, fld.name, fld.type(val))
    return cfg

cfg = load_config()
