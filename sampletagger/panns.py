import numpy as np

_PANNS_WARNED = False

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


def get_panns():
    from . import workers
    if workers._PANNS is None:
        import torch
        torch.set_num_threads(1)
        from panns_inference import AudioTagging, labels
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        workers._PANNS = AudioTagging(checkpoint_path=None, device=device)
        workers._PANNS_LABELS = labels
    return workers._PANNS, workers._PANNS_LABELS

def _panns_forward(y, sr, topk=8):
    """One CNN14 forward pass.
    Returns (instrument_or_None, confidence_or_None, embedding_or_None).
    instrument = top mapped AudioSet prediction; confidence = softmax score [0-1]."""
    try:
        at, labels = get_panns()
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
    except Exception as e:
        # Per-file decode errors are expected and tolerated, but a systemic failure
        # (model load, bad call) would otherwise be invisible — warn once per process.
        global _PANNS_WARNED
        if not _PANNS_WARNED:
            import sys
            print(f"[panns] inference failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            _PANNS_WARNED = True
        return None, None, None


def classify_instrument_panns(y, sr):
    return _panns_forward(y, sr)[0]

def load_head(device):
    from panns_inference import AudioTagging, labels as panns_labels
    at = AudioTagging(checkpoint_path=None, device=device)
    model = getattr(at.model, 'module', at.model)
    return model.fc_audioset.eval(), panns_labels
