import numpy as np
from .constants import SR


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
