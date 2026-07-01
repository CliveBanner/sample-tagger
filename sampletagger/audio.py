import numpy as np
from .constants import SR, LOOP_MIN_SEC, LOOP_BAR_TOLERANCE, HARMONIC_RATIO_TONAL, BPM_MIN, BPM_MAX

KRUMHANSL_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KRUMHANSL_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

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
