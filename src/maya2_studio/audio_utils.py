"""Pure-NumPy / SoundFile audio utilities and acoustic voice sculpting."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from maya2_studio.catalog import SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class ReferenceClipStats:
    """Summary of an uploaded or recorded reference audio clip."""

    path: Path
    duration_s: float
    sample_rate: int
    peak: float
    warning: str | None = None


def read_audio_mono(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Read any SoundFile-compatible audio clip as float32 mono at `target_sr`."""
    data, sr = sf.read(str(path), dtype="float32")
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if len(arr) > 0:
        arr = arr - float(np.mean(arr))
    if int(sr) != int(target_sr) and len(arr) > 1:
        duration = len(arr) / float(sr)
        new_len = max(1, int(round(duration * target_sr)))
        old_x = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        arr = np.interp(new_x, old_x, arr).astype(np.float32)
        sr = target_sr
    return arr, int(sr)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    """Write float32 audio normalized safely inside [-1, 1] as 16-bit PCM WAV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if len(arr) else 0.0
    if peak > 0.99:
        arr = arr * (0.98 / peak)
    sf.write(str(out), arr, int(sr), subtype="PCM_16")
    return out


def pcm16_bytes_to_wav(pcm_bytes: bytes, out_path: str | Path, sr: int = SAMPLE_RATE) -> Path:
    """Wrap raw signed 16-bit little-endian mono PCM bytes into a standard WAV file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    sf.write(str(out), samples, int(sr), subtype="PCM_16")
    return out


def wav_bytes_to_file(wav_bytes: bytes, out_path: str | Path) -> Path:
    """Persist WAV bytes directly (or re-encode if needed) to `out_path`."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if wav_bytes[:4] == b"RIFF":
        out.write_bytes(wav_bytes)
        return out
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return write_wav(out, data, sr)


def audio_duration_seconds(path: str | Path) -> float:
    """Return audio duration in seconds."""
    info = sf.info(str(path))
    return float(info.duration)


def prepare_reference_clip(
    source_path: str | Path,
    dest_dir: str | Path | None = None,
    max_duration_s: float = 15.0,
) -> ReferenceClipStats:
    """Validate and normalize a reference clip to 24 kHz mono WAV.

    Maya 2 requires at least 0.1 s (2,400 samples at 24 kHz) and works best with
    3-10 s of clean single-speaker speech.
    """
    arr, sr = read_audio_mono(source_path, target_sr=SAMPLE_RATE)
    if len(arr) < 2400:
        raise ValueError("Reference clip is too short (< 0.1 s). Provide 3-10 s of clean speech.")

    warning = None
    dur = len(arr) / float(sr)
    if dur < 2.5:
        warning = f"Reference clip is short ({dur:.1f}s); 3-10s gives stronger clone similarity."
    elif dur > max_duration_s:
        arr = arr[: int(max_duration_s * sr)]
        dur = len(arr) / float(sr)
        warning = f"Trimmed reference clip to {dur:.1f}s (3-10s is ideal for Maya 2)."

    peak = float(np.max(np.abs(arr))) if len(arr) else 0.0
    if peak < 1e-4:
        raise ValueError("Reference clip appears silent. Check your microphone or file.")
    if peak < 0.15:
        arr = arr * (0.7 / peak)
        peak = float(np.max(np.abs(arr)))

    if dest_dir is None:
        fd, tmp = tempfile.mkstemp(prefix="maya2_ref_", suffix=".wav")
        os.close(fd)
        out_path = Path(tmp)
    else:
        out_path = Path(dest_dir) / f"ref_{os.urandom(6).hex()}.wav"
    write_wav(out_path, arr, sr)
    return ReferenceClipStats(
        path=out_path,
        duration_s=round(dur, 2),
        sample_rate=sr,
        peak=round(peak, 3),
        warning=warning,
    )


def _stft(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Compute real STFT using Hanning windowing."""
    win = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    padded = np.pad(x, (pad, pad), mode="reflect")
    n_frames = max(1, 1 + (len(padded) - n_fft) // hop)
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:n_frames]
    return np.fft.rfft(frames * win[None, :], axis=1)


def _istft(spec: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Reconstruct real signal via overlap-add inverse STFT."""
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = spec.shape[0]
    out_len = (n_frames - 1) * hop + n_fft
    y = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    frames = np.fft.irfft(spec, n=n_fft, axis=1).astype(np.float32)
    win_sq = win * win
    for i in range(n_frames):
        start = i * hop
        y[start : start + n_fft] += frames[i] * win
        wsum[start : start + n_fft] += win_sq
    pad = n_fft // 2
    mask = wsum > 1e-6
    y[mask] /= wsum[mask]
    return y[pad:-pad] if len(y) > 2 * pad else y


def _phase_vocoder_stretch(x: np.ndarray, rate: float, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Time-stretch `x` by `rate` (>1 faster/shorter, <1 slower/longer) without changing pitch."""
    if abs(rate - 1.0) < 1e-3 or len(x) < n_fft:
        return x
    spec = _stft(x, n_fft=n_fft, hop=hop)
    n_frames, n_bins = spec.shape
    time_steps = np.arange(0.0, max(1.0, n_frames - 1.0), rate, dtype=np.float64)
    if len(time_steps) < 2:
        return x

    omega = 2.0 * np.pi * hop * np.arange(n_bins, dtype=np.float64) / n_fft
    phase_acc = np.angle(spec[0]).astype(np.float64)
    out_spec = np.zeros((len(time_steps), n_bins), dtype=np.complex64)

    for idx, t in enumerate(time_steps):
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, n_frames - 1)
        frac = t - i0
        mag = (1.0 - frac) * np.abs(spec[i0]) + frac * np.abs(spec[i1])
        out_spec[idx] = mag * np.exp(1j * phase_acc)

        if i1 > i0:
            dp = np.angle(spec[i1]) - np.angle(spec[i0]) - omega
            dp = dp - 2.0 * np.pi * np.round(dp / (2.0 * np.pi))
            phase_acc += omega + dp
        else:
            phase_acc += omega

    return _istft(out_spec, n_fft=n_fft, hop=hop)


def _apply_spectral_tilt(x: np.ndarray, sr: int, tilt: float) -> np.ndarray:
    """Apply smooth frequency-domain warmth (-1) vs brightness (+1) spectral tilt."""
    if abs(tilt) < 1e-3 or len(x) < 32:
        return x
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    pivot_hz = 1200.0
    # Smooth log-frequency curve bounded to +-4.5 dB.
    norm = np.tanh(np.log2(np.maximum(freqs, 40.0) / pivot_hz) / 2.0)
    db_gain = tilt * 4.5 * norm
    gain = np.power(10.0, db_gain / 20.0).astype(np.float32)
    spec = np.fft.rfft(x) * gain
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def _rms_normalize(x: np.ndarray, target_rms: float = 0.09) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    if rms < 1e-6:
        return x
    y = x * (target_rms / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.95:
        y = y * (0.95 / peak)
    return y.astype(np.float32)


def sculpt_voice_reference(
    primary_audio_path: str | Path,
    *,
    pitch_semitones: float = 0.0,
    tempo_rate: float = 1.0,
    warmth_brightness: float = 0.0,
    secondary_audio_path: str | Path | None = None,
    secondary_weight: float = 0.0,
    out_path: str | Path | None = None,
) -> tuple[Path, float]:
    """Create a sculpted Maya 2 reference clip using DSP pitch, tempo, tone, and optional voice blending.

    - Pitch shift uses phase-vocoder time-stretching + linear resampling so tempo stays intact.
    - Secondary voice blending appends a loudness-matched slice of a second voice clip with a smooth
      crossfade so Maya 2's speaker encoder / ICL prompt blends vocal identity cues.
    """
    y, sr = read_audio_mono(primary_audio_path, target_sr=SAMPLE_RATE)
    if len(y) < 2400:
        raise ValueError("Primary seed clip is too short (< 0.1 s).")

    # 1. Pitch shift (time-preserving) + tempo stretch combined in one vocoder pass where possible.
    pitch_factor = float(2.0 ** (pitch_semitones / 12.0))
    # To raise pitch by pitch_factor without shortening duration:
    # first stretch duration by pitch_factor (vocoder rate = 1/pitch_factor), then resample by pitch_factor.
    vocoder_rate = float(tempo_rate) / pitch_factor
    if abs(vocoder_rate - 1.0) > 1e-3:
        y = _phase_vocoder_stretch(y, rate=vocoder_rate)

    if abs(pitch_factor - 1.0) > 1e-3 and len(y) > 4:
        new_len = max(2400, int(round(len(y) / pitch_factor)))
        old_x = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        y = np.interp(new_x, old_x, y).astype(np.float32)

    # 2. Spectral warmth / brightness tilt.
    y = _apply_spectral_tilt(y, sr, float(warmth_brightness))
    y = _rms_normalize(y)

    # 3. Optional secondary voice conditioning blend (concatenated with smooth crossfade so speech remains intelligible).
    if secondary_audio_path and secondary_weight > 0.02:
        y2, _ = read_audio_mono(secondary_audio_path, target_sr=SAMPLE_RATE)
        if len(y2) >= 2400:
            y2 = _rms_normalize(y2)
            # Proportionally take primary + secondary segments capped at ~9 s total.
            max_total = int(9.0 * sr)
            w2 = min(max(float(secondary_weight), 0.05), 0.75)
            primary_keep = min(len(y), int(max_total * (1.0 - w2)))
            secondary_keep = min(len(y2), int(max_total * w2))
            seg1 = y[: max(2400, primary_keep)]
            seg2 = y2[: max(2400, secondary_keep)]
            fade = min(int(0.04 * sr), len(seg1) // 4, len(seg2) // 4)
            gap = np.zeros(int(0.12 * sr), dtype=np.float32)
            if fade > 16:
                seg1 = seg1.copy()
                seg2 = seg2.copy()
                seg1[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
                seg2[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            y = np.concatenate([seg1, gap, seg2]).astype(np.float32)

    if out_path is None:
        fd, tmp = tempfile.mkstemp(prefix="maya2_sculpt_", suffix=".wav")
        os.close(fd)
        dest = Path(tmp)
    else:
        dest = Path(out_path)
    write_wav(dest, y, sr)
    return dest, round(len(y) / float(sr), 2)

