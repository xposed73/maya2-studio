"""Tests for NumPy audio utilities and the acoustic voice sculptor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from maya2_studio.audio_utils import (
    audio_duration_seconds,
    pcm16_bytes_to_wav,
    prepare_reference_clip,
    read_audio_mono,
    sculpt_voice_reference,
    write_wav,
)


def _sine_clip(path: Path, freq: float = 220.0, duration_s: float = 3.0, sr: int = 24000) -> Path:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    wave = 0.25 * np.sin(2.0 * np.pi * freq * t) + 0.08 * np.sin(2.0 * np.pi * freq * 2.0 * t)
    return write_wav(path, wave.astype(np.float32), sr)


def test_prepare_reference_clip_resamples_and_validates(tmp_path: Path) -> None:
    src = _sine_clip(tmp_path / "48k.wav", duration_s=3.5, sr=48000)
    stats = prepare_reference_clip(src, dest_dir=tmp_path)
    try:
        assert stats.sample_rate == 24000
        assert 3.4 <= stats.duration_s <= 3.6
        assert stats.warning is None
    finally:
        stats.path.unlink(missing_ok=True)

    too_short = _sine_clip(tmp_path / "short.wav", duration_s=0.05, sr=24000)
    with pytest.raises(ValueError, match="too short"):
        prepare_reference_clip(too_short, dest_dir=tmp_path)

    silent = write_wav(tmp_path / "silent.wav", np.zeros(24000 * 3, dtype=np.float32), 24000)
    with pytest.raises(ValueError, match="silent"):
        prepare_reference_clip(silent, dest_dir=tmp_path)


def test_pcm16_bytes_to_wav(tmp_path: Path) -> None:
    samples = (np.sin(np.linspace(0, 20, 24000)) * 12000).astype("<i2")
    out = pcm16_bytes_to_wav(samples.tobytes(), tmp_path / "pcm.wav", sr=24000)
    arr, sr = read_audio_mono(out, target_sr=24000)
    assert sr == 24000
    assert len(arr) == 24000


def test_sculpt_voice_reference_pitch_tempo_warmth_and_blend(tmp_path: Path) -> None:
    primary = _sine_clip(tmp_path / "p.wav", freq=200.0, duration_s=4.0, sr=24000)
    secondary = _sine_clip(tmp_path / "s.wav", freq=320.0, duration_s=4.0, sr=24000)

    out_pitch, dur_pitch = sculpt_voice_reference(
        primary,
        pitch_semitones=2.0,
        tempo_rate=1.0,
        warmth_brightness=-0.5,
        out_path=tmp_path / "sculpt_pitch.wav",
    )
    assert out_pitch.exists()
    # Time-preserving pitch shift should stay close to original 4.0s duration.
    assert 3.5 <= dur_pitch <= 4.5

    out_tempo, dur_tempo = sculpt_voice_reference(
        primary,
        pitch_semitones=0.0,
        tempo_rate=1.15,
        warmth_brightness=0.5,
        out_path=tmp_path / "sculpt_tempo.wav",
    )
    assert 3.1 <= dur_tempo <= 3.7
    assert audio_duration_seconds(out_tempo) == pytest.approx(dur_tempo, abs=0.05)

    out_blend, dur_blend = sculpt_voice_reference(
        primary,
        pitch_semitones=-1.5,
        tempo_rate=0.95,
        warmth_brightness=0.2,
        secondary_audio_path=secondary,
        secondary_weight=0.35,
        out_path=tmp_path / "sculpt_blend.wav",
    )
    assert out_blend.exists()
    assert dur_blend >= 3.0
    arr, _ = read_audio_mono(out_blend)
    assert np.isfinite(arr).all()
    assert float(np.max(np.abs(arr))) > 0.05

