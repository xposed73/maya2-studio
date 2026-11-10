"""Tests for Maya 2 catalog metadata and VoiceStore persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from maya2_studio.audio_utils import write_wav
from maya2_studio.catalog import (
    BUILTIN_VOICES,
    EMOTION_TAGS,
    LANGUAGES,
    compose_design_description,
    filter_builtin_voices,
    format_persona_summary,
)
from maya2_studio.voice_store import VoiceStore, sanitize_voice_id


def test_catalog_completeness() -> None:
    assert len(LANGUAGES) == 12
    assert len(BUILTIN_VOICES) == 20
    assert len(EMOTION_TAGS) == 10
    codes = {lang.code for lang in LANGUAGES}
    assert codes == {"as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}


def test_filter_builtin_voices() -> None:
    all_rows = filter_builtin_voices()
    assert len(all_rows) == 20

    tamil_females = filter_builtin_voices(gender="female", lang_code="ta")
    names = {r[0] for r in tamil_females}
    assert "Kayal" in names
    assert "Ritvik" not in names

    assert "plain" in format_persona_summary("plain")
    assert "Ahana" in format_persona_summary("Ahana")


def test_compose_design_description() -> None:
    desc = compose_design_description(
        gender="Female",
        age="30s",
        accent="indian",
        pitch="Normal",
        timbre="warm",
        pacing="conversational",
        tone="calm",
        intensity="medium",
        role="audiobook narrator",
    )
    assert "female" in desc.lower()
    assert "indian accent" in desc.lower()
    assert "warm timbre" in desc.lower()


def test_voice_store_crud(tmp_path: Path) -> None:
    store = VoiceStore(root=tmp_path / "voices")
    src = tmp_path / "sample.wav"
    tone = 0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, 3.0, 24000 * 3, dtype=np.float32))
    write_wav(src, tone, 24000)

    prof = store.save_profile(
        name="My Custom Voice!",
        source_type="cloned",
        ref_audio_path=src,
        ref_text="Hello world",
        ref_lang="en",
        mode="icl_spk",
        notes="test clip",
    )
    assert prof.name == sanitize_voice_id("My Custom Voice!")
    assert Path(prof.ref_audio_path).exists()

    fetched = store.get(prof.name.lower())
    assert fetched is not None
    assert fetched.ref_text == "Hello world"
    assert len(store.list_profiles()) == 1

    assert store.delete(prof.name) is True
    assert store.get(prof.name) is None
    assert store.list_profiles() == []

    with pytest.raises(ValueError):
        sanitize_voice_id("!!!")

