"""Tests for Maya 2 backend dispatch and Gradio Blocks construction."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from maya2_studio.app import create_demo
from maya2_studio.audio_utils import write_wav
from maya2_studio.backends import (
    BackendConfig,
    BackendError,
    GenerationRequest,
    synthesize_maya2,
)
from maya2_studio.voice_store import VoiceStore


def _wav_bytes(duration_s: float = 1.5, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    samples = 0.2 * np.sin(2 * np.pi * 240 * np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32))
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_maya3_server_tts_and_clone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, dict | None]] = []
    wav_payload = _wav_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path == "/v1/voices":
            return httpx.Response(200, json={"ok": True})
        if request.method == "POST" and request.url.path == "/v1/audio/speech":
            return httpx.Response(200, content=wav_payload, headers={"X-Maya3-Voice": "Ahana"})
        if request.method == "DELETE" and request.url.path.startswith("/v1/voices/"):
            return httpx.Response(200, json={"deleted": True})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def mock_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", mock_client)

    cfg = BackendConfig(backend="maya3_server", server_url="http://localhost:8080")
    res = synthesize_maya2(GenerationRequest(text="नमस्ते", lang="hi", voice="Ahana"), cfg)
    try:
        assert Path(res.audio_path).exists()
        assert res.duration_s == pytest.approx(1.5, abs=0.05)
        assert calls[-1][1] == "/v1/audio/speech"
    finally:
        Path(res.audio_path).unlink(missing_ok=True)

    # Voice cloning path registers, synthesizes, and cleans up temporary voice id.
    ref_wav = write_wav(
        tmp_path / "ref.wav",
        0.2 * np.sin(np.linspace(0, 100, 24000 * 3, dtype=np.float32)),
        24000,
    )
    calls.clear()
    res_clone = synthesize_maya2(
        GenerationRequest(
            text="Hello world",
            lang="en",
            ref_audio=str(ref_wav),
            ref_text="Reference words",
            ref_lang="hi",
            clone_mode="icl_spk",
        ),
        cfg,
    )
    try:
        assert Path(res_clone.audio_path).exists()
        paths = [c[1] for c in calls]
        assert "/v1/voices" in paths
        assert "/v1/audio/speech" in paths
        assert any(p.startswith("/v1/voices/studio_tmp_") for p in paths)
    finally:
        Path(res_clone.audio_path).unlink(missing_ok=True)


def test_maya_cloud_pcm_and_clone_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    pcm = (np.sin(np.linspace(0, 60, 24000)) * 10000).astype("<i2").tobytes()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test_key"
        return httpx.Response(200, content=pcm)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real_client(*a, transport=transport, **kw))

    cfg = BackendConfig(backend="maya_cloud", maya_api_key="test_key", cloud_model="Maya 2 Native Emotional")
    res = synthesize_maya2(GenerationRequest(text="[excited] नमस्ते", lang="hi", voice="Ritvik"), cfg)
    try:
        assert Path(res.audio_path).exists()
        assert res.voice_label == "Ananya"
        assert "mapped" in res.notes
    finally:
        Path(res.audio_path).unlink(missing_ok=True)

    with pytest.raises(BackendError, match="does not accept reference-audio"):
        synthesize_maya2(
            GenerationRequest(text="Hello", lang="en", ref_audio="dummy.wav", ref_text="hi"),
            cfg,
        )


def test_create_demo_builds_all_tabs(tmp_path: Path) -> None:
    store = VoiceStore(root=tmp_path / "voices")
    demo, launch_kwargs = create_demo(store=store)
    assert demo is not None
    assert "theme" in launch_kwargs
    assert "css" in launch_kwargs
    assert str(store.root) in launch_kwargs["allowed_paths"]


def test_local_maya_mac_backend_tts_and_clone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys
    import types
    from maya2_studio import backends

    calls: list[dict] = []

    class FakeMayaTTS:
        device = "mlx"
        load_s = 1.2
        voices = ["plain", "Ahana", "Ritvik"]

        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.append({"init": kwargs})
            return cls()

        def generate(self, text: str, **kwargs):
            calls.append({"generate": text, "kw": kwargs})
            wav = 0.2 * np.sin(np.linspace(0, 80, 24000, dtype=np.float32))
            return wav, {"reason": "aeos_argmax", "rtf": 0.45, "ttfa_s": 0.12}

        def close(self) -> None:
            calls.append({"closed": True})

    fake_mod = types.ModuleType("maya_mac")
    fake_mod.MayaTTS = FakeMayaTTS
    monkeypatch.setitem(sys.modules, "maya_mac", fake_mod)
    backends.unload_local_engine()

    cfg = BackendConfig(backend="local", local_precision="mlx_8bit")
    try:
        preload_msg = backends.preload_local_engine(cfg)
        assert "maya_mac" in preload_msg
        assert calls[0]["init"]["backbone_bits"] == 8
        assert calls[0]["init"]["depth_bits"] == 8

        res = synthesize_maya2(
            GenerationRequest(text="नमस्ते", lang="hi", voice="Ahana", seed=7),
            cfg,
        )
        assert Path(res.audio_path).exists()
        assert "RTF=0.45" in res.notes
        Path(res.audio_path).unlink(missing_ok=True)

        ref_wav = write_wav(
            tmp_path / "ref.wav",
            0.2 * np.sin(np.linspace(0, 100, 24000 * 3, dtype=np.float32)),
            24000,
        )
        res_clone = synthesize_maya2(
            GenerationRequest(
                text="Hello",
                lang="en",
                voice="ignored_when_cloning",
                ref_audio=str(ref_wav),
                ref_text="Exact transcript",
                ref_lang="hi",
                clone_mode="icl_spk",
            ),
            cfg,
        )
        assert Path(res_clone.audio_path).exists()
        # Ensure voice is set to None when ref_audio is passed (required by maya-mac/maya3).
        assert calls[-1]["kw"]["voice"] is None
        assert calls[-1]["kw"]["ref_lang"] == "hi"
        Path(res_clone.audio_path).unlink(missing_ok=True)
    finally:
        backends.unload_local_engine()

