"""Unified inference backends for Maya 2 TTS, voice cloning, and voice design."""

from __future__ import annotations

import base64
import gc
import hashlib
import os
import random
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
from gradio_client import Client, handle_file

from maya2_studio.audio_utils import (
    audio_duration_seconds,
    pcm16_bytes_to_wav,
    prepare_reference_clip,
    wav_bytes_to_file,
    write_wav,
)
from maya2_studio.catalog import CLOUD_VOICES, HF_MODEL_REPO, SAMPLE_RATE

HF_STUDIO_SPACE: Final[str] = "maya-research/maya2-native-studio"
HF_STUDIO_URL: Final[str] = "https://maya-research-maya2-native-studio.hf.space"
HF_COMMUNITY_SPACE: Final[str] = "hugging-apps/maya2-native"
HF_MAYA1_SPACE: Final[str] = "maya-research/maya1"
MAYA_CLOUD_URL: Final[str] = "https://tts.mayaresearch.ai/v1/tts"

BACKEND_CHOICES: Final[list[tuple[str, str]]] = [
    ("Local Maya 2 Model (Apple Silicon MLX / MPS / CUDA)", "local"),
    ("Self-Hosted maya3.server (HTTP API)", "maya3_server"),
    ("Hugging Face Space (maya-research/maya2-native-studio)", "hf_studio"),
    ("Hugging Face Space (hugging-apps/maya2-native)", "hf_community"),
    ("Maya Cloud API (tts.mayaresearch.ai)", "maya_cloud"),
]

LOCAL_PRECISION_CHOICES: Final[list[tuple[str, str]]] = [
    ("MLX 8-bit (fast, ~4.5-7 GB RAM — recommended for 16 GB Mac)", "mlx_8bit"),
    ("MLX fp16 (exact, ~8.5 GB RAM)", "mlx_fp16"),
    ("MLX bf16 (exact)", "mlx_bf16"),
    ("MLX 4-bit (lowest RAM)", "mlx_4bit"),
    ("PyTorch MPS / CPU (maya-mac torch fallback)", "torch_mps"),
    ("Official maya3 CUDA engine (Linux + NVIDIA GPU)", "cuda_maya3"),
]


class BackendError(RuntimeError):
    """Actionable failure raised by a Maya inference backend."""


@dataclass(slots=True)
class BackendConfig:
    """Connection and model selection settings for Maya 2 inference."""

    backend: str = "local"
    local_precision: str = "mlx_8bit"
    local_models_dir: str = ""
    local_capacity: int = 2048
    hf_token: str = ""
    server_url: str = "http://127.0.0.1:8080"
    maya_api_key: str = ""
    cloud_model: str = "Maya 2 Native"
    local_model_id: str = HF_MODEL_REPO

    def resolved_hf_token(self) -> str | None:
        tok = (self.hf_token or os.environ.get("HF_TOKEN") or "").strip()
        return tok or None

    def resolved_maya_key(self) -> str | None:
        key = (self.maya_api_key or os.environ.get("MAYA_API_KEY") or "").strip()
        return key or None


@dataclass(slots=True)
class GenerationRequest:
    """Unified synthesis / cloning request across all Maya 2 backends."""

    text: str
    lang: str = "hi"
    voice: str = "Ahana"
    normalize: bool = True
    seed: int = 0
    randomize_seed: bool = False
    ref_audio: str | None = None
    ref_text: str = ""
    ref_lang: str = "hi"
    clone_mode: str = "icl_spk"
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    max_frames: int | None = None


@dataclass(slots=True)
class GenerationResult:
    """Completed audio output and telemetry from a synthesis call."""

    audio_path: str
    duration_s: float
    sample_rate: int
    seed: int
    backend: str
    voice_label: str
    lang: str
    elapsed_s: float
    reason: str = ""
    notes: str = ""


_CLIENT_LOCK = threading.Lock()
_GRADIO_CLIENTS: dict[tuple[str, str], Client] = {}
_LOCAL_ENGINE: Any = None
_LOCAL_ENGINE_KEY: tuple[str, str, int] | None = None
_LOCAL_ENGINE_KIND: str = ""


def _get_gradio_client(space_id: str, token: str | None) -> Client:
    key = (space_id, token or "")
    with _CLIENT_LOCK:
        client = _GRADIO_CLIENTS.get(key)
        if client is None:
            client = Client(space_id, token=token, verbose=False)
            _GRADIO_CLIENTS[key] = client
        return client


def _reset_gradio_client(space_id: str, token: str | None) -> None:
    key = (space_id, token or "")
    with _CLIENT_LOCK:
        _GRADIO_CLIENTS.pop(key, None)


def _new_output_wav(prefix: str = "maya2_out_") -> Path:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
    os.close(fd)
    return Path(path)


def _upload_to_studio(ref_wav: Path, token: str | None) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(timeout=45.0, follow_redirects=True) as http:
        with ref_wav.open("rb") as fh:
            resp = http.post(
                f"{HF_STUDIO_URL}/gradio_api/upload",
                headers=headers,
                files={"files": (ref_wav.name, fh, "audio/wav")},
            )
    if resp.status_code != 200:
        raise BackendError(
            f"Failed to upload reference clip to Studio Space ({resp.status_code}): {resp.text[:300]}"
        )
    paths = resp.json()
    if not isinstance(paths, list) or not paths:
        raise BackendError(f"Unexpected upload response from Studio Space: {paths!r}")
    return str(paths[0])


def _download_studio_audio(audio_url: str, token: str | None) -> Path:
    url = audio_url if audio_url.startswith("http") else f"{HF_STUDIO_URL}{audio_url}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(timeout=60.0, follow_redirects=True) as http:
        resp = http.get(url, headers=headers)
        if resp.status_code != 200:
            raise BackendError(
                f"Failed to download generated audio from Studio Space ({resp.status_code})."
            )
        return wav_bytes_to_file(resp.content, _new_output_wav())


def _format_hf_error(err: Exception, space_id: str) -> str:
    msg = str(err).strip() or type(err).__name__
    low = msg.lower()
    if "quota" in low or "zerogpu" in low or "gpu limit" in low:
        return (
            f"ZeroGPU quota exceeded on {space_id}. Switch to 'Local Maya 2 Model', "
            "or paste an HF_TOKEN in Backend Settings."
        )
    if "apperror" in type(err).__name__.lower() or "show_error=true" in low:
        return (
            f"Upstream Space {space_id} returned an error (often anonymous ZeroGPU quota "
            "or invalid reference clip). Switch to 'Local Maya 2 Model' or add an HF_TOKEN."
        )
    return f"Space {space_id} error: {msg}"


def _prepare_clone_clip(ref_audio: str, ref_text: str, require_transcript: bool) -> Path:
    if require_transcript and not (ref_text or "").strip():
        raise BackendError(
            "Voice cloning in ICL mode requires the exact transcript of the reference clip."
        )
    stats = prepare_reference_clip(ref_audio)
    return stats.path


def _run_hf_studio(req: GenerationRequest, cfg: BackendConfig) -> GenerationResult:
    token = cfg.resolved_hf_token()
    client = _get_gradio_client(HF_STUDIO_SPACE, token)
    ref_server_path: str | None = None
    tmp_ref: Path | None = None
    is_clone = bool(req.ref_audio)
    if is_clone:
        tmp_ref = _prepare_clone_clip(req.ref_audio, req.ref_text, require_transcript=True)
        ref_server_path = _upload_to_studio(tmp_ref, token)

    voice_arg = "plain" if req.voice in ("", "(none)", "plain") else req.voice
    t0 = time.perf_counter()
    try:
        job = client.submit(
            text=req.text,
            lang=req.lang,
            voice="Ritvik" if is_clone else voice_arg,
            normalize=req.normalize,
            seed=req.seed,
            randomize=req.randomize_seed and req.seed <= 0,
            ref_audio=ref_server_path,
            ref_text=req.ref_text.strip() if is_clone else "",
            ref_lang=req.ref_lang,
            api_name="/speak_stream",
        )
        final_event: dict | None = None
        pcm_chunks: list[bytes] = []
        for ev in job:
            if not isinstance(ev, dict):
                continue
            ev_type = ev.get("type")
            if ev_type == "error":
                raise BackendError(
                    _format_hf_error(RuntimeError(ev.get("message", "Unknown error")), HF_STUDIO_SPACE)
                )
            if ev_type == "audio" and ev.get("audio"):
                pcm_chunks.append(base64.b64decode(ev["audio"]))
            elif ev_type == "done":
                final_event = ev

        elapsed = round(time.perf_counter() - t0, 2)
        if final_event and final_event.get("audio_url"):
            out_path = _download_studio_audio(final_event["audio_url"], token)
            dur = float(final_event.get("duration") or audio_duration_seconds(out_path))
            seed_used = int(final_event.get("seed") or req.seed)
        elif pcm_chunks:
            out_path = pcm16_bytes_to_wav(b"".join(pcm_chunks), _new_output_wav(), SAMPLE_RATE)
            dur = round(audio_duration_seconds(out_path), 2)
            seed_used = int(req.seed)
        else:
            raise BackendError("Studio Space stream completed without returning audio.")

        label = "cloned" if is_clone else voice_arg
        return GenerationResult(
            audio_path=str(out_path),
            duration_s=dur,
            sample_rate=SAMPLE_RATE,
            seed=seed_used,
            backend="hf_studio",
            voice_label=label,
            lang=req.lang,
            elapsed_s=elapsed,
            reason="stream_done",
        )
    except BackendError:
        raise
    except Exception as exc:
        _reset_gradio_client(HF_STUDIO_SPACE, token)
        raise BackendError(_format_hf_error(exc, HF_STUDIO_SPACE)) from exc
    finally:
        if tmp_ref is not None:
            tmp_ref.unlink(missing_ok=True)


def _run_hf_community(req: GenerationRequest, cfg: BackendConfig) -> GenerationResult:
    token = cfg.resolved_hf_token()
    client = _get_gradio_client(HF_COMMUNITY_SPACE, token)
    is_clone = bool(req.ref_audio)
    tmp_ref: Path | None = None
    ref_payload = None
    if is_clone:
        tmp_ref = _prepare_clone_clip(req.ref_audio, req.ref_text, require_transcript=True)
        ref_payload = handle_file(str(tmp_ref))

    voice_arg = "(none)" if (is_clone or req.voice in ("", "(none)", "plain")) else req.voice
    t0 = time.perf_counter()
    try:
        res = client.predict(
            text=req.text,
            lang=req.lang,
            voice=voice_arg,
            normalize=req.normalize,
            seed=int(req.seed),
            ref_audio=ref_payload,
            ref_text=req.ref_text.strip() if is_clone else "",
            ref_lang=req.ref_lang,
            api_name="/generate",
        )
        if not res or not Path(str(res)).exists():
            raise BackendError("Community Space returned no audio file.")
        out_path = _new_output_wav()
        shutil.copyfile(str(res), out_path)
        elapsed = round(time.perf_counter() - t0, 2)
        dur = round(audio_duration_seconds(out_path), 2)
        return GenerationResult(
            audio_path=str(out_path),
            duration_s=dur,
            sample_rate=SAMPLE_RATE,
            seed=int(req.seed),
            backend="hf_community",
            voice_label="cloned" if is_clone else voice_arg,
            lang=req.lang,
            elapsed_s=elapsed,
        )
    except BackendError:
        raise
    except Exception as exc:
        _reset_gradio_client(HF_COMMUNITY_SPACE, token)
        raise BackendError(_format_hf_error(exc, HF_COMMUNITY_SPACE)) from exc
    finally:
        if tmp_ref is not None:
            tmp_ref.unlink(missing_ok=True)


def _run_maya3_server(req: GenerationRequest, cfg: BackendConfig) -> GenerationResult:
    base = cfg.server_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"

    t0 = time.perf_counter()
    is_clone = bool(req.ref_audio)
    tmp_ref: Path | None = None
    temp_voice_id: str | None = None

    with httpx.Client(timeout=180.0) as http:
        try:
            voice_id = "plain" if req.voice in ("", "(none)") else req.voice
            if is_clone:
                need_text = req.clone_mode in ("icl_spk", "icl")
                tmp_ref = _prepare_clone_clip(req.ref_audio, req.ref_text, require_transcript=need_text)
                audio_b64 = base64.b64encode(tmp_ref.read_bytes()).decode("ascii")
                digest = hashlib.sha1(audio_b64[:4096].encode("ascii")).hexdigest()[:10]
                temp_voice_id = f"studio_tmp_{digest}"
                reg_resp = http.post(
                    f"{base}/v1/voices",
                    json={
                        "voice_id": temp_voice_id,
                        "ref_audio_b64": audio_b64,
                        "ref_text": req.ref_text.strip() or None,
                        "lang": req.ref_lang,
                        "mode": req.clone_mode,
                        "replace": True,
                    },
                )
                if reg_resp.status_code >= 400:
                    raise BackendError(
                        f"maya3.server rejected clone registration ({reg_resp.status_code}): "
                        f"{reg_resp.text[:300]}"
                    )
                voice_id = temp_voice_id

            payload: dict[str, object] = {
                "model": "maya3",
                "input": req.text,
                "language": req.lang,
                "voice": voice_id,
                "response_format": "wav",
                "normalize": req.normalize,
            }
            if req.seed and req.seed > 0:
                payload["seed"] = int(req.seed)
            if req.temperature is not None:
                payload["temperature"] = float(req.temperature)
            if req.top_k is not None:
                payload["top_k"] = int(req.top_k)
            if req.top_p is not None:
                payload["top_p"] = float(req.top_p)
            if req.repetition_penalty is not None:
                payload["repetition_penalty"] = float(req.repetition_penalty)
            if req.max_frames is not None:
                payload["max_frames"] = int(req.max_frames)

            resp = http.post(f"{base}/v1/audio/speech", json=payload)
            if resp.status_code >= 400:
                raise BackendError(
                    f"maya3.server synthesis failed ({resp.status_code}): {resp.text[:300]}"
                )
            out_path = wav_bytes_to_file(resp.content, _new_output_wav())
            dur = round(audio_duration_seconds(out_path), 2)
            elapsed = round(time.perf_counter() - t0, 2)
            return GenerationResult(
                audio_path=str(out_path),
                duration_s=dur,
                sample_rate=SAMPLE_RATE,
                seed=int(req.seed),
                backend="maya3_server",
                voice_label=voice_id if not is_clone else f"cloned ({req.clone_mode})",
                lang=req.lang,
                elapsed_s=elapsed,
                reason=resp.headers.get("X-Maya3-Voice", ""),
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"Could not reach maya3.server at {base}: {exc}") from exc
        finally:
            if temp_voice_id:
                try:
                    http.delete(f"{base}/v1/voices/{temp_voice_id}")
                except Exception:
                    pass
            if tmp_ref is not None:
                tmp_ref.unlink(missing_ok=True)


def _run_maya_cloud(req: GenerationRequest, cfg: BackendConfig) -> GenerationResult:
    if req.ref_audio:
        raise BackendError(
            "Hosted Maya Cloud API (tts.mayaresearch.ai) does not accept reference-audio "
            "voice cloning. Switch to Local Maya 2 Model, maya3.server, or HF Space."
        )
    api_key = cfg.resolved_maya_key()
    if not api_key:
        raise BackendError(
            "Maya Cloud API requires MAYA_API_KEY. Paste your key in Backend Settings "
            "or switch to 'Local Maya 2 Model'."
        )
    voice = req.voice if req.voice in CLOUD_VOICES else CLOUD_VOICES[0]
    note = ""
    if req.voice not in CLOUD_VOICES:
        note = f"Cloud API supports Ananya/Arjun; mapped '{req.voice}' -> '{voice}'."

    body: dict[str, object] = {
        "text": req.text,
        "voice": voice,
        "language": req.lang,
        "model": cfg.cloud_model or "Maya 2 Native",
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=90.0) as http:
        try:
            resp = http.post(
                MAYA_CLOUD_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"Maya Cloud API network error: {exc}") from exc
    if resp.status_code >= 400:
        raise BackendError(f"Maya Cloud API error ({resp.status_code}): {resp.text[:300]}")

    out_path = pcm16_bytes_to_wav(resp.content, _new_output_wav(), SAMPLE_RATE)
    dur = round(audio_duration_seconds(out_path), 2)
    return GenerationResult(
        audio_path=str(out_path),
        duration_s=dur,
        sample_rate=SAMPLE_RATE,
        seed=0,
        backend="maya_cloud",
        voice_label=voice,
        lang=req.lang,
        elapsed_s=round(time.perf_counter() - t0, 2),
        reason=cfg.cloud_model,
        notes=note,
    )


def _close_cached_local_engine() -> None:
    global _LOCAL_ENGINE, _LOCAL_ENGINE_KEY, _LOCAL_ENGINE_KIND
    if _LOCAL_ENGINE is not None:
        try:
            _LOCAL_ENGINE.close()
        except Exception:
            pass
    _LOCAL_ENGINE = None
    _LOCAL_ENGINE_KEY = None
    _LOCAL_ENGINE_KIND = ""
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def unload_local_engine() -> str:
    """Release the cached local Maya 2 model from unified/GPU memory."""
    with _CLIENT_LOCK:
        was_loaded = _LOCAL_ENGINE is not None
        _close_cached_local_engine()
    return "Local model unloaded from memory." if was_loaded else "No local model was loaded."


def _build_maya_mac_kwargs(cfg: BackendConfig) -> dict[str, Any]:
    precision = cfg.local_precision or "mlx_8bit"
    kw: dict[str, Any] = {
        "capacity": int(cfg.local_capacity or 2048),
        "warmup": True,
    }
    if cfg.local_models_dir.strip():
        kw["models_dir"] = os.path.expanduser(cfg.local_models_dir.strip())

    if precision == "mlx_8bit":
        kw.update(backend="mlx", backbone_bits=8, depth_bits=8)
    elif precision == "mlx_4bit":
        kw.update(backend="mlx", backbone_bits=4, depth_bits=4)
    elif precision == "mlx_fp16":
        kw.update(backend="mlx", backbone_dtype="fp16", depth_dtype="fp16")
    elif precision == "mlx_bf16":
        kw.update(backend="mlx", backbone_dtype="bf16", depth_dtype="bf16")
    elif precision == "torch_mps":
        kw.update(backend="torch")
    else:
        kw.update(backend="auto", backbone_bits=8, depth_bits=8)
    return kw


def _get_local_engine(cfg: BackendConfig) -> tuple[Any, str]:
    global _LOCAL_ENGINE, _LOCAL_ENGINE_KEY, _LOCAL_ENGINE_KIND
    key = (
        cfg.local_precision or "mlx_8bit",
        (cfg.local_models_dir or cfg.local_model_id or "").strip(),
        int(cfg.local_capacity or 2048),
    )
    with _CLIENT_LOCK:
        if _LOCAL_ENGINE is not None and _LOCAL_ENGINE_KEY == key:
            return _LOCAL_ENGINE, _LOCAL_ENGINE_KIND

        _close_cached_local_engine()

        prefer_cuda = cfg.local_precision == "cuda_maya3"
        if not prefer_cuda:
            try:
                from maya_mac import MayaTTS

                tts = MayaTTS.from_pretrained(**_build_maya_mac_kwargs(cfg))
                _LOCAL_ENGINE = tts
                _LOCAL_ENGINE_KEY = key
                _LOCAL_ENGINE_KIND = "maya_mac"
                return tts, "maya_mac"
            except ImportError:
                pass

        try:
            from maya3 import Maya3TTS

            tts = Maya3TTS.from_pretrained(cfg.local_model_id or HF_MODEL_REPO, max_seats=1)
            _LOCAL_ENGINE = tts
            _LOCAL_ENGINE_KEY = key
            _LOCAL_ENGINE_KIND = "maya3"
            return tts, "maya3"
        except ImportError as exc:
            raise BackendError(
                "No local Maya 2 runtime is available in this environment. "
                "On Apple Silicon Macs run `uv sync` with Python 3.12 (installs `maya-mac`); "
                "on Linux + NVIDIA GPUs install `maya3` with CUDA 12.8."
            ) from exc


def preload_local_engine(cfg: BackendConfig) -> str:
    """Eagerly load and warm up the local Maya 2 model."""
    t0 = time.perf_counter()
    tts, kind = _get_local_engine(cfg)
    dt = round(time.perf_counter() - t0, 1)
    dev = getattr(tts, "device", kind)
    load_s = getattr(tts, "load_s", dt)
    return (
        f"Local model ready ({kind}, device={dev}, precision={cfg.local_precision}) "
        f"in {load_s:.1f}s · {len(tts.voices)} voices loaded."
    )


def _run_local_model(req: GenerationRequest, cfg: BackendConfig) -> GenerationResult:
    tts, kind = _get_local_engine(cfg)
    is_clone = bool(req.ref_audio)
    tmp_ref: Path | None = None
    t0 = time.perf_counter()
    seed_used = int(req.seed) if (req.seed and req.seed > 0) else (
        random.randint(1, 2**31 - 1) if req.randomize_seed else 0
    )
    try:
        kw: dict[str, object] = {
            "lang": req.lang,
            "normalize": req.normalize,
            "return_info": True,
        }
        if seed_used > 0 or not req.randomize_seed:
            kw["seed"] = int(seed_used)
        if req.temperature is not None:
            kw["temperature"] = float(req.temperature)
        if req.top_k is not None:
            kw["top_k"] = int(req.top_k)
        if req.top_p is not None:
            kw["top_p"] = float(req.top_p)
        if req.repetition_penalty is not None:
            kw["repetition_penalty"] = float(req.repetition_penalty)
        if req.max_frames is not None:
            kw["max_frames"] = int(req.max_frames)

        if is_clone:
            need_text = req.clone_mode in ("icl_spk", "icl")
            tmp_ref = _prepare_clone_clip(req.ref_audio, req.ref_text, require_transcript=need_text)
            # maya-mac and maya3 forbid combining a named voice with ref_audio.
            kw["voice"] = None
            kw["ref_audio"] = str(tmp_ref)
            kw["ref_text"] = req.ref_text.strip()
            kw["ref_lang"] = req.ref_lang
            kw["mode"] = req.clone_mode
        else:
            kw["voice"] = "plain" if req.voice in ("", "(none)") else req.voice

        wav, info = tts.generate(req.text, **kw)
        out_path = write_wav(_new_output_wav(), wav, SAMPLE_RATE)
        dur = round(audio_duration_seconds(out_path), 2)
        elapsed = round(time.perf_counter() - t0, 2)

        info = info or {}
        rtf = info.get("rtf")
        ttfa = info.get("ttfa_s")
        perf_notes = []
        if isinstance(rtf, (int, float)):
            perf_notes.append(f"RTF={rtf:.2f}")
        if isinstance(ttfa, (int, float)):
            perf_notes.append(f"TTFA={ttfa * 1000:.0f}ms")
        perf_notes.append(f"device={getattr(tts, 'device', kind)}")

        return GenerationResult(
            audio_path=str(out_path),
            duration_s=dur,
            sample_rate=SAMPLE_RATE,
            seed=seed_used,
            backend=f"local ({kind}:{cfg.local_precision})",
            voice_label=f"cloned ({req.clone_mode})" if is_clone else str(kw.get("voice", "plain")),
            lang=req.lang,
            elapsed_s=elapsed,
            reason=str(info.get("reason", "")),
            notes=" · ".join(perf_notes),
        )
    finally:
        if tmp_ref is not None:
            tmp_ref.unlink(missing_ok=True)


def synthesize_maya2(req: GenerationRequest, cfg: BackendConfig) -> GenerationResult:
    """Dispatch synthesis or voice cloning request to the selected Maya 2 backend."""
    text = (req.text or "").strip()
    if not text:
        raise BackendError("Please enter text to synthesize.")

    dispatch = {
        "local": _run_local_model,
        "local_maya3": _run_local_model,
        "hf_studio": _run_hf_studio,
        "hf_community": _run_hf_community,
        "maya3_server": _run_maya3_server,
        "maya_cloud": _run_maya_cloud,
    }
    handler = dispatch.get(cfg.backend)
    if handler is None:
        raise BackendError(f"Unknown backend: {cfg.backend!r}")
    return handler(req, cfg)


def design_voice_via_maya1(
    *,
    description: str,
    anchor_text: str,
    preset_name: str = "Female British",
    temperature: float = 0.4,
    max_tokens: int = 1500,
    hf_token: str = "",
) -> tuple[str, str]:
    """Generate a prompt-designed voice prototype clip via the Maya 1 Space.

    Returns (local_wav_path, status_message).
    """
    desc = (description or "").strip()
    text = (anchor_text or "").strip()
    if not desc or not text:
        raise BackendError("Both voice description and design anchor text are required.")

    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip() or None
    client = _get_gradio_client(HF_MAYA1_SPACE, token)
    t0 = time.perf_counter()
    try:
        audio_file, status = client.predict(
            preset_name=preset_name,
            description=desc,
            text=text,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            api_name="/generate_speech",
        )
        if not audio_file or not Path(str(audio_file)).exists():
            raise BackendError(f"Maya 1 returned no audio ({status}).")
        out_path = _new_output_wav(prefix="maya1_design_")
        shutil.copyfile(str(audio_file), out_path)
        elapsed = round(time.perf_counter() - t0, 2)
        dur = round(audio_duration_seconds(out_path), 2)
        return str(out_path), f"Designed voice prototype ({dur:.2f}s) in {elapsed:.1f}s — {status}"
    except BackendError:
        raise
    except Exception as exc:
        _reset_gradio_client(HF_MAYA1_SPACE, token)
        raise BackendError(_format_hf_error(exc, HF_MAYA1_SPACE)) from exc


def register_voice_on_server(
    cfg: BackendConfig,
    *,
    voice_id: str,
    ref_audio_path: str,
    ref_text: str,
    ref_lang: str,
    mode: str = "icl_spk",
) -> str:
    """Persist a custom cloned/designed voice on a running maya3.server instance."""
    if cfg.backend != "maya3_server":
        raise BackendError("Server registration applies only when backend is 'maya3_server'.")
    base = cfg.server_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    tmp_ref = _prepare_clone_clip(ref_audio_path, ref_text, require_transcript=mode != "spk")
    try:
        audio_b64 = base64.b64encode(tmp_ref.read_bytes()).decode("ascii")
        with httpx.Client(timeout=45.0) as http:
            resp = http.post(
                f"{base}/v1/voices",
                json={
                    "voice_id": voice_id,
                    "ref_audio_b64": audio_b64,
                    "ref_text": ref_text.strip() or None,
                    "lang": ref_lang,
                    "mode": mode,
                    "replace": True,
                },
            )
        if resp.status_code >= 400:
            raise BackendError(f"Server rejected voice registration ({resp.status_code}): {resp.text[:300]}")
        return f"Registered '{voice_id}' on {base}."
    finally:
        tmp_ref.unlink(missing_ok=True)


def test_backend_connection(cfg: BackendConfig) -> str:
    """Verify connectivity and readiness for the selected Maya 2 backend."""
    if cfg.backend in ("local", "local_maya3"):
        lines: list[str] = []
        try:
            import maya_mac

            mlx_ok = False
            try:
                import mlx.core as mx  # noqa: F401

                mlx_ok = True
            except Exception:
                pass
            loaded = "loaded" if _LOCAL_ENGINE is not None else "ready (will lazy-load on first run)"
            lines.append(
                f"`maya-mac` installed (MLX={mlx_ok}, precision={cfg.local_precision}, status={loaded})."
            )
        except ImportError:
            pass

        try:
            import maya3  # noqa: F401

            lines.append("`maya3` CUDA package installed.")
        except ImportError:
            pass

        if not lines:
            return (
                "Neither `maya-mac` nor `maya3` is installed in the active environment. "
                "Run `uv sync` with Python 3.12 on Apple Silicon."
            )

        from huggingface_hub import constants

        maya_cache = Path(constants.HF_HUB_CACHE) / "models--maya-research--maya2-native"
        if maya_cache.exists():
            lines.append(f"Cached weights detected at `{maya_cache}`.")
        return " ".join(lines)

    if cfg.backend == "hf_studio":
        client = _get_gradio_client(HF_STUDIO_SPACE, cfg.resolved_hf_token())
        info = client.view_api(print_info=False, return_format="dict")
        eps = list(info.get("named_endpoints", {}).keys())
        auth = "authenticated HF token" if cfg.resolved_hf_token() else "anonymous ZeroGPU quota"
        return f"Connected to {HF_STUDIO_SPACE} ({auth}). Endpoints: {', '.join(eps)}"

    if cfg.backend == "hf_community":
        client = _get_gradio_client(HF_COMMUNITY_SPACE, cfg.resolved_hf_token())
        info = client.view_api(print_info=False, return_format="dict")
        eps = list(info.get("named_endpoints", {}).keys())
        auth = "authenticated HF token" if cfg.resolved_hf_token() else "anonymous ZeroGPU quota"
        return f"Connected to {HF_COMMUNITY_SPACE} ({auth}). Endpoints: {', '.join(eps)}"

    if cfg.backend == "maya3_server":
        base = cfg.server_url.strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            base = f"http://{base}"
        with httpx.Client(timeout=10.0) as http:
            r = http.get(f"{base}/health")
        return f"maya3.server {base} responded HTTP {r.status_code}: {r.text[:240]}"

    if cfg.backend == "maya_cloud":
        key = cfg.resolved_maya_key()
        if not key:
            return "No MAYA_API_KEY configured. Provide a key or switch to Local Maya 2 Model."
        return f"Maya Cloud API configured (model={cfg.cloud_model}, endpoint={MAYA_CLOUD_URL})."

    return f"Unknown backend: {cfg.backend}"
