"""Gradio 6 studio UI for testing Maya 2 locally, voice cloning, and voice design."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from maya2_studio.audio_utils import prepare_reference_clip, sculpt_voice_reference
from maya2_studio.backends import (
    BACKEND_CHOICES,
    LOCAL_PRECISION_CHOICES,
    BackendConfig,
    BackendError,
    GenerationRequest,
    design_voice_via_maya1,
    preload_local_engine,
    register_voice_on_server,
    synthesize_maya2,
    test_backend_connection,
    unload_local_engine,
)
from maya2_studio.catalog import (
    BUILTIN_VOICES,
    DESIGN_PRESETS,
    EMOTION_TAGS,
    LANGUAGE_BY_CODE,
    LANGUAGE_CHOICES,
    compose_design_description,
    default_cache_dir,
    fetch_official_sample,
    filter_builtin_voices,
    format_persona_summary,
)
from maya2_studio.voice_store import VoiceStore

CUSTOM_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}
.maya-hero {
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 16px 20px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(16, 185, 129, 0.08));
    margin-bottom: 10px;
}
.maya-hero h1 {
    margin: 0 0 4px 0;
    font-size: 1.65rem;
}
.maya-hero p {
    margin: 0;
    opacity: 0.85;
}
.tag-btn button {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
    font-size: 0.8rem !important;
}
"""

CUSTOM_PREFIX = "custom:"


def _builtin_voice_choices(store: VoiceStore) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for p in store.list_profiles():
        tag = {
            "cloned": "Cloned",
            "designed_maya1": "Designed (Maya1)",
            "designed_sculpt": "Sculpted",
        }.get(p.source_type, "Custom")
        choices.append((f"★ {p.name} [{tag} · {p.ref_lang}]", f"{CUSTOM_PREFIX}{p.name}"))
    for v in BUILTIN_VOICES:
        choices.append((f"{v.name} ({v.gender}, {v.age_band})", v.name))
    choices.append(("plain (unconditioned baseline)", "plain"))
    return choices


def _saved_voice_names(store: VoiceStore) -> list[str]:
    return [p.name for p in store.list_profiles()]


def _saved_voice_rows(store: VoiceStore) -> list[list[str]]:
    rows: list[list[str]] = []
    for p in store.list_profiles():
        rows.append(
            [
                p.name,
                p.source_type,
                p.ref_lang,
                p.mode,
                p.ref_text[:60] + ("…" if len(p.ref_text) > 60 else ""),
                p.notes or p.design_prompt[:60],
            ]
        )
    return rows


def _make_backend_cfg(
    backend: str,
    local_precision: str,
    local_models_dir: str,
    hf_token: str,
    server_url: str,
    maya_api_key: str,
    cloud_model: str,
) -> BackendConfig:
    return BackendConfig(
        backend=backend,
        local_precision=local_precision or "mlx_8bit",
        local_models_dir=local_models_dir or "",
        hf_token=hf_token or "",
        server_url=server_url or "http://127.0.0.1:8080",
        maya_api_key=maya_api_key or "",
        cloud_model=cloud_model or "Maya 2 Native",
    )


def create_demo(
    *,
    default_backend: str = "local",
    default_precision: str = "mlx_8bit",
    default_server_url: str = "http://127.0.0.1:8080",
    store: VoiceStore | None = None,
) -> tuple[gr.Blocks, dict[str, Any]]:
    """Build the Gradio Blocks application and recommended launch kwargs."""
    voice_store = store or VoiceStore()

    with gr.Blocks(title="Maya 2 Studio — Local TTS, Cloning & Voice Design") as demo:
        history_state = gr.State([])
        sculpted_ref_state = gr.State(None)
        sculpted_transcript_state = gr.State("")
        sculpted_lang_state = gr.State("hi")
        maya1_clip_state = gr.State(None)
        maya1_transcript_state = gr.State("")

        gr.HTML(
            """
            <div class="maya-hero">
              <h1>🎙️ Maya 2 Studio</h1>
              <p>Local Apple Silicon MLX / MPS / CUDA inference, zero-shot cross-lingual voice cloning,
              and voice design for <b>Maya 2 Native</b> (11 Indic languages + English · 20 built-in voices · 24 kHz audio).</p>
            </div>
            """
        )

        # ------------------------------------------------------------------ Shared Backend Bar
        with gr.Accordion("⚙️ Local Model & Inference Backend", open=True):
            with gr.Row():
                backend_dd = gr.Dropdown(
                    choices=BACKEND_CHOICES,
                    value=default_backend,
                    label="Inference Backend",
                    scale=2,
                )
                local_precision_dd = gr.Dropdown(
                    choices=LOCAL_PRECISION_CHOICES,
                    value=default_precision,
                    label="Local Engine / Quantization",
                    scale=2,
                )
                local_models_dir_tb = gr.Textbox(
                    label="Local Model Dir (optional $MAYA_MODEL_DIR; blank uses HF cache)",
                    value=os.environ.get("MAYA_MODEL_DIR", ""),
                    placeholder="Auto-resolve from ~/.cache/huggingface/hub",
                    scale=2,
                )
            with gr.Row():
                preload_btn = gr.Button("🚀 Preload Local Model", variant="primary", scale=1)
                unload_btn = gr.Button("🧹 Unload Local Model", variant="secondary", scale=1)
                test_conn_btn = gr.Button("Check Backend Status", variant="secondary", scale=1)

            with gr.Accordion("Optional Remote / Server / Cloud Settings", open=False):
                with gr.Row():
                    hf_token_tb = gr.Textbox(
                        label="HF Token (for HF Spaces / Maya 1 remote design)",
                        type="password",
                        value=os.environ.get("HF_TOKEN", ""),
                        placeholder="hf_...",
                        scale=2,
                    )
                    server_url_tb = gr.Textbox(
                        label="maya3.server URL",
                        value=default_server_url,
                        placeholder="http://127.0.0.1:8080",
                        scale=2,
                    )
                    maya_key_tb = gr.Textbox(
                        label="MAYA_API_KEY (Maya Cloud API)",
                        type="password",
                        value=os.environ.get("MAYA_API_KEY", ""),
                        placeholder="maya_...",
                        scale=2,
                    )
                    cloud_model_dd = gr.Dropdown(
                        choices=["Maya 2 Native", "Maya 2 Native Emotional", "Maya 2 Global"],
                        value="Maya 2 Native",
                        label="Maya Cloud Model",
                        scale=1,
                    )

            conn_status_md = gr.Markdown(
                "Active backend: **Local Maya 2 Model** (lazy-loads on first generation, or click **Preload Local Model**)."
            )

        backend_inputs = [
            backend_dd,
            local_precision_dd,
            local_models_dir_tb,
            hf_token_tb,
            server_url_tb,
            maya_key_tb,
            cloud_model_dd,
        ]

        def on_test_connection(backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model):
            cfg = _make_backend_cfg(
                backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model
            )
            try:
                return f"✅ {test_backend_connection(cfg)}"
            except Exception as exc:
                return f"⚠️ {exc}"

        def on_preload_local(backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model):
            cfg = _make_backend_cfg(
                backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model
            )
            try:
                return f"✅ {preload_local_engine(cfg)}"
            except Exception as exc:
                return f"⚠️ {exc}"

        test_conn_btn.click(fn=on_test_connection, inputs=backend_inputs, outputs=[conn_status_md])
        preload_btn.click(fn=on_preload_local, inputs=backend_inputs, outputs=[conn_status_md])
        unload_btn.click(fn=lambda: f"ℹ️ {unload_local_engine()}", outputs=[conn_status_md])

        with gr.Tabs():
            # ============================================================== TAB 1: TTS
            with gr.Tab("🗣️ Voice Generation (TTS)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        tts_lang = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            value="hi",
                            label="Target Language",
                            filterable=True,
                        )
                        with gr.Row():
                            load_example_btn = gr.Button(
                                "Load Showcase Text for Language", size="sm"
                            )
                            play_official_btn = gr.Button(
                                "Play Official HF Reference Sample", size="sm", variant="secondary"
                            )

                        tts_text = gr.Textbox(
                            label="Text to Speak",
                            value=LANGUAGE_BY_CODE["hi"].sample_text,
                            lines=4,
                            placeholder="Enter text in any of the 12 supported languages…",
                        )

                        gr.Markdown(
                            "**Inline Emotion / Delivery Tags** *(best-effort on local Maya 2 Native; verified on Maya 2 Native Emotional)*"
                        )
                        with gr.Row():
                            tag_buttons = [
                                gr.Button(tag, size="sm", elem_classes=["tag-btn"])
                                for tag, _ in EMOTION_TAGS[:5]
                            ]
                        with gr.Row():
                            tag_buttons += [
                                gr.Button(tag, size="sm", elem_classes=["tag-btn"])
                                for tag, _ in EMOTION_TAGS[5:]
                            ]

                        tts_voice = gr.Dropdown(
                            choices=_builtin_voice_choices(voice_store),
                            value="Ritvik",
                            label="Voice (Built-in Maya 2 Personas or Saved Custom Voices)",
                            filterable=True,
                        )
                        persona_info_md = gr.Markdown(format_persona_summary("Ritvik"))

                        with gr.Accordion("Generation & Sampling Controls", open=False):
                            with gr.Row():
                                tts_normalize = gr.Checkbox(
                                    value=True,
                                    label="Normalize text (numbers, currency, dates, URLs)",
                                )
                                tts_seed = gr.Number(
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    maximum=2**31 - 1,
                                    label="Seed (0 = random)",
                                )
                                tts_randomize = gr.Checkbox(
                                    value=True,
                                    label="Randomize when seed = 0",
                                )
                            gr.Markdown(
                                "*Sampler overrides apply to Local Maya 2 (`maya-mac`/`maya3`) and `maya3.server`:*"
                            )
                            with gr.Row():
                                tts_temp = gr.Slider(
                                    0.1, 1.5, value=0.9, step=0.05, label="Temperature"
                                )
                                tts_top_k = gr.Slider(
                                    1, 512, value=64, step=1, label="Top-K"
                                )
                                tts_top_p = gr.Slider(
                                    0.1, 1.0, value=1.0, step=0.05, label="Top-P"
                                )
                                tts_rep = gr.Slider(
                                    0.0, 2.5, value=0.8, step=0.05, label="Repetition Penalty"
                                )
                                tts_use_sampler = gr.Checkbox(
                                    value=False,
                                    label="Override sampler defaults",
                                )

                    with gr.Column(scale=2):
                        tts_run_btn = gr.Button(
                            "▶ Generate Speech", variant="primary", size="lg"
                        )
                        tts_audio_out = gr.Audio(
                            label="Generated Speech (24 kHz mono)",
                            type="filepath",
                        )
                        tts_meta_md = gr.Markdown("Ready to generate.")
                        tts_history_df = gr.Dataframe(
                            headers=["Voice", "Lang", "Duration (s)", "Backend", "Text Preview"],
                            datatype=["str", "str", "number", "str", "str"],
                            type="array",
                            label="Session History (latest 10)",
                            interactive=False,
                        )

                def on_select_voice(voice_val: str):
                    if voice_val and voice_val.startswith(CUSTOM_PREFIX):
                        name = voice_val[len(CUSTOM_PREFIX) :]
                        prof = voice_store.get(name)
                        if prof:
                            return (
                                f"**★ {prof.name}** (`{prof.source_type}`, ref lang `{prof.ref_lang}`, "
                                f"mode `{prof.mode}`)  \nTranscript: *{prof.ref_text or '(none)'}*"
                            )
                    return format_persona_summary(voice_val)

                tts_voice.change(fn=on_select_voice, inputs=[tts_voice], outputs=[persona_info_md])

                def on_load_example(lang_code: str):
                    info = LANGUAGE_BY_CODE[lang_code]
                    return info.sample_text, info.sample_voice, format_persona_summary(info.sample_voice)

                load_example_btn.click(
                    fn=on_load_example,
                    inputs=[tts_lang],
                    outputs=[tts_text, tts_voice, persona_info_md],
                )

                def on_play_official(lang_code: str):
                    try:
                        wav_path, transcript, code = fetch_official_sample(lang_code)
                        info = LANGUAGE_BY_CODE[code]
                        return (
                            str(wav_path),
                            f"🎧 Official Maya 2 sample: **{info.sample_voice}** speaking **{info.name}** (`{info.sample_filename}`).  \nText: *{transcript}*",
                        )
                    except Exception as exc:
                        raise gr.Error(f"Could not download official sample: {exc}")

                play_official_btn.click(
                    fn=on_play_official,
                    inputs=[tts_lang],
                    outputs=[tts_audio_out, tts_meta_md],
                )

                for btn, (tag_str, _) in zip(tag_buttons, EMOTION_TAGS):

                    def _make_tag_inserter(tag_literal: str):
                        def _insert(current_text: str) -> str:
                            cur = (current_text or "").strip()
                            if not cur:
                                return f"{tag_literal} "
                            if cur.startswith("["):
                                return f"{cur} {tag_literal}"
                            return f"{tag_literal} {cur}"

                        return _insert

                    btn.click(fn=_make_tag_inserter(tag_str), inputs=[tts_text], outputs=[tts_text])

                def on_tts_generate(
                    text: str,
                    lang: str,
                    voice_val: str,
                    normalize: bool,
                    seed: float,
                    randomize: bool,
                    use_sampler: bool,
                    temp: float,
                    top_k: float,
                    top_p: float,
                    rep_pen: float,
                    backend: str,
                    precision: str,
                    models_dir: str,
                    hf_token: str,
                    server_url: str,
                    maya_key: str,
                    cloud_model: str,
                    history: list,
                ):
                    cfg = _make_backend_cfg(
                        backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model
                    )
                    ref_audio = None
                    ref_text = ""
                    ref_lang = lang
                    clone_mode = "icl_spk"
                    voice_name = voice_val

                    if voice_val and voice_val.startswith(CUSTOM_PREFIX):
                        custom_name = voice_val[len(CUSTOM_PREFIX) :]
                        prof = voice_store.get(custom_name)
                        if prof is None:
                            raise gr.Error(f"Saved voice '{custom_name}' was not found on disk.")
                        ref_audio = prof.ref_audio_path
                        ref_text = prof.ref_text
                        ref_lang = prof.ref_lang
                        clone_mode = prof.mode
                        voice_name = prof.name

                    req = GenerationRequest(
                        text=text,
                        lang=lang,
                        voice=voice_name,
                        normalize=bool(normalize),
                        seed=int(seed or 0),
                        randomize_seed=bool(randomize),
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        ref_lang=ref_lang,
                        clone_mode=clone_mode,
                        temperature=float(temp) if use_sampler else None,
                        top_k=int(top_k) if use_sampler else None,
                        top_p=float(top_p) if use_sampler else None,
                        repetition_penalty=float(rep_pen) if use_sampler else None,
                    )
                    try:
                        res = synthesize_maya2(req, cfg)
                    except BackendError as exc:
                        raise gr.Error(str(exc))

                    display_voice = (
                        voice_val[len(CUSTOM_PREFIX) :]
                        if voice_val and voice_val.startswith(CUSTOM_PREFIX)
                        else res.voice_label
                    )
                    meta = (
                        f"✅ **{res.duration_s:.1f}s audio** in **{res.elapsed_s:.1f}s** · "
                        f"voice: `{display_voice}` · lang: `{res.lang}` · seed: `{res.seed}` · "
                        f"backend: `{res.backend}`"
                    )
                    if res.notes:
                        meta += f"  \nℹ️ {res.notes}"

                    new_row = [
                        display_voice,
                        res.lang,
                        round(res.duration_s, 2),
                        res.backend,
                        text.strip()[:45] + ("…" if len(text.strip()) > 45 else ""),
                    ]
                    updated = [new_row] + list(history or [])
                    updated = updated[:10]
                    return res.audio_path, meta, updated, updated

                tts_run_btn.click(
                    fn=on_tts_generate,
                    inputs=[
                        tts_text,
                        tts_lang,
                        tts_voice,
                        tts_normalize,
                        tts_seed,
                        tts_randomize,
                        tts_use_sampler,
                        tts_temp,
                        tts_top_k,
                        tts_top_p,
                        tts_rep,
                        *backend_inputs,
                        history_state,
                    ],
                    outputs=[tts_audio_out, tts_meta_md, tts_history_df, history_state],
                )

            # ============================================================== TAB 2: VOICE CLONING
            with gr.Tab("🧬 Voice Cloning"):
                gr.Markdown(
                    "Clone any speaker locally from a clean **3–10 second** reference clip and synthesize speech "
                    "in **any of the 12 Maya 2 languages** (including cross-lingual cloning, e.g. Hindi reference speaking Tamil or English).  \n"
                    "*Ethical use: only clone voices you own or have explicit speaker consent to use.*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio Clip (3–10 s clean speech)",
                            sources=["upload", "microphone"],
                            type="filepath",
                            format="wav",
                        )
                        with gr.Row():
                            load_ritvik_ref_btn = gr.Button(
                                "Load Official Ritvik Sample (hi)", size="sm"
                            )
                            inspect_ref_btn = gr.Button("Inspect Clip", size="sm")
                        clone_clip_status = gr.Markdown("Upload or record a reference clip.")

                        clone_ref_text = gr.Textbox(
                            label="Exact Transcript of Reference Clip",
                            placeholder="Type exactly what is spoken in the reference audio…",
                            lines=3,
                        )
                        with gr.Row():
                            clone_ref_lang = gr.Dropdown(
                                choices=LANGUAGE_CHOICES,
                                value="hi",
                                label="Reference Language",
                                filterable=True,
                            )
                            clone_mode_dd = gr.Dropdown(
                                choices=[
                                    ("icl_spk — Audio + ECAPA (default, best similarity)", "icl_spk"),
                                    ("icl — Reference audio ICL only", "icl"),
                                    ("spk — ECAPA only (no transcript needed locally)", "spk"),
                                ],
                                value="icl_spk",
                                label="Conditioning Mode",
                            )

                    with gr.Column(scale=1):
                        clone_target_lang = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            value="en",
                            label="Target Output Language (Cross-Lingual OK)",
                            filterable=True,
                        )
                        clone_target_text = gr.Textbox(
                            label="Target Text to Speak in Cloned Voice",
                            value="Hello! This is a cross-lingual voice cloning test powered locally by Maya 2.",
                            lines=4,
                        )
                        with gr.Row():
                            clone_normalize = gr.Checkbox(value=True, label="Normalize text")
                            clone_seed = gr.Number(
                                value=0, precision=0, minimum=0, label="Seed (0 = random)"
                            )
                            clone_randomize = gr.Checkbox(value=True, label="Randomize seed")

                        clone_run_btn = gr.Button(
                            "▶ Generate Cloned Speech", variant="primary", size="lg"
                        )
                        clone_audio_out = gr.Audio(
                            label="Cloned Speech Output (24 kHz mono)",
                            type="filepath",
                        )
                        clone_meta_md = gr.Markdown("Ready to clone.")

                        with gr.Accordion("💾 Save Cloned Voice to Voice Library", open=True):
                            with gr.Row():
                                save_clone_name = gr.Textbox(
                                    label="Voice Name",
                                    placeholder="e.g., my_custom_narrator",
                                )
                                save_clone_notes = gr.Textbox(
                                    label="Notes (optional)",
                                    placeholder="Warm conversational voice",
                                )
                            save_clone_btn = gr.Button("Save to Voice Library", variant="secondary")
                            save_clone_status = gr.Markdown("")

                def on_load_ritvik_ref():
                    try:
                        path, txt, code = fetch_official_sample("hi")
                        stats = prepare_reference_clip(path)
                        return (
                            str(path),
                            txt,
                            code,
                            f"✅ Loaded official `hi_ritvik.wav` ({stats.duration_s}s, {stats.sample_rate} Hz).",
                        )
                    except Exception as exc:
                        raise gr.Error(f"Failed to load sample clip: {exc}")

                load_ritvik_ref_btn.click(
                    fn=on_load_ritvik_ref,
                    outputs=[clone_ref_audio, clone_ref_text, clone_ref_lang, clone_clip_status],
                )

                def on_inspect_ref(ref_path: str | None):
                    if not ref_path:
                        return "⚠️ Upload or record a reference clip first."
                    try:
                        stats = prepare_reference_clip(ref_path)
                        stats.path.unlink(missing_ok=True)
                        msg = f"✅ Duration: **{stats.duration_s}s** · Sample rate: `{stats.sample_rate} Hz` · Peak: `{stats.peak}`"
                        if stats.warning:
                            msg += f"  \n⚠️ {stats.warning}"
                        return msg
                    except Exception as exc:
                        return f"⚠️ {exc}"

                inspect_ref_btn.click(
                    fn=on_inspect_ref, inputs=[clone_ref_audio], outputs=[clone_clip_status]
                )

                def on_clone_generate(
                    ref_audio: str | None,
                    ref_text: str,
                    ref_lang: str,
                    mode: str,
                    target_text: str,
                    target_lang: str,
                    normalize: bool,
                    seed: float,
                    randomize: bool,
                    backend: str,
                    precision: str,
                    models_dir: str,
                    hf_token: str,
                    server_url: str,
                    maya_key: str,
                    cloud_model: str,
                ):
                    if not ref_audio:
                        raise gr.Error("Upload or record a reference clip first.")
                    cfg = _make_backend_cfg(
                        backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model
                    )
                    req = GenerationRequest(
                        text=target_text,
                        lang=target_lang,
                        voice="cloned",
                        normalize=bool(normalize),
                        seed=int(seed or 0),
                        randomize_seed=bool(randomize),
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        ref_lang=ref_lang,
                        clone_mode=mode,
                    )
                    try:
                        res = synthesize_maya2(req, cfg)
                    except BackendError as exc:
                        raise gr.Error(str(exc))

                    extra = f" · {res.notes}" if res.notes else ""
                    return (
                        res.audio_path,
                        f"✅ Generated **{res.duration_s:.1f}s** cloned speech in **{res.elapsed_s:.1f}s** "
                        f"(`{ref_lang}` → `{target_lang}`, seed `{res.seed}`, backend `{res.backend}`{extra}).",
                    )

                clone_run_btn.click(
                    fn=on_clone_generate,
                    inputs=[
                        clone_ref_audio,
                        clone_ref_text,
                        clone_ref_lang,
                        clone_mode_dd,
                        clone_target_text,
                        clone_target_lang,
                        clone_normalize,
                        clone_seed,
                        clone_randomize,
                        *backend_inputs,
                    ],
                    outputs=[clone_audio_out, clone_meta_md],
                )

            # ============================================================== TAB 3: VOICE DESIGN STUDIO
            with gr.Tab("🎨 Voice Design Studio"):
                gr.Markdown(
                    "**Maya 2 Native** natively conditions on named personas or reference audio/ECAPA embeddings rather than `<description>` text prompts. "
                    "Use **Local Voice Designer & Sculptor** to invent, pitch/tone-sculpt, and blend custom voices **100% locally** on your Maya 2 model, "
                    "or use the **Maya 1 → Maya 2 Bridge** for prompt-based voice descriptions."
                )

                with gr.Tabs():
                    # ---- Sub-tab A: 100% Local Maya 2 Voice Designer & Sculptor
                    with gr.Tab("🎛️ Local Voice Designer & Sculptor (Maya 2)"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                sc_seed_mode = gr.Radio(
                                    choices=[
                                        ("Generate Seed with Local Maya 2 Persona(s)", "maya2_persona"),
                                        ("Use Cached Official Sample / Uploaded Clip", "sample_or_upload"),
                                    ],
                                    value="maya2_persona",
                                    label="Voice Blueprint Source",
                                )
                                with gr.Row():
                                    sc_persona_a = gr.Dropdown(
                                        choices=[v.name for v in BUILTIN_VOICES],
                                        value="Vaidehi",
                                        label="Primary Base Persona",
                                    )
                                    sc_persona_b = gr.Dropdown(
                                        choices=["(none)"] + [v.name for v in BUILTIN_VOICES],
                                        value="(none)",
                                        label="Blend Persona B (Optional)",
                                    )
                                sc_base_lang = gr.Dropdown(
                                    choices=LANGUAGE_CHOICES,
                                    value="en",
                                    label="Seed Language",
                                    filterable=True,
                                )
                                sc_custom_clip = gr.Audio(
                                    label="Optional Uploaded / Recorded Seed Clip",
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    format="wav",
                                )
                                sc_custom_text = gr.Textbox(
                                    label="Seed Anchor Phrase / Transcript",
                                    value="Hello! This is a clean reference voice sample designed for natural multilingual speech.",
                                    lines=2,
                                )
                                with gr.Row():
                                    sc_pitch = gr.Slider(
                                        -4.0, 4.0, value=0.0, step=0.25, label="Pitch Shift (semitones)"
                                    )
                                    sc_tempo = gr.Slider(
                                        0.85, 1.15, value=1.0, step=0.02, label="Tempo Rate"
                                    )
                                    sc_warmth = gr.Slider(
                                        -1.0, 1.0, value=0.0, step=0.1, label="Warmth (-) / Brightness (+)"
                                    )
                                sc_blend_weight = gr.Slider(
                                    0.0,
                                    0.65,
                                    value=0.35,
                                    step=0.05,
                                    label="Persona B Blend Weight (when Persona B is chosen)",
                                )
                                sc_preview_btn = gr.Button(
                                    "1️⃣ Mint & Sculpt Voice Blueprint", variant="primary"
                                )

                            with gr.Column(scale=1):
                                sc_ref_audio_out = gr.Audio(
                                    label="Designed Voice Reference Blueprint", type="filepath"
                                )
                                sc_ref_status = gr.Markdown(
                                    "Configure persona(s) and vocal traits, then click **Mint & Sculpt Voice Blueprint**."
                                )
                                sc_target_lang = gr.Dropdown(
                                    choices=LANGUAGE_CHOICES,
                                    value="hi",
                                    label="Maya 2 Target Language",
                                    filterable=True,
                                )
                                sc_target_text = gr.Textbox(
                                    label="Maya 2 Target Text",
                                    value="नमस्ते! यह मेरी नई डिज़ाइन की गई कस्टम आवाज़ है।",
                                    lines=3,
                                )
                                sc_synth_btn = gr.Button(
                                    "2️⃣ Synthesize Designed Voice in Maya 2", variant="primary"
                                )
                                sc_final_audio = gr.Audio(
                                    label="Maya 2 Designed Voice Output (24 kHz mono)",
                                    type="filepath",
                                )
                                sc_final_status = gr.Markdown("")

                                with gr.Row():
                                    sc_save_name = gr.Textbox(
                                        label="Save Designed Voice As",
                                        placeholder="e.g., warm_vaidehi_custom",
                                    )
                                    sc_save_btn = gr.Button("💾 Save to Voice Library")
                                sc_save_status = gr.Markdown("")

                        def on_sc_mode_or_lang(seed_mode: str, lang_code: str):
                            if seed_mode == "sample_or_upload":
                                return LANGUAGE_BY_CODE[lang_code].sample_text
                            if lang_code == "en":
                                return "Hello! This is a clean reference voice sample designed for natural multilingual speech."
                            return LANGUAGE_BY_CODE[lang_code].sample_text

                        sc_seed_mode.change(
                            fn=on_sc_mode_or_lang,
                            inputs=[sc_seed_mode, sc_base_lang],
                            outputs=[sc_custom_text],
                        )
                        sc_base_lang.change(
                            fn=on_sc_mode_or_lang,
                            inputs=[sc_seed_mode, sc_base_lang],
                            outputs=[sc_custom_text],
                        )

                        def on_sculpt_preview(
                            seed_mode: str,
                            persona_a: str,
                            persona_b: str,
                            base_lang: str,
                            custom_clip: str | None,
                            seed_text: str,
                            pitch: float,
                            tempo: float,
                            warmth: float,
                            blend_weight: float,
                            backend: str,
                            precision: str,
                            models_dir: str,
                            hf_token: str,
                            server_url: str,
                            maya_key: str,
                            cloud_model: str,
                        ):
                            try:
                                cfg = _make_backend_cfg(
                                    backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model
                                )
                                ref_txt = (seed_text or "").strip()
                                if not ref_txt:
                                    ref_txt = LANGUAGE_BY_CODE[base_lang].sample_text

                                sec_path = None
                                used_blend_weight = 0.0
                                if custom_clip:
                                    primary_path = Path(custom_clip)
                                    ref_l = base_lang
                                elif seed_mode == "maya2_persona":
                                    res_a = synthesize_maya2(
                                        GenerationRequest(
                                            text=ref_txt,
                                            lang=base_lang,
                                            voice=persona_a,
                                            normalize=True,
                                            seed=101,
                                            randomize_seed=False,
                                        ),
                                        cfg,
                                    )
                                    primary_path = Path(res_a.audio_path)
                                    ref_l = base_lang
                                    if persona_b and persona_b != "(none)" and persona_b != persona_a and blend_weight > 0.02:
                                        res_b = synthesize_maya2(
                                            GenerationRequest(
                                                text=ref_txt,
                                                lang=base_lang,
                                                voice=persona_b,
                                                normalize=True,
                                                seed=202,
                                                randomize_seed=False,
                                            ),
                                            cfg,
                                        )
                                        sec_path = Path(res_b.audio_path)
                                        used_blend_weight = float(blend_weight)
                                        ref_txt = f"{ref_txt} {ref_txt}"
                                else:
                                    primary_path, official_txt, ref_l = fetch_official_sample(base_lang)
                                    ref_txt = seed_text.strip() or official_txt

                                sculpted_path, dur = sculpt_voice_reference(
                                    primary_path,
                                    pitch_semitones=float(pitch),
                                    tempo_rate=float(tempo),
                                    warmth_brightness=float(warmth),
                                    secondary_audio_path=sec_path,
                                    secondary_weight=used_blend_weight,
                                )
                                blend_desc = (
                                    f" + `{persona_b}` ({int(used_blend_weight * 100)}%)"
                                    if sec_path is not None
                                    else ""
                                )
                                status = (
                                    f"✅ Voice blueprint ready ({dur:.2f}s) from `{persona_a}`{blend_desc} — "
                                    f"pitch `{pitch:+.2f} st`, tempo `{tempo:.2f}x`, warmth `{warmth:+.1f}`."
                                )
                                return str(sculpted_path), status, str(sculpted_path), ref_txt, ref_l
                            except Exception as exc:
                                raise gr.Error(f"Voice design failed: {exc}")

                        sc_preview_btn.click(
                            fn=on_sculpt_preview,
                            inputs=[
                                sc_seed_mode,
                                sc_persona_a,
                                sc_persona_b,
                                sc_base_lang,
                                sc_custom_clip,
                                sc_custom_text,
                                sc_pitch,
                                sc_tempo,
                                sc_warmth,
                                sc_blend_weight,
                                *backend_inputs,
                            ],
                            outputs=[
                                sc_ref_audio_out,
                                sc_ref_status,
                                sculpted_ref_state,
                                sculpted_transcript_state,
                                sculpted_lang_state,
                            ],
                        )

                        def on_sculpt_synth(
                            sculpted_clip: str | None,
                            ref_txt: str,
                            ref_lang: str,
                            target_lang: str,
                            target_text: str,
                            backend: str,
                            precision: str,
                            models_dir: str,
                            hf_token: str,
                            server_url: str,
                            maya_key: str,
                            cloud_model: str,
                        ):
                            if not sculpted_clip:
                                raise gr.Error("Click '1. Mint & Sculpt Voice Blueprint' first.")
                            cfg = _make_backend_cfg(
                                backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model
                            )
                            req = GenerationRequest(
                                text=target_text,
                                lang=target_lang,
                                voice="sculpted",
                                normalize=True,
                                seed=0,
                                randomize_seed=True,
                                ref_audio=sculpted_clip,
                                ref_text=ref_txt,
                                ref_lang=ref_lang,
                                clone_mode="icl_spk",
                            )
                            try:
                                res = synthesize_maya2(req, cfg)
                            except BackendError as exc:
                                raise gr.Error(str(exc))
                            return (
                                res.audio_path,
                                f"✅ Synthesized **{res.duration_s:.1f}s** designed voice in `{target_lang}` (`{res.backend}`).",
                            )

                        sc_synth_btn.click(
                            fn=on_sculpt_synth,
                            inputs=[
                                sculpted_ref_state,
                                sculpted_transcript_state,
                                sculpted_lang_state,
                                sc_target_lang,
                                sc_target_text,
                                *backend_inputs,
                            ],
                            outputs=[sc_final_audio, sc_final_status],
                        )

                    # ---- Sub-tab B: Maya 1 -> Maya 2 Bridge (Optional Remote Prompt Design)
                    with gr.Tab("✨ Prompt-Based Voice Design (Maya 1 → Maya 2 Bridge)"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                design_preset_dd = gr.Dropdown(
                                    choices=list(DESIGN_PRESETS.keys()),
                                    value="Warm Indian Narrator (Female)",
                                    label="Design Preset Template",
                                )
                                with gr.Row():
                                    d_gender = gr.Dropdown(
                                        ["Female", "Male"], value="Female", label="Gender"
                                    )
                                    d_age = gr.Dropdown(
                                        ["20s", "30s", "40s", "50s"], value="30s", label="Age"
                                    )
                                    d_accent = gr.Dropdown(
                                        ["indian", "american", "british", "neutral"],
                                        value="indian",
                                        label="Accent",
                                        allow_custom_value=True,
                                    )
                                with gr.Row():
                                    d_pitch = gr.Dropdown(
                                        ["Low", "Normal", "High"], value="Normal", label="Pitch"
                                    )
                                    d_timbre = gr.Dropdown(
                                        ["warm", "clear", "bright", "deep", "raspy", "throaty", "velvety"],
                                        value="warm",
                                        label="Timbre",
                                        allow_custom_value=True,
                                    )
                                    d_pacing = gr.Dropdown(
                                        ["slow", "conversational", "brisk"],
                                        value="conversational",
                                        label="Pacing",
                                    )
                                with gr.Row():
                                    d_tone = gr.Dropdown(
                                        ["calm", "cheerful", "professional", "neutral", "sarcastic", "empathetic"],
                                        value="calm",
                                        label="Tone",
                                        allow_custom_value=True,
                                    )
                                    d_intensity = gr.Dropdown(
                                        ["low", "medium", "high"], value="medium", label="Intensity"
                                    )
                                    d_role = gr.Textbox(
                                        value="audiobook narrator", label="Role / Domain"
                                    )

                                d_description = gr.Textbox(
                                    label="Composed Voice Description (Editable)",
                                    value=compose_design_description(
                                        gender="Female",
                                        age="30s",
                                        accent="indian",
                                        pitch="Normal",
                                        timbre="warm",
                                        pacing="conversational",
                                        tone="calm",
                                        intensity="medium",
                                        role="audiobook narrator",
                                    ),
                                    lines=3,
                                )
                                d_anchor_text = gr.Textbox(
                                    label="Step 1 Anchor Phrase (spoken in English via Maya 1 Space)",
                                    value=DESIGN_PRESETS["Warm Indian Narrator (Female)"]["anchor_text"],
                                    lines=2,
                                )
                                with gr.Row():
                                    d_temp = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Maya 1 Temp")
                                    d_max_tok = gr.Slider(
                                        400, 2048, value=1200, step=100, label="Max Tokens"
                                    )
                                d_step1_btn = gr.Button(
                                    "1️⃣ Design Voice Prototype (Maya 1 Space)", variant="secondary"
                                )

                            with gr.Column(scale=1):
                                d_proto_audio = gr.Audio(
                                    label="Designed Voice Prototype Clip (or upload any reference)",
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                )
                                d_proto_status = gr.Markdown(
                                    "Generate a prototype via Maya 1 Space or upload a clip, then speak in Step 2 using your local Maya 2 model."
                                )

                                gr.Markdown("### 2️⃣ Speak in Any Maya 2 Language with Designed Voice")
                                d_target_lang = gr.Dropdown(
                                    choices=LANGUAGE_CHOICES,
                                    value="hi",
                                    label="Maya 2 Target Language",
                                    filterable=True,
                                )
                                d_target_text = gr.Textbox(
                                    label="Maya 2 Target Text",
                                    value="नमस्ते! यह मेरी नई डिज़ाइन की गई आवाज़ है, जो अब हिंदी में बात कर रही है।",
                                    lines=3,
                                )
                                d_step2_btn = gr.Button(
                                    "2️⃣ Synthesize with Local Maya 2", variant="primary"
                                )
                                d_final_audio = gr.Audio(
                                    label="Maya 2 Designed Voice Output (24 kHz mono)",
                                    type="filepath",
                                )
                                d_final_status = gr.Markdown("")

                                with gr.Row():
                                    d_save_name = gr.Textbox(
                                        label="Save Designed Voice As",
                                        placeholder="e.g., warm_indian_narrator",
                                    )
                                    d_save_btn = gr.Button("💾 Save to Voice Library")
                                d_save_status = gr.Markdown("")

                        def on_design_preset(preset_name: str):
                            p = DESIGN_PRESETS[preset_name]
                            desc = compose_design_description(
                                gender=p["gender"],
                                age=p["age"],
                                accent=p["accent"],
                                pitch=p["pitch"],
                                timbre=p["timbre"],
                                pacing=p["pacing"],
                                tone=p["tone"],
                                intensity=p["intensity"],
                                role=p["role"],
                            )
                            return (
                                p["gender"],
                                p["age"],
                                p["accent"],
                                p["pitch"],
                                p["timbre"],
                                p["pacing"],
                                p["tone"],
                                p["intensity"],
                                p["role"],
                                desc,
                                p["anchor_text"],
                            )

                        design_preset_dd.change(
                            fn=on_design_preset,
                            inputs=[design_preset_dd],
                            outputs=[
                                d_gender,
                                d_age,
                                d_accent,
                                d_pitch,
                                d_timbre,
                                d_pacing,
                                d_tone,
                                d_intensity,
                                d_role,
                                d_description,
                                d_anchor_text,
                            ],
                        )

                        trait_inputs = [
                            d_gender,
                            d_age,
                            d_accent,
                            d_pitch,
                            d_timbre,
                            d_pacing,
                            d_tone,
                            d_intensity,
                            d_role,
                        ]

                        def on_trait_change(g, a, ac, pi, ti, pa, to, inten, ro):
                            return compose_design_description(
                                gender=g,
                                age=a,
                                accent=ac,
                                pitch=pi,
                                timbre=ti,
                                pacing=pa,
                                tone=to,
                                intensity=inten,
                                role=ro,
                            )

                        for comp in trait_inputs:
                            comp.change(
                                fn=on_trait_change, inputs=trait_inputs, outputs=[d_description]
                            )

                        def on_design_step1(
                            preset_name: str,
                            description: str,
                            anchor_text: str,
                            temp: float,
                            max_tok: float,
                            hf_token: str,
                        ):
                            try:
                                maya1_preset = (
                                    preset_name
                                    if preset_name in ("Male American", "Female British", "Robot", "Singer")
                                    else "Female British"
                                )
                                wav_path, status = design_voice_via_maya1(
                                    description=description,
                                    anchor_text=anchor_text,
                                    preset_name=maya1_preset,
                                    temperature=float(temp),
                                    max_tokens=int(max_tok),
                                    hf_token=hf_token,
                                )
                                return wav_path, f"✅ {status}", wav_path, anchor_text.strip()
                            except BackendError as exc:
                                raise gr.Error(str(exc))

                        d_step1_btn.click(
                            fn=on_design_step1,
                            inputs=[
                                design_preset_dd,
                                d_description,
                                d_anchor_text,
                                d_temp,
                                d_max_tok,
                                hf_token_tb,
                            ],
                            outputs=[
                                d_proto_audio,
                                d_proto_status,
                                maya1_clip_state,
                                maya1_transcript_state,
                            ],
                        )

                        def on_design_step2(
                            proto_clip: str | None,
                            proto_text: str,
                            target_lang: str,
                            target_text: str,
                            backend: str,
                            precision: str,
                            models_dir: str,
                            hf_token: str,
                            server_url: str,
                            maya_key: str,
                            cloud_model: str,
                        ):
                            if not proto_clip:
                                raise gr.Error("Run Step 1 first (or upload a prototype clip) before synthesizing.")
                            cfg = _make_backend_cfg(
                                backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model
                            )
                            req = GenerationRequest(
                                text=target_text,
                                lang=target_lang,
                                voice="designed",
                                normalize=True,
                                seed=0,
                                randomize_seed=True,
                                ref_audio=proto_clip,
                                ref_text=proto_text,
                                ref_lang="en",
                                clone_mode="icl_spk",
                            )
                            try:
                                res = synthesize_maya2(req, cfg)
                            except BackendError as exc:
                                raise gr.Error(str(exc))
                            return (
                                res.audio_path,
                                f"✅ Synthesized **{res.duration_s:.1f}s** in `{target_lang}` using your designed voice (`{res.backend}`).",
                            )

                        d_step2_btn.click(
                            fn=on_design_step2,
                            inputs=[
                                d_proto_audio,
                                d_anchor_text,
                                d_target_lang,
                                d_target_text,
                                *backend_inputs,
                            ],
                            outputs=[d_final_audio, d_final_status],
                        )

            # ============================================================== TAB 4: VOICE LIBRARY & PERSONAS
            with gr.Tab("📚 Voice Library & Personas"):
                gr.Markdown("### Built-in Maya 2 Personas (20 voices)")
                with gr.Row():
                    filter_gender = gr.Dropdown(
                        ["all", "female", "male"], value="all", label="Gender"
                    )
                    filter_age = gr.Dropdown(
                        ["all", "young", "adult", "mature"], value="all", label="Age Band"
                    )
                    filter_lang = gr.Dropdown(
                        [("All languages", "all")] + LANGUAGE_CHOICES,
                        value="all",
                        label="Trained Language (>= 1h)",
                    )
                persona_df = gr.Dataframe(
                    headers=["Voice", "Gender", "Age", "Trained Languages", "Description"],
                    value=filter_builtin_voices(),
                    datatype=["str", "str", "str", "str", "str"],
                    type="array",
                    interactive=False,
                    show_search="filter",
                )

                for fcomp in (filter_gender, filter_age, filter_lang):
                    fcomp.change(
                        fn=lambda g, a, l: filter_builtin_voices(g, a, l),
                        inputs=[filter_gender, filter_age, filter_lang],
                        outputs=[persona_df],
                    )

                gr.Markdown("### Saved Custom Voices (Cloned & Designed)")
                saved_voices_df = gr.Dataframe(
                    headers=["Name", "Source", "Ref Lang", "Mode", "Transcript", "Notes / Design"],
                    value=_saved_voice_rows(voice_store),
                    datatype=["str", "str", "str", "str", "str", "str"],
                    type="array",
                    interactive=False,
                )
                with gr.Row():
                    lib_voice_dd = gr.Dropdown(
                        choices=_saved_voice_names(voice_store),
                        label="Select Saved Voice",
                        scale=2,
                    )
                    lib_preview_btn = gr.Button("Preview Clip", scale=1)
                    lib_push_server_btn = gr.Button("Push to maya3.server", scale=1)
                    lib_delete_btn = gr.Button("Delete Voice", variant="stop", scale=1)
                lib_audio_preview = gr.Audio(label="Saved Reference Clip", type="filepath")
                lib_status_md = gr.Markdown("")

                def _refresh_voice_ui(status_msg: str = ""):
                    return (
                        gr.Dropdown(choices=_builtin_voice_choices(voice_store)),
                        _saved_voice_rows(voice_store),
                        gr.Dropdown(choices=_saved_voice_names(voice_store)),
                        status_msg,
                    )

                def on_save_clone(
                    name: str,
                    ref_audio: str | None,
                    ref_text: str,
                    ref_lang: str,
                    mode: str,
                    notes: str,
                ):
                    if not ref_audio:
                        raise gr.Error("Upload or record a reference clip before saving.")
                    if mode != "spk" and not (ref_text or "").strip():
                        raise gr.Error("Provide the reference transcript (required unless mode is 'spk').")
                    stats = prepare_reference_clip(ref_audio)
                    try:
                        prof = voice_store.save_profile(
                            name=name,
                            source_type="cloned",
                            ref_audio_path=stats.path,
                            ref_text=ref_text,
                            ref_lang=ref_lang,
                            mode=mode,
                            notes=notes,
                        )
                    finally:
                        stats.path.unlink(missing_ok=True)
                    tts_dd, rows, lib_dd, _ = _refresh_voice_ui()
                    return (
                        tts_dd,
                        rows,
                        lib_dd,
                        f"✅ Saved cloned voice **'{prof.name}'**! Select `★ {prof.name}` in Tab 1.",
                    )

                save_clone_btn.click(
                    fn=on_save_clone,
                    inputs=[
                        save_clone_name,
                        clone_ref_audio,
                        clone_ref_text,
                        clone_ref_lang,
                        clone_mode_dd,
                        save_clone_notes,
                    ],
                    outputs=[tts_voice, saved_voices_df, lib_voice_dd, save_clone_status],
                )

                def on_save_maya1_design(
                    name: str,
                    proto_audio: str | None,
                    anchor_text: str,
                    description: str,
                ):
                    if not proto_audio:
                        raise gr.Error("Generate the Maya 1 prototype clip in Step 1 before saving.")
                    stats = prepare_reference_clip(proto_audio)
                    try:
                        prof = voice_store.save_profile(
                            name=name,
                            source_type="designed_maya1",
                            ref_audio_path=stats.path,
                            ref_text=anchor_text,
                            ref_lang="en",
                            mode="icl_spk",
                            design_prompt=description,
                            notes="Designed via Maya 1 -> Maya 2 bridge",
                        )
                    finally:
                        stats.path.unlink(missing_ok=True)
                    tts_dd, rows, lib_dd, _ = _refresh_voice_ui()
                    return (
                        tts_dd,
                        rows,
                        lib_dd,
                        f"✅ Saved designed voice **'{prof.name}'**! Select `★ {prof.name}` in Tab 1.",
                    )

                d_save_btn.click(
                    fn=on_save_maya1_design,
                    inputs=[d_save_name, d_proto_audio, d_anchor_text, d_description],
                    outputs=[tts_voice, saved_voices_df, lib_voice_dd, d_save_status],
                )

                def on_save_sculpt_design(
                    name: str,
                    sculpt_audio: str | None,
                    sculpt_text: str,
                    sculpt_lang: str,
                    pitch: float,
                    tempo: float,
                    warmth: float,
                ):
                    if not sculpt_audio:
                        raise gr.Error("Mint & sculpt the voice blueprint in Step 1 before saving.")
                    prof = voice_store.save_profile(
                        name=name,
                        source_type="designed_sculpt",
                        ref_audio_path=sculpt_audio,
                        ref_text=sculpt_text,
                        ref_lang=sculpt_lang,
                        mode="icl_spk",
                        notes=f"pitch={pitch:+.2f}st, tempo={tempo:.2f}x, warmth={warmth:+.1f}",
                    )
                    tts_dd, rows, lib_dd, _ = _refresh_voice_ui()
                    return (
                        tts_dd,
                        rows,
                        lib_dd,
                        f"✅ Saved designed voice **'{prof.name}'**! Select `★ {prof.name}` in Tab 1.",
                    )

                sc_save_btn.click(
                    fn=on_save_sculpt_design,
                    inputs=[
                        sc_save_name,
                        sculpted_ref_state,
                        sculpted_transcript_state,
                        sculpted_lang_state,
                        sc_pitch,
                        sc_tempo,
                        sc_warmth,
                    ],
                    outputs=[tts_voice, saved_voices_df, lib_voice_dd, sc_save_status],
                )

                def on_lib_preview(name: str | None):
                    if not name:
                        raise gr.Error("Select a saved voice first.")
                    prof = voice_store.get(name)
                    if prof is None:
                        raise gr.Error(f"Saved voice '{name}' not found.")
                    return (
                        prof.ref_audio_path,
                        f"🎧 **{prof.name}** (`{prof.source_type}`, `{prof.ref_lang}`) — *{prof.ref_text}*",
                    )

                lib_preview_btn.click(
                    fn=on_lib_preview,
                    inputs=[lib_voice_dd],
                    outputs=[lib_audio_preview, lib_status_md],
                )

                def on_lib_push_server(
                    name: str | None,
                    backend: str,
                    precision: str,
                    models_dir: str,
                    hf_token: str,
                    server_url: str,
                    maya_key: str,
                    cloud_model: str,
                ):
                    if not name:
                        raise gr.Error("Select a saved voice first.")
                    prof = voice_store.get(name)
                    if prof is None:
                        raise gr.Error(f"Saved voice '{name}' not found.")
                    cfg = _make_backend_cfg(
                        backend, precision, models_dir, hf_token, server_url, maya_key, cloud_model
                    )
                    try:
                        return f"✅ {register_voice_on_server(cfg, voice_id=prof.name, ref_audio_path=prof.ref_audio_path, ref_text=prof.ref_text, ref_lang=prof.ref_lang, mode=prof.mode)}"
                    except BackendError as exc:
                        raise gr.Error(str(exc))

                lib_push_server_btn.click(
                    fn=on_lib_push_server,
                    inputs=[lib_voice_dd, *backend_inputs],
                    outputs=[lib_status_md],
                )

                def on_lib_delete(name: str | None):
                    if not name:
                        raise gr.Error("Select a saved voice to delete.")
                    deleted = voice_store.delete(name)
                    tts_dd, rows, lib_dd, _ = _refresh_voice_ui()
                    msg = f"🗑️ Deleted '{name}'." if deleted else f"⚠️ '{name}' not found."
                    return tts_dd, rows, lib_dd, msg

                lib_delete_btn.click(
                    fn=on_lib_delete,
                    inputs=[lib_voice_dd],
                    outputs=[tts_voice, saved_voices_df, lib_voice_dd, lib_status_md],
                )

                with gr.Accordion("📖 Quick Local Python Reference (Apple Silicon MLX)", open=False):
                    gr.Markdown(
                        """
```python
import soundfile as sf
from maya_mac import MayaTTS

# Uses cached ~/.cache/huggingface/hub/models--maya-research--maya2-native
tts = MayaTTS.from_pretrained(backbone_bits=8, depth_bits=8)
wav = tts.generate("नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।", lang="hi", voice="Ahana", normalize=True)
sf.write("out.wav", wav, 24000)

# Zero-shot cross-lingual voice cloning
tts.register_voice("my_voice", ref_audio="ref.wav", ref_text="...", lang="hi", mode="icl_spk")
cloned = tts.generate("வணக்கம், எப்படி இருக்கிறீர்கள்?", lang="ta", voice="my_voice", normalize=True)
sf.write("cloned_ta.wav", cloned, 24000)
tts.close()
```
                        """
                    )

    allowed = [
        str(voice_store.root),
        str(default_cache_dir()),
        tempfile.gettempdir(),
    ]
    launch_kwargs: dict[str, Any] = {
        "theme": gr.themes.Soft(primary_hue="indigo", secondary_hue="emerald"),
        "css": CUSTOM_CSS,
        "allowed_paths": allowed,
    }
    return demo, launch_kwargs


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for launching Maya 2 Studio."""
    parser = argparse.ArgumentParser(description="Maya 2 Studio Gradio UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument(
        "--backend",
        default="local",
        choices=[code for _, code in BACKEND_CHOICES],
        help="Default Maya 2 inference backend (default: local)",
    )
    parser.add_argument(
        "--precision",
        default="mlx_8bit",
        choices=[code for _, code in LOCAL_PRECISION_CHOICES],
        help="Local engine/quantization mode (default: mlx_8bit)",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8080",
        help="maya3.server HTTP base URL",
    )
    args = parser.parse_args(argv)

    demo, launch_kwargs = create_demo(
        default_backend=args.backend,
        default_precision=args.precision,
        default_server_url=args.server_url,
    )
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        **launch_kwargs,
    )
