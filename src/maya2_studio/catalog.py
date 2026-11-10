"""Authoritative Maya 2 metadata: languages, personas, emotion tags, and presets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

HF_MODEL_REPO: Final[str] = "maya-research/maya2-native"
HF_SAMPLES_BASE: Final[str] = (
    "https://huggingface.co/maya-research/maya2-native/resolve/main/samples"
)

SAMPLE_RATE: Final[int] = 24000


@dataclass(frozen=True, slots=True)
class LanguageInfo:
    """Metadata and showcase example for one supported Maya 2 language."""

    code: str
    name: str
    native_name: str
    sample_text: str
    sample_voice: str
    sample_filename: str


LANGUAGES: Final[tuple[LanguageInfo, ...]] = (
    LanguageInfo(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        sample_text="क्या आपका वर्तमान पता 45, एमजी रोड, अंधेरी ईस्ट, मुंबई 400069 है? यदि हाँ, तो 1 दबाएं।",
        sample_voice="Ritvik",
        sample_filename="hi_ritvik.wav",
    ),
    LanguageInfo(
        code="en",
        name="English",
        native_name="English",
        sample_text="The EMI for your home loan is due by 12-10-24; please ensure your account has sufficient funds.",
        sample_voice="Siddhant",
        sample_filename="en_siddhant.wav",
    ),
    LanguageInfo(
        code="bn",
        name="Bengali",
        native_name="বাংলা",
        sample_text="12/08/2023 তারিখে আপনার 1,500 টাকার refund process হয়ে গেছে, status update চেক করুন।",
        sample_voice="Jahnavi",
        sample_filename="bn_jahnavi.wav",
    ),
    LanguageInfo(
        code="ta",
        name="Tamil",
        native_name="தமிழ்",
        sample_text="கடந்த நிதியாண்டில் வங்கியின் மொத்த லாபம் ₹3.5 கோடி, இதில் ₹45 லட்சம் முதலீட்டாளர்களுக்கு வழங்கப்பட்டது.",
        sample_voice="Abhay",
        sample_filename="ta_abhay.wav",
    ),
    LanguageInfo(
        code="te",
        name="Telugu",
        native_name="తెలుగు",
        sample_text="సేవింగ్స్ తో పోలిస్తే ఫిక్స్డ్ డిపాజిట్ వడ్డీ నిష్పత్తి 3:2 గా ఉండి, 7.25% ఇస్తుంది.",
        sample_voice="Nivedita",
        sample_filename="te_nivedita.wav",
    ),
    LanguageInfo(
        code="mr",
        name="Marathi",
        native_name="मराठी",
        sample_text="मला 98765 43210 या नंबरवर OTP आला नाही, कृपया पुन्हा resend करा.",
        sample_voice="Charvi",
        sample_filename="mr_charvi.wav",
    ),
    LanguageInfo(
        code="gu",
        name="Gujarati",
        native_name="ગુજરાતી",
        sample_text="જી બિલકુલ, મને સિસ્ટમમાં દેખાય છે કે તમારું ₹2,500 નું પેમેન્ટ 15મી તારીખે સક્સેસફુલ થઈ ગયું છે.",
        sample_voice="Vaidehi",
        sample_filename="gu_vaidehi.wav",
    ),
    LanguageInfo(
        code="kn",
        name="Kannada",
        native_name="ಕನ್ನಡ",
        sample_text="ಹೌದಾ... ಅದು ನಿಜಕ್ಕೂ ಕಷ್ಟ. ನಿಮ್ಮ ಅಂತಿಮ ಬಿಲ್ ₹45,000 ಆಗಿದೆ, ಅದರಲ್ಲಿ ಇನ್ಶೂರೆನ್ಸ್ ಕವರ್ ಆಗಿದೆಯಾ ಅಂತ ಚೆಕ್ ಮಾಡ್ತೀನಿ.",
        sample_voice="Satvik",
        sample_filename="kn_satvik.wav",
    ),
    LanguageInfo(
        code="ml",
        name="Malayalam",
        native_name="മലയാളം",
        sample_text="റദ്ദാക്കിയ ടിക്കറ്റിന്റെ റീഫണ്ട് തുകയായ ₹1,800 മൂന്ന് ദിവസത്തിനുള്ളിൽ ലഭിക്കും.",
        sample_voice="Anirban",
        sample_filename="ml_anirban.wav",
    ),
    LanguageInfo(
        code="pa",
        name="Punjabi",
        native_name="ਪੰਜਾਬੀ",
        sample_text="NEFT ਰਾਹੀਂ ਫੰਡ ਟ੍ਰਾਂਸਫਰ 00:30 ਤੋਂ 23:30 ਤੱਕ ਉਪਲਬਧ ਹਨ।",
        sample_voice="Malini",
        sample_filename="pa_malini.wav",
    ),
    LanguageInfo(
        code="or",
        name="Odia",
        native_name="ଓଡ଼ିଆ",
        sample_text="ଦୟାକରି ମନେରଖନ୍ତୁ ଯେ EMI ପ୍ରତ୍ୟେକ ମାସର 5 ତାରିଖରେ କଟିବ ଏବଂ ଏହା ଆପଣଙ୍କର 3ୟ କିସ୍ତି।",
        sample_voice="Ahana",
        sample_filename="or_ahana.wav",
    ),
    LanguageInfo(
        code="as",
        name="Assamese",
        native_name="অসমীয়া",
        sample_text="এন ই এফ টি লেনদেনটো সম্পূৰ্ণ হ'বলৈ প্ৰায় 2 ঘণ্টা 30 মিনিট সময় লাগিব পাৰে।",
        sample_voice="Nakul",
        sample_filename="as_nakul.wav",
    ),
)

LANGUAGE_BY_CODE: Final[dict[str, LanguageInfo]] = {lang.code: lang for lang in LANGUAGES}
LANGUAGE_CHOICES: Final[list[tuple[str, str]]] = [
    (f"{lang.name} ({lang.native_name}) [{lang.code}]", lang.code) for lang in LANGUAGES
]


@dataclass(frozen=True, slots=True)
class BuiltinVoice:
    """Built-in Maya 2 Native persona from voices/voices.json."""

    name: str
    gender: str
    age_band: str
    langs: tuple[str, ...]
    description: str


BUILTIN_VOICES: Final[tuple[BuiltinVoice, ...]] = (
    BuiltinVoice(
        "Ahana",
        "female",
        "young",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"),
        "Young female voice, bright, youthful",
    ),
    BuiltinVoice(
        "Vaidehi",
        "female",
        "adult",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"),
        "Adult female voice, calm, gentle",
    ),
    BuiltinVoice(
        "Malini",
        "female",
        "mature",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"),
        "Mature female voice, steady narrator",
    ),
    BuiltinVoice(
        "Charvi",
        "female",
        "young",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"),
        "Young female voice, energetic, upbeat",
    ),
    BuiltinVoice(
        "Jahnavi",
        "female",
        "adult",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "ta", "te"),
        "Adult female voice, warm, friendly",
    ),
    BuiltinVoice(
        "Anirban",
        "male",
        "adult",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"),
        "Adult male voice, steady narrator",
    ),
    BuiltinVoice(
        "Nakul",
        "male",
        "young",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"),
        "Young male voice, bright, youthful",
    ),
    BuiltinVoice(
        "Vasant",
        "male",
        "mature",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "ta", "te"),
        "Mature male voice, steady narrator",
    ),
    BuiltinVoice(
        "Siddhant",
        "male",
        "adult",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"),
        "Adult male voice, clear, professional",
    ),
    BuiltinVoice(
        "Satvik",
        "male",
        "young",
        ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"),
        "Young male voice, warm, friendly",
    ),
    BuiltinVoice(
        "Abhay",
        "male",
        "adult",
        ("en", "hi"),
        "Adult male voice, natural conversational style; trained on English, Hindi",
    ),
    BuiltinVoice(
        "Nivedita",
        "female",
        "adult",
        ("en", "hi"),
        "Adult female voice, natural conversational style; trained on English, Hindi",
    ),
    BuiltinVoice(
        "Ishwari",
        "female",
        "adult",
        ("hi",),
        "Adult female voice, natural conversational style; trained on Hindi",
    ),
    BuiltinVoice(
        "Ritvik",
        "male",
        "adult",
        ("hi",),
        "Adult male voice, natural conversational style; trained on Hindi",
    ),
    BuiltinVoice(
        "Vrinda",
        "female",
        "young",
        ("hi",),
        "Young female voice, natural conversational style; trained on Hindi",
    ),
    BuiltinVoice(
        "Sreekar",
        "male",
        "adult",
        ("en", "hi", "te"),
        "Adult male voice, natural conversational style; trained on English, Hindi, Telugu",
    ),
    BuiltinVoice(
        "Kayal",
        "female",
        "adult",
        ("en", "ta"),
        "Adult female voice, natural conversational style; trained on English, Tamil",
    ),
    BuiltinVoice(
        "Shaunak",
        "male",
        "young",
        ("en", "hi"),
        "Young male voice, natural conversational style; trained on English, Hindi",
    ),
    BuiltinVoice(
        "Akshat",
        "male",
        "adult",
        ("en", "hi"),
        "Adult male voice, natural conversational style; trained on English, Hindi",
    ),
    BuiltinVoice(
        "Tanushree",
        "female",
        "adult",
        ("en", "hi"),
        "Adult female voice, natural conversational style; trained on English, Hindi",
    ),
)

VOICE_BY_NAME: Final[dict[str, BuiltinVoice]] = {v.name: v for v in BUILTIN_VOICES}
CLOUD_VOICES: Final[tuple[str, ...]] = ("Ananya", "Arjun")

EMOTION_TAGS: Final[list[tuple[str, str]]] = [
    ("[excited]", "faster, brighter"),
    ("[whispers]", "quieter, closer"),
    ("[curious]", "rising, questioning"),
    ("[laughs]", "laughter in the line"),
    ("[sighs]", "an audible sigh"),
    ("[sad]", "slower, heavier"),
    ("[nervous]", "hesitant"),
    ("[frustrated]", "tighter, clipped"),
    ("[clears throat]", "throat-clear before the line"),
    ("[coughs]", "cough before the line"),
]

# Maya 1 structured voice design presets and attribute builders.
DESIGN_PRESETS: Final[dict[str, dict[str, str]]] = {
    "Warm Indian Narrator (Female)": {
        "gender": "Female",
        "age": "30s",
        "accent": "indian",
        "pitch": "Normal",
        "timbre": "warm",
        "pacing": "conversational",
        "tone": "calm",
        "intensity": "medium",
        "role": "audiobook narrator",
        "anchor_text": (
            "Welcome back! Today we are exploring how clear, natural storytelling "
            "brings every single detail to life."
        ),
    },
    "Crisp News & IVR Anchor (Male)": {
        "gender": "Male",
        "age": "30s",
        "accent": "indian",
        "pitch": "Normal",
        "timbre": "clear",
        "pacing": "conversational",
        "tone": "professional",
        "intensity": "medium",
        "role": "news presenter",
        "anchor_text": (
            "Good afternoon. Your scheduled verification is complete, and all systems "
            "are running smoothly right now."
        ),
    },
    "Bright Host (Young Female)": {
        "gender": "Female",
        "age": "20s",
        "accent": "american",
        "pitch": "High",
        "timbre": "bright",
        "pacing": "brisk",
        "tone": "cheerful",
        "intensity": "medium",
        "role": "podcast host",
        "anchor_text": (
            "Hey everyone, thanks for tuning in today! We have an amazing lineup "
            "of stories ready to share with you."
        ),
    },
    "Female British Host": {
        "gender": "Female",
        "age": "30s",
        "accent": "british",
        "pitch": "Normal",
        "timbre": "throaty",
        "pacing": "conversational",
        "tone": "sarcastic",
        "intensity": "low",
        "role": "podcast interviewer",
        "anchor_text": (
            "You propose that the key to happiness is simply ignoring all external pressures. "
            "<chuckle> I'm sure it works brilliantly in theory."
        ),
    },
    "Male American Storyteller": {
        "gender": "Male",
        "age": "20s",
        "accent": "american",
        "pitch": "High",
        "timbre": "raspy",
        "pacing": "brisk",
        "tone": "neutral",
        "intensity": "medium",
        "role": "short form narrator",
        "anchor_text": (
            "And of course, the so-called easy shortcut didn't work at all. "
            "What a surprise. <sigh>"
        ),
    },
    "Deep Mature Sage (Male)": {
        "gender": "Male",
        "age": "50s",
        "accent": "british",
        "pitch": "Low",
        "timbre": "deep",
        "pacing": "slow",
        "tone": "calm",
        "intensity": "low",
        "role": "documentary narrator",
        "anchor_text": (
            "Across centuries of history, patient voices have carried wisdom "
            "from one generation to the next."
        ),
    },
}


def compose_design_description(
    *,
    gender: str,
    age: str,
    accent: str,
    pitch: str,
    timbre: str,
    pacing: str,
    tone: str,
    intensity: str,
    role: str,
) -> str:
    """Build a Maya 1 voice-design description string from structured controls."""
    g = gender.lower()
    return (
        f"Realistic {g} voice in the {age} age with a {accent.lower()} accent. "
        f"{pitch} pitch, {timbre.lower()} timbre, {pacing.lower()} pacing, "
        f"{tone.lower()} tone delivery at {intensity.lower()} intensity, "
        f"{role.lower()} role, natural delivery"
    )


def format_persona_summary(voice_name: str) -> str:
    """Return human-readable persona details for a built-in Maya 2 voice."""
    if voice_name in ("(none)", "plain"):
        return "**plain** — Unconditioned baseline voice (no persona prefix or speaker embedding)."
    v = VOICE_BY_NAME.get(voice_name)
    if v is None:
        return f"**{voice_name}**"
    trained = ", ".join(v.langs)
    return (
        f"**{v.name}** ({v.gender}, {v.age_band}) — {v.description}.  \n"
        f"Trained >=1h on: `{trained}` (cross-lingual on all 12 supported languages)."
    )


def filter_builtin_voices(
    gender: str = "all",
    age_band: str = "all",
    lang_code: str = "all",
) -> list[list[str]]:
    """Filter built-in Maya 2 personas into table rows for the directory UI."""
    rows: list[list[str]] = []
    for v in BUILTIN_VOICES:
        if gender != "all" and v.gender != gender:
            continue
        if age_band != "all" and v.age_band != age_band:
            continue
        if lang_code != "all" and lang_code not in v.langs:
            continue
        rows.append([v.name, v.gender, v.age_band, ", ".join(v.langs), v.description])
    return rows


def default_cache_dir() -> Path:
    """Return default local cache directory for downloaded official samples."""
    return Path.home() / ".cache" / "maya2-studio" / "samples"


def fetch_official_sample(lang_code: str, cache_dir: Path | None = None) -> tuple[Path, str, str]:
    """Download (with local cache) one of the 12 official Maya 2 showcase WAV clips.

    Returns (local_wav_path, exact_transcript, lang_code).
    """
    info = LANGUAGE_BY_CODE.get(lang_code)
    if info is None:
        raise ValueError(f"Unsupported language code: {lang_code!r}")
    target_dir = cache_dir or default_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / info.sample_filename
    if not dest.exists() or dest.stat().st_size < 1024:
        url = f"{HF_SAMPLES_BASE}/{info.sample_filename}"
        with httpx.Client(timeout=45.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
    return dest, info.sample_text, info.code

