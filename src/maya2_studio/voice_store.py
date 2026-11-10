"""Local persistent storage for custom cloned and designed Maya 2 voices."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(slots=True)
class VoiceProfile:
    """Saved custom voice reference (cloned or designed)."""

    name: str
    source_type: str  # "cloned" | "designed_maya1" | "designed_sculpt"
    ref_audio_path: str
    ref_text: str
    ref_lang: str
    mode: str = "icl_spk"
    design_prompt: str = ""
    notes: str = ""
    created_at: float = 0.0


def default_store_dir() -> Path:
    """Return directory for persisted custom voice profiles."""
    env_dir = os.environ.get("MAYA2_VOICES_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".cache" / "maya2-studio" / "voices"


def sanitize_voice_id(name: str) -> str:
    """Normalize a user-provided voice name to a Maya-safe identifier."""
    cleaned = _SAFE_NAME.sub("_", (name or "").strip()).strip("._-")
    if not cleaned:
        raise ValueError("Voice name must contain at least one letter or digit.")
    if not cleaned[0].isalnum():
        cleaned = f"v_{cleaned}"
    return cleaned[:64]


class VoiceStore:
    """Filesystem-backed custom voice library."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_store_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "voices.json"

    def _read_raw(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_raw(self, data: dict[str, dict]) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.index_path)

    def list_profiles(self) -> list[VoiceProfile]:
        """Return all saved voice profiles sorted by newest first."""
        raw = self._read_raw()
        profiles: list[VoiceProfile] = []
        for item in raw.values():
            try:
                p = VoiceProfile(**item)
                if Path(p.ref_audio_path).exists():
                    profiles.append(p)
            except TypeError:
                continue
        profiles.sort(key=lambda p: p.created_at, reverse=True)
        return profiles

    def get(self, name: str) -> VoiceProfile | None:
        """Look up a saved voice profile by exact or case-insensitive name."""
        raw = self._read_raw()
        if name in raw:
            p = VoiceProfile(**raw[name])
            return p if Path(p.ref_audio_path).exists() else None
        low = name.lower()
        for k, v in raw.items():
            if k.lower() == low:
                p = VoiceProfile(**v)
                return p if Path(p.ref_audio_path).exists() else None
        return None

    def save_profile(
        self,
        *,
        name: str,
        source_type: str,
        ref_audio_path: str | Path,
        ref_text: str,
        ref_lang: str,
        mode: str = "icl_spk",
        design_prompt: str = "",
        notes: str = "",
    ) -> VoiceProfile:
        """Persist a reference audio clip and voice profile metadata."""
        vid = sanitize_voice_id(name)
        src = Path(ref_audio_path)
        if not src.exists():
            raise FileNotFoundError(f"Reference audio does not exist: {src}")

        dest_wav = self.root / f"{vid}.wav"
        if src.resolve() != dest_wav.resolve():
            shutil.copyfile(src, dest_wav)

        profile = VoiceProfile(
            name=vid,
            source_type=source_type,
            ref_audio_path=str(dest_wav),
            ref_text=(ref_text or "").strip(),
            ref_lang=(ref_lang or "en").strip().lower(),
            mode=mode or "icl_spk",
            design_prompt=(design_prompt or "").strip(),
            notes=(notes or "").strip(),
            created_at=time.time(),
        )
        raw = self._read_raw()
        raw[vid] = asdict(profile)
        self._write_raw(raw)
        return profile

    def delete(self, name: str) -> bool:
        """Remove a saved voice profile and its stored clip."""
        raw = self._read_raw()
        target_key = None
        if name in raw:
            target_key = name
        else:
            low = name.lower()
            for k in raw:
                if k.lower() == low:
                    target_key = k
                    break
        if target_key is None:
            return False
        entry = raw.pop(target_key)
        self._write_raw(raw)
        wav = Path(entry.get("ref_audio_path", ""))
        if wav.exists() and wav.parent.resolve() == self.root.resolve():
            wav.unlink(missing_ok=True)
        return True

