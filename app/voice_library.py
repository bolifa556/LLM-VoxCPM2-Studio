from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from .config import APP_ROOT, VOICE_SYSTEM_ROOT, VOICE_USER_ROOT
from .models import VoiceRecord

VOICE_MANIFEST = "voice.json"
REFERENCE_TEXT_FILE = "reference.txt"
REFERENCE_AUDIO_CANDIDATES = ("reference.wav", "reference.mp3", "reference.flac", "reference.m4a", "reference.ogg")

# The project no longer bootstraps system voices or generates reference audio automatically.
BOOTSTRAP_VOICES: list[dict] = []


def _manifest_path(root: Path, voice_id: str) -> Path:
    return root / voice_id / VOICE_MANIFEST


def _reference_text_path(voice_dir: Path) -> Path:
    return voice_dir / REFERENCE_TEXT_FILE


def _find_reference_audio_path(voice_dir: Path) -> Optional[Path]:
    for candidate in REFERENCE_AUDIO_CANDIDATES:
        path = voice_dir / candidate
        if path.exists():
            return path
    return None


def _write_manifest(root: Path, record: VoiceRecord) -> None:
    voice_dir = root / record.id
    voice_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "id": record.id,
        "name": record.name,
        "scope": record.scope,
        "metadata": record.metadata,
    }
    (_manifest_path(root, record.id)).write_text(record.__class__.model_validate(manifest_payload).model_dump_json(indent=2), encoding="utf-8")
    if record.reference_text:
        _reference_text_path(voice_dir).write_text(record.reference_text, encoding="utf-8")


def ensure_voice_library() -> None:
    VOICE_SYSTEM_ROOT.mkdir(parents=True, exist_ok=True)
    VOICE_USER_ROOT.mkdir(parents=True, exist_ok=True)


def bootstrap_voice_audio(tts_adapter) -> None:
    return None


def list_voices() -> List[VoiceRecord]:
    ensure_voice_library()
    voices: List[VoiceRecord] = []
    for root, scope in ((VOICE_SYSTEM_ROOT, "system"), (VOICE_USER_ROOT, "user")):
        for voice_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            try:
                manifest_path = voice_dir / VOICE_MANIFEST
                payload = {}
                if manifest_path.exists():
                    payload = VoiceRecord.model_validate_json(manifest_path.read_text(encoding="utf-8")).model_dump(mode="json")
                reference_text = _reference_text_path(voice_dir).read_text(encoding="utf-8").strip() if _reference_text_path(voice_dir).exists() else ""
                reference_audio = _find_reference_audio_path(voice_dir)
                record = VoiceRecord(
                    id=str(payload.get("id") or voice_dir.name),
                    name=str(payload.get("name") or voice_dir.name),
                    scope=scope,
                    reference_text=reference_text,
                    reference_audio_path=str(reference_audio.relative_to(APP_ROOT)).replace("\\", "/") if reference_audio else None,
                    metadata=dict(payload.get("metadata") or {}),
                )
                voices.append(record)
            except Exception:
                continue
    return voices


def get_voice(voice_id: Optional[str]) -> Optional[VoiceRecord]:
    if not voice_id:
        return None
    for voice in list_voices():
        if voice.id == voice_id:
            return voice
    return None


def save_generated_voice(
    name: str,
    audio_source: Path,
    reference_text: str,
    source: str,
) -> VoiceRecord:
    ensure_voice_library()
    voice_id = f"{name.strip().lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
    voice_dir = VOICE_USER_ROOT / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    target_audio = voice_dir / "reference.wav"
    shutil.copy2(audio_source, target_audio)
    record = VoiceRecord(
        id=voice_id,
        name=name.strip(),
        scope="user",
        reference_text=reference_text.strip(),
        reference_audio_path=str(target_audio.relative_to(APP_ROOT)).replace("\\", "/"),
        metadata={"source": source},
    )
    _write_manifest(VOICE_USER_ROOT, record)
    return record
