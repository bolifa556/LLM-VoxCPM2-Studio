from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SegmentRequest(BaseModel):
    text: str
    llm_provider_id: Optional[str] = None
    tts_adapter_id: Optional[str] = None
    soul_override: Optional[str] = None
    config_override: Optional[Dict[str, Any]] = None
    selected_voice_id: Optional[str] = None
    reference_text: Optional[str] = None
    control_hint: str = ""
    reference_mode: bool = False


class SegmentItem(BaseModel):
    index: int
    text: str
    control: str = ""
    pause_ms: int = 300
    emotion: str = "neutral"
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None


class GenerateRequest(BaseModel):
    text: str
    segments: List[SegmentItem]
    llm_provider_id: Optional[str] = None
    tts_adapter_id: Optional[str] = None
    config_override: Optional[Dict[str, Any]] = None
    selected_voice_id: Optional[str] = None
    reference_text: Optional[str] = None
    reference_audio_data_url: Optional[str] = None
    control_hint: str = ""
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None


class SaveVoiceRequest(BaseModel):
    name: str
    audio_data_url: str
    reference_text: str
    source: str = "user_saved"


class VoiceRecord(BaseModel):
    id: str
    name: str
    scope: str
    reference_text: str = ""
    reference_audio_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
