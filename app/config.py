from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Literal

from pydantic import BaseModel, Field

APP_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_ROOT / "config" / "app_config.json"
PROMPT_PATH = APP_ROOT / "prompts" / "tts_segmentation_prompt.md"
PROMPT_RUNTIME_PATH = APP_ROOT / "prompts" / "tts_runtime_template.md"
SOUL_PATH = APP_ROOT / "prompts" / "soul.md"
THOUGHTS_PATH = APP_ROOT / "notes" / "project_thoughts.md"
VOICE_ROOT = APP_ROOT / "voice"
VOICE_SYSTEM_ROOT = VOICE_ROOT / "system"
VOICE_USER_ROOT = VOICE_ROOT / "usr"
DATA_ROOT = APP_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "output"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7860


class LLMProviderConfig(BaseModel):
    provider_type: Literal["openai_compatible"] = "openai_compatible"
    enabled: bool = True
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 60.0


class LLMConfig(BaseModel):
    active_provider: str = "openai"
    providers: Dict[str, LLMProviderConfig] = Field(
        default_factory=lambda: {"openai": LLMProviderConfig()}
    )


class VoxCPMLocalConfig(BaseModel):
    adapter_type: Literal["voxcpm_local"] = "voxcpm_local"
    enabled: bool = True
    model_path: str = "openbmb/VoxCPM2"
    download_dir: str = ""
    device: str = "auto"
    default_cfg_value: float = 1.45
    default_inference_timesteps: int = 8
    normalize: bool = True
    load_denoiser: bool = False
    output_dir: str = "data/output"


class TTSConfig(BaseModel):
    active_adapter: str = "voxcpm_local"
    adapters: Dict[str, VoxCPMLocalConfig] = Field(
        default_factory=lambda: {"voxcpm_local": VoxCPMLocalConfig()}
    )


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)


def ensure_runtime_dirs() -> None:
    for path in (
        CONFIG_PATH.parent,
        PROMPT_PATH.parent,
        PROMPT_RUNTIME_PATH.parent,
        SOUL_PATH.parent,
        THOUGHTS_PATH.parent,
        VOICE_ROOT,
        VOICE_SYSTEM_ROOT,
        VOICE_USER_ROOT,
        DATA_ROOT,
        OUTPUT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    ensure_runtime_dirs()
    if not CONFIG_PATH.exists():
        config = AppConfig()
        save_config(config)
        return config

    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    llm_data = data.get("llm")
    if isinstance(llm_data, dict):
        providers = llm_data.get("providers")
        if isinstance(providers, dict) and "default" in providers and "openai" not in providers:
            providers["openai"] = providers.pop("default")
        if llm_data.get("active_provider") == "default":
            llm_data["active_provider"] = "openai"
    return AppConfig.model_validate(data)


def save_config(config: AppConfig) -> AppConfig:
    ensure_runtime_dirs()
    CONFIG_PATH.write_text(
        json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config


def merge_config_override(base: AppConfig, override: dict | None) -> AppConfig:
    if not override:
        return base
    merged = base.model_dump(mode="json")
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return AppConfig.model_validate(merged)
