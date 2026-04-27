from __future__ import annotations

import json

_ADAPTER_CACHE: dict[tuple[str, str], object] = {}


def get_tts_adapter(config, adapter_id: str | None = None):
    chosen_id = adapter_id or config.tts.active_adapter
    settings = config.tts.adapters[chosen_id]
    cache_key = (
        chosen_id,
        json.dumps(settings.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
    )
    cached = _ADAPTER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if settings.adapter_type == "voxcpm_local":
        from .voxcpm_local import VoxCPMLocalAdapter

        adapter = VoxCPMLocalAdapter(settings)
        _ADAPTER_CACHE[cache_key] = adapter
        return adapter
    raise KeyError(f"Unsupported TTS adapter: {chosen_id}")
