from __future__ import annotations

import json
import re
from typing import Any, Dict, List
from urllib.parse import urljoin

import httpx

from .config import AppConfig
from .models import SegmentItem
from .prompts import build_segmentation_prompt
from .tts_adapters.registry import get_tts_adapter


class LLMUnavailableError(RuntimeError):
    pass


TARGET_SEGMENT_CHARS_MIN = 120
TARGET_SEGMENT_CHARS_MAX = 240
MAX_SEGMENT_CHARS = 300
MERGE_SHORT_CHARS = 70
TARGET_SEGMENT_WORDS_MIN = 55
TARGET_SEGMENT_WORDS_MAX = 105
MAX_SEGMENT_WORDS = 140
MERGE_SHORT_WORDS = 38
DEFAULT_CFG_VALUE = 1.6
DEFAULT_INFERENCE_TIMESTEPS = 8
REFERENCE_CFG_MIN = 1.25
REFERENCE_CFG_MAX = 1.60
REFERENCE_CFG_MAX_SHORT_CONTROL = 1.55
REFERENCE_CFG_MAX_MEDIUM_CONTROL = 1.48
REFERENCE_CFG_MAX_LONG_CONTROL = 1.40
REFERENCE_CFG_MAX_VERY_LONG_CONTROL = 1.34
REFERENCE_STEPS_MIN = 7
REFERENCE_STEPS_MAX = 9
GENERAL_CFG_MIN = 1.30
GENERAL_CFG_MAX = 1.90
GENERAL_STEPS_MIN = 4
GENERAL_STEPS_MAX = 12
DEFAULT_PAUSE_MS = 420
MAX_PAUSE_MS = 950


def _extract_json_block(content: str) -> Any:
    fenced = re.search(r"```json\s*(.*?)```", content, flags=re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1).strip())

    first = content.find("{")
    last = content.rfind("}")
    if first != -1 and last != -1 and last > first:
        return json.loads(content[first : last + 1])
    raise ValueError("LLM 返回内容中没有可解析的 JSON。")


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text or ""))


def _english_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or ""))


def _split_lang(text: str) -> str:
    return "zh" if _has_cjk(text) else "en"


def _split_len(text: str, lang: str | None = None) -> int:
    text = re.sub(r"\s+", " ", (text or "").strip())
    lang = lang or _split_lang(text)
    if lang == "zh":
        return len(re.sub(r"\s+", "", text))
    return _english_word_count(text)


def _target_min_len(text: str) -> int:
    return TARGET_SEGMENT_CHARS_MIN if _split_lang(text) == "zh" else TARGET_SEGMENT_WORDS_MIN


def _target_soft_len(text: str) -> int:
    return TARGET_SEGMENT_CHARS_MAX if _split_lang(text) == "zh" else TARGET_SEGMENT_WORDS_MAX


def _target_max_len(text: str) -> int:
    return MAX_SEGMENT_CHARS if _split_lang(text) == "zh" else MAX_SEGMENT_WORDS


def _merge_short_len(text: str) -> int:
    return MERGE_SHORT_CHARS if _split_lang(text) == "zh" else MERGE_SHORT_WORDS


def _safe_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(min_value, min(parsed, max_value))


def _safe_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(min_value, min(parsed, max_value))


def _cfg_cap_by_control_length(control: str) -> float:
    length = len((control or "").strip())
    if length <= 40:
        return REFERENCE_CFG_MAX_SHORT_CONTROL
    if length <= 120:
        return REFERENCE_CFG_MAX_MEDIUM_CONTROL
    if length <= 240:
        return REFERENCE_CFG_MAX_LONG_CONTROL
    return REFERENCE_CFG_MAX_VERY_LONG_CONTROL


def _normalize_cfg_value(value: Any, default: float, reference_mode: bool, control: str = "") -> float:
    if reference_mode:
        cfg = _safe_float(value, default, REFERENCE_CFG_MIN, REFERENCE_CFG_MAX)
        return min(cfg, _cfg_cap_by_control_length(control))
    return _safe_float(value, default, GENERAL_CFG_MIN, GENERAL_CFG_MAX)


def _normalize_steps(value: Any, default: int, reference_mode: bool) -> int:
    if reference_mode:
        return _safe_int(value, default, REFERENCE_STEPS_MIN, REFERENCE_STEPS_MAX)
    return _safe_int(value, default, GENERAL_STEPS_MIN, GENERAL_STEPS_MAX)


def _join_segment_text(left: str, right: str) -> str:
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    if _has_cjk(left + right):
        if re.search(r"[A-Za-z0-9]$", left) and re.search(r"^[A-Za-z0-9]", right):
            return f"{left} {right}"
        return left + right
    return f"{left} {right}"


def _split_units(text: str) -> List[str]:
    normalized = (text or "").strip()
    if not normalized:
        return []
    units = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;…])|(?<=\n)", normalized)
        if part and part.strip()
    ]
    return units or [normalized]


def _fallback_segments(text: str, control_hint: str, reference_mode: bool) -> List[SegmentItem]:
    units = _split_units(text)
    merged_units: List[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        candidate = _join_segment_text(current, unit)
        if _split_len(candidate) <= _target_soft_len(candidate):
            current = candidate
            continue
        merged_units.append(current)
        current = unit
    if current:
        merged_units.append(current)

    final_units: List[str] = []
    for unit in merged_units:
        if final_units and _split_len(unit) < _merge_short_len(unit):
            joined = _join_segment_text(final_units[-1], unit)
            if _split_len(joined) <= _target_max_len(joined):
                final_units[-1] = joined
                continue
        final_units.append(unit)

    segments: List[SegmentItem] = []
    for index, part in enumerate(final_units, start=1):
        pause_ms = 420 if re.search(r"[。！？!?；;…]$", part) else 260
        segments.append(
            SegmentItem(
                index=index,
                text=part,
                control=control_hint or "自然清晰，停连顺畅，关键短语轻微加重",
                pause_ms=0 if index == len(final_units) else pause_ms,
                emotion="neutral",
                cfg_value=_normalize_cfg_value(DEFAULT_CFG_VALUE, DEFAULT_CFG_VALUE, reference_mode, control_hint),
                inference_timesteps=_normalize_steps(DEFAULT_INFERENCE_TIMESTEPS, DEFAULT_INFERENCE_TIMESTEPS, reference_mode),
            )
        )
    return segments


def _merge_segment_pair(left: SegmentItem, right: SegmentItem, reference_mode: bool) -> SegmentItem:
    return SegmentItem(
        index=left.index,
        text=_join_segment_text(left.text, right.text),
        control=left.control or right.control,
        pause_ms=right.pause_ms,
        emotion=left.emotion if left.emotion != "neutral" else right.emotion,
        cfg_value=max(
            _normalize_cfg_value(left.cfg_value, DEFAULT_CFG_VALUE, reference_mode, left.control),
            _normalize_cfg_value(right.cfg_value, DEFAULT_CFG_VALUE, reference_mode, right.control),
        ),
        inference_timesteps=max(
            _normalize_steps(left.inference_timesteps, DEFAULT_INFERENCE_TIMESTEPS, reference_mode),
            _normalize_steps(right.inference_timesteps, DEFAULT_INFERENCE_TIMESTEPS, reference_mode),
        ),
    )


def _merge_short_segments(segments: List[SegmentItem], reference_mode: bool) -> List[SegmentItem]:
    if len(segments) <= 1:
        return segments

    out: List[SegmentItem] = []
    for seg in segments:
        if not out:
            out.append(seg)
            continue

        prev = out[-1]
        prev_len = _split_len(prev.text)
        cur_len = _split_len(seg.text)
        joined = _join_segment_text(prev.text, seg.text)
        joined_len = _split_len(joined)
        target_min = _target_min_len(joined)
        target_max = _target_max_len(joined)
        merge_short = _merge_short_len(joined)

        if (prev_len < merge_short or cur_len < merge_short or prev_len < target_min or cur_len < target_min) and joined_len <= target_max:
            out[-1] = _merge_segment_pair(prev, seg, reference_mode)
            continue
        out.append(seg)

    if len(out) >= 2:
        last = out[-1]
        prev = out[-2]
        if _split_len(last.text) < _merge_short_len(last.text):
            joined = _join_segment_text(prev.text, last.text)
            if _split_len(joined) <= _target_max_len(joined):
                out[-2] = _merge_segment_pair(prev, last, reference_mode)
                out.pop()

    for index, seg in enumerate(out, start=1):
        seg.index = index
        if index == len(out):
            seg.pause_ms = 0
    return out


def _resolve_model_name(provider) -> str:
    configured = (provider.model or "").strip()
    if configured:
        return configured

    models_url = urljoin(provider.base_url.rstrip("/") + "/", "models")
    headers: Dict[str, str] = {}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"

    try:
        with httpx.Client(timeout=min(provider.timeout_seconds, 12.0)) as client:
            response = client.get(models_url, headers=headers)
            response.raise_for_status()
        payload = response.json()
        for item in payload.get("data", []):
            model_id = str(item.get("id", "")).strip()
            if model_id:
                return model_id
    except Exception:
        pass

    return "gpt-4o-mini"


def _normalize_llm_segments(raw: Any, control_hint: str, reference_mode: bool) -> List[SegmentItem]:
    if isinstance(raw, dict):
        raw_segments = raw.get("segments") if isinstance(raw.get("segments"), list) else [raw]
    elif isinstance(raw, list):
        raw_segments = raw
    else:
        raise ValueError("LLM JSON must be an array or an object")

    segments: List[SegmentItem] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        control = str(item.get("control", control_hint or "")).strip()
        segments.append(
            SegmentItem(
                index=len(segments) + 1,
                text=text,
                control=control,
                pause_ms=_safe_int(item.get("pause_ms", DEFAULT_PAUSE_MS), DEFAULT_PAUSE_MS, 0, MAX_PAUSE_MS),
                emotion=str(item.get("emotion", "neutral")).strip() or "neutral",
                cfg_value=_normalize_cfg_value(item.get("cfg_value", DEFAULT_CFG_VALUE), DEFAULT_CFG_VALUE, reference_mode, control),
                inference_timesteps=_normalize_steps(item.get("inference_timesteps", DEFAULT_INFERENCE_TIMESTEPS), DEFAULT_INFERENCE_TIMESTEPS, reference_mode),
            )
        )

    if not segments:
        raise ValueError("LLM returned no usable segments")
    return _merge_short_segments(segments, reference_mode)


def segment_text_with_llm(
    config: AppConfig,
    text: str,
    llm_provider_id: str | None,
    tts_adapter_id: str | None,
    control_hint: str,
    soul_override: str | None,
    reference_mode: bool,
    reference_text: str | None,
) -> List[SegmentItem]:
    provider_id = llm_provider_id or config.llm.active_provider
    provider = config.llm.providers.get(provider_id)
    if not provider or not provider.enabled:
        return _fallback_segments(text, control_hint, reference_mode)

    adapter = get_tts_adapter(config, tts_adapter_id)
    prompt = build_segmentation_prompt(
        user_text=text,
        request_format_name=adapter.format_name,
        request_format_markdown=adapter.request_format_markdown(),
        soul_markdown=(soul_override or "").strip(),
        control_hint=control_hint,
        reference_mode=reference_mode,
        reference_text=reference_text,
    )
    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"

    payload: Dict[str, Any] = {
        "model": _resolve_model_name(provider),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 TTS 分段导演。请把文本整理成适合长句语音生成的自然段，"
                    "尽量避免切得太碎。中文普通叙述通常每段目标 120 到 240 个汉字，最多 300 个汉字；"
                    "英文普通叙述通常每段目标 55 到 105 个词，最多 140 个词。"
                    "你还要为每段给出 emotion、control、pause_ms、cfg_value、inference_timesteps。"
                    "如果启用了参考音频，cfg_value 更保守一些。只返回合法 JSON。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    try:
        with httpx.Client(timeout=provider.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        raw = _extract_json_block(content)
        return _normalize_llm_segments(raw, control_hint=control_hint, reference_mode=reference_mode)
    except Exception:
        return _fallback_segments(text, control_hint, reference_mode)
