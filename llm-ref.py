# ============================================================
# 配置区：参考音频 / Ultimate 克隆版
# ============================================================
# 用途：
#   - 适合每次请求都上传 reference audio + reference text 的场景。
#   - 默认优先使用 VoxCPM2 的 high_similarity / Ultimate 文本引导极致克隆。
#   - 音色相似度优先；LLM 只负责分段、情绪分析、标点、停顿、重音和少量官方非语言标签。
#
# 请求建议：
#   - references[0].audio：参考音频 data URI。
#   - references[0].text：参考音频准确文本，越准确越适合 Ultimate 克隆。
#   - 默认 clone_mode=high_similarity；需要更强情绪控制时再传 extra_body.clone_mode="style_control"。
# ============================================================

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, Response, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Tuple, Iterator

import io
import os
import re
import json
import uuid
import base64
import tempfile
import traceback
import subprocess
import urllib.request
import urllib.error
import struct
import threading
import time
from datetime import datetime

import numpy as np
import soundfile as sf

from voxcpm import VoxCPM

try:
    import torch
    torch.set_float32_matmul_precision("high")
except Exception:
    torch = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# ============================================================
# 你主要需要改的地方
# ============================================================

# 1) 基础 VoxCPM2 模型目录。
#    参考音频版只使用原始 VoxCPM2，不包含其它音色分支。
#    Docker 里可用 -e VOXCPM_MODEL_PATH=/models/VoxCPM2 覆盖。
MODEL_PATH = os.environ.get("VOXCPM_MODEL_PATH", "/models/VoxCPM2")

# 2) voice 字段只作为 OpenAI TTS 兼容占位。
#    实际音色由每次请求传入的 reference audio 决定。
DEFAULT_VOICE_NAME = "reference_audio"

# 服务配置
HOST = "0.0.0.0"
PORT = 8000
# Docker 里可用 -e OPENAI_MODEL_NAME=voxcpm2 覆盖。
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "voxcpm2").strip() or "voxcpm2"
OUTPUT_DIR = "/data/voxcpm-data"

# VoxCPM 默认推理参数
DEFAULT_CFG_VALUE = 1.6
DEFAULT_INFERENCE_TIMESTEPS = 8
LOAD_DENOISER = False
DEFAULT_DENOISE = False
DEFAULT_NORMALIZE = True
DEFAULT_RETRY_BADCASE = True
DEFAULT_RETRY_BADCASE_MAX_TIMES = 3
DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD = 6.0
DEFAULT_MIN_LEN = 2
DEFAULT_MAX_LEN = 4096

# 参考音频处理
MAX_REFERENCE_AUDIO_BYTES = 30 * 1024 * 1024
CONVERT_REFERENCE_AUDIO_TO_WAV = True
REFERENCE_WAV_SAMPLE_RATE = 16000

# LLM 文本导演配置
# 以下几项适配你的 docker run -e 参数；其它 LLM 参数仍在 Python 里固定。
TTS_LLM_ENABLE = _env_bool("TTS_LLM_ENABLE", True)
TTS_LLM_BASE_URL = os.environ.get("TTS_LLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
TTS_LLM_MODEL = os.environ.get("TTS_LLM_MODEL", "").strip()
TTS_LLM_API_KEY = os.environ.get("TTS_LLM_API_KEY", "EMPTY")
TTS_LLM_TIMEOUT = 30.0
TTS_LLM_TEMPERATURE = 0.2
TTS_LLM_MAX_TOKENS = 3072
TTS_LLM_FALLBACK_ON_ERROR = False
_DISCOVERED_LLM_MODEL: Optional[str] = None

DEFAULT_STYLE_PROMPT = (
    "参考音频音色优先：如果提供了参考音频和准确参考文本，默认使用 VoxCPM2 的 high_similarity / Ultimate 文本引导极致克隆思路，"
    "把参考音频作为已说出的前文进行音频续写，尽量复刻同一说话人的音色、口音、节奏和情绪底色。"
    "Ultimate 克隆下不要强行使用括号 Control Instruction 改声线；情绪主要通过正文语义、标点、自然停连和少量官方非语言标签体现。"
    "可控克隆 style_control 模式才使用轻量 control，且只能描述语气、语速、停连、重音、笑意、叹息等表演动作。"
    "如果 control 较长，后端会自动降低 cfg_value，让音色更收敛到参考音频。"
    "不要写会改变音色身份的词，例如换声线、换音色、男声、女声、年龄、性别、另一个人的声音等。"
    "分段按两个自然句或一个完整意群组织，避免一两句话就重启音色；"
    "英文约 55-105 个词一段，中文约 120-240 字一段；后端尊重 LLM 分段，不再二次合并。"
)

# 分段和停顿策略。LLM 会给建议，后端按 VoxCPM 官方 split_paragraph 的默认思路做归一化：
# 中文按字符、英文按词数近似官方 tokenizer 长度；LLM 分段结果不再被后端二次合并，pause 做兜底。
DEFAULT_SEGMENT = True
DEFAULT_LLM_PREPROCESS = True
DEFAULT_AUTO_EMOTION = True

# VoxCPM 官方工具 split_paragraph 默认 token_min_n=60、token_max_n=80、merge_len=20。
# 按你日志里“每两句合为一段”的长度校准：
# 英文约 55-105 words 一段，中文约 120-240 字一段。后端尊重 LLM 分段，不再二次合并。
# 中文直接按字符数；英文这里没有官方 tokenizer，按单词数近似处理。
TARGET_SEGMENT_CHARS_MIN = 120
TARGET_SEGMENT_CHARS_MAX = 240
MAX_SEGMENT_CHARS = 300
MERGE_SHORT_CHARS = 70
TARGET_SEGMENT_WORDS_MIN = 55
TARGET_SEGMENT_WORDS_MAX = 105
MAX_SEGMENT_WORDS = 140
MERGE_SHORT_WORDS = 38

# Ultimate/high_similarity 模式下，官方非语言标签是弱提示。
# 为了让笑声更容易被听出来，这里把 [laughing] 辅助成可朗读的短笑声提示。
# 如果你只想保留官方标签本身，可以把它改成 False。
ENABLE_LAUGH_TEXT_CUE = True

DEFAULT_PAUSE_MS = 420
MIN_PAUSE_MS = 50
MAX_PAUSE_MS = 950
COMMA_PAUSE_MIN = 120
COMMA_PAUSE_MAX = 260
SENTENCE_PAUSE_MIN = 360
SENTENCE_PAUSE_MAX = 620
LONG_PAUSE_MIN = 560
LONG_PAUSE_MAX = 900
CROSSFADE_MS = 25
FADE_MS = 5
TRIM_SILENCE = True
STREAM_CHUNK_MS = 250

# 参考音频优先策略：先保同一说话人身份，control 只做表演指导。
REFERENCE_CONTROL_ANCHOR = "保持参考音频的同一说话人音色、口音和发声习惯"
REFERENCE_CONTROL_MAX_CHARS = 600
MERGED_CONTROL_MAX_CHARS = 900
REFERENCE_DEFAULT_CFG_VALUE = 1.45
REFERENCE_CFG_MIN = 1.25
REFERENCE_CFG_MAX = 1.60
# control 真正拼进括号时，控制词越长，cfg 越要保守。
REFERENCE_CFG_MAX_SHORT_CONTROL = 1.55
REFERENCE_CFG_MAX_MEDIUM_CONTROL = 1.48
REFERENCE_CFG_MAX_LONG_CONTROL = 1.40
REFERENCE_CFG_MAX_VERY_LONG_CONTROL = 1.34
REFERENCE_DEFAULT_INFERENCE_TIMESTEPS = 8
REFERENCE_INFERENCE_TIMESTEPS_MIN = 7
REFERENCE_INFERENCE_TIMESTEPS_MAX = 9

_SAVE_LOCK = threading.Lock()
_MODEL_LOCK = threading.RLock()

# ============================================================
# FastAPI / model
# FastAPI / model
# ============================================================

app = FastAPI(title="VoxCPM2 Reference Clone Server")
model = None

_AUDIO_EXT_BY_MIME = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
}


class TTSRequest(BaseModel):
    text: str
    cfg_value: float = DEFAULT_CFG_VALUE
    inference_timesteps: int = DEFAULT_INFERENCE_TIMESTEPS
    voice: str = DEFAULT_VOICE_NAME


class OpenAITTSRequest(BaseModel):
    model: str = OPENAI_MODEL_NAME
    input: str
    voice: str = "default"
    speed: float = 1.0
    response_format: str = "wav"
    stream: Optional[bool] = None
    extra_body: Optional[Dict[str, Any]] = None
    references: Optional[List[Dict[str, Any]]] = None
    reference_voice: Optional[str] = None
    speaker: Optional[str] = None
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None


class TTSPlanDebugRequest(BaseModel):
    text: str
    speed: float = 1.0
    voice: str = DEFAULT_VOICE_NAME
    prompt_text: Optional[str] = None
    has_reference_audio: bool = True
    extra_body: Optional[Dict[str, Any]] = None


# ============================================================
# 基础工具
# ============================================================

def _safe_int(value: Any, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _safe_float(value: Any, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _json_preview(obj: Any, max_chars: int = 4000) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)[:max_chars]
    except Exception:
        return repr(obj)[:max_chars]


def _json_pretty(obj: Any, max_chars: int = 12000) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)[:max_chars]
    except Exception:
        return repr(obj)[:max_chars]


def _get_extra(req: OpenAITTSRequest) -> Dict[str, Any]:
    return req.extra_body if isinstance(req.extra_body, dict) else {}


def _is_stream_request(req: OpenAITTSRequest) -> bool:
    extra = _get_extra(req)
    if req.stream is not None:
        return bool(req.stream)
    return bool(extra.get("stream", False))


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text or ""))


def _english_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or ""))


OFFICIAL_NONVERBAL_TAG_DESCRIPTIONS: Dict[str, str] = {
    "[laughing]": (
        "轻笑、愉快地笑、带笑意地开口。"
        "适合开心、释然、害羞、温柔自嘲、幸福感溢出的句子。"
        "不要用于大笑、狂笑或严肃悲伤场景。"
    ),
    "[sigh]": (
        "轻叹气。"
        "适合疲惫、遗憾、释然、无奈、心事很重、说话前短暂泄气的场景。"
        "不要频繁使用。"
    ),
    "[Uhm]": (
        "犹豫、思考、欲言又止的填充音。"
        "适合角色一边想一边说、措辞谨慎、不确定、害羞或情绪卡住的场景。"
        "通常放在句首或转折前。"
    ),
    "[Shh]": (
        "轻声示意安静、压低声音。"
        "适合秘密、安抚、贴近耳语、不要惊动别人、温柔制止的场景。"
        "不要用于普通叙述。"
    ),
    "[Question-ah]": (
        "带疑问感的语气助词，偏轻微疑惑或确认。"
        "适合温和追问、轻声确认、带一点不确定的问句。"
    ),
    "[Question-ei]": (
        "带疑问感的语气助词，偏惊讶、挑眉、没想到。"
        "适合突然发现问题、轻微意外、反问或惊奇的问句。"
    ),
    "[Question-en]": (
        "带疑问感的语气助词，偏沉吟、认真确认。"
        "适合低声思考后的疑问、克制的追问、带迟疑的确认。"
    ),
    "[Question-oh]": (
        "带疑问感的语气助词，偏柔和、拉长、带一点恍然。"
        "适合温柔疑问、轻声回应、带情绪余韵的问句。"
    ),
    "[Surprise-wa]": (
        "惊讶、忽然被触动或被吓到的短促反应。"
        "适合突然发现、惊喜、被震撼、情绪瞬间抬起。"
        "不要用于平静叙述。"
    ),
    "[Surprise-yo]": (
        "更明显的惊讶或感叹。"
        "适合强烈惊喜、不可思议、情绪外放的瞬间。"
        "比 [Surprise-wa] 更明显，使用要更克制。"
    ),
    "[Dissatisfaction-hnn]": (
        "不满、别扭、轻哼、压着情绪的反应。"
        "适合傲娇、不服气、委屈、轻微抗拒、嘴硬但不是愤怒爆发的场景。"
    ),
}

# 后端白名单仍然只使用 tag 本身，避免把说明文字混进正文。
OFFICIAL_NONVERBAL_TAGS = set(OFFICIAL_NONVERBAL_TAG_DESCRIPTIONS.keys())


def _format_nonverbal_tag_guide() -> str:
    return "\n".join(
        f"- {tag}: {description}"
        for tag, description in OFFICIAL_NONVERBAL_TAG_DESCRIPTIONS.items()
    )


def _sanitize_control_instruction(value: str) -> str:
    """
    VoxCPM2 的 Control Instruction 需要放在目标文本开头的括号里。
    这里保留自然语言表演指导，但去掉括号、JSON/Markdown 痕迹和容易污染正文的字段名。
    """
    value = str(value or "").strip()
    value = re.sub(r"^`+|`+$", "", value)
    value = re.sub(r"(?i)^\s*(control\s*instruction|control|style|emotion)\s*[:：]\s*", "", value)
    value = re.sub(r"[()（）{}]", "", value)
    value = re.sub(r"[\"“”]", "", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" ，,。.;；")
    if not value:
        return ""
    return value[:1200]




def _truncate_control_by_clauses(value: str, max_chars: int) -> str:
    value = _sanitize_control_instruction(value)
    if len(value) <= max_chars:
        return value
    clauses = [c.strip() for c in re.split(r"[，,；;。]+", value) if c.strip()]
    kept: List[str] = []
    total = 0
    for clause in clauses:
        add_len = len(clause) + (1 if kept else 0)
        if kept and total + add_len > max_chars:
            break
        if not kept and len(clause) > max_chars:
            return clause[:max_chars].rstrip(" ，,。.;；")
        kept.append(clause)
        total += add_len
    return "，".join(kept).strip(" ，,。.;；") or value[:max_chars].rstrip(" ，,。.;；")


def _reference_control_clauses(value: str) -> List[str]:
    """
    参考音频模式下，保留与音色无关的表演指令，并按原顺序拆成子句。
    允许保留停顿位置、关键词重音、情绪转折、语速、音量、笑意/叹息等；
    删除会改变说话人身份或声线质感的描述。
    """
    value = _sanitize_control_instruction(value)
    if not value:
        return []

    replacements = {
        "压低声音": "音量略低",
        "声音压低": "音量略低",
        "放轻声音": "音量放轻",
        "声音放轻": "音量放轻",
        "提高声音": "音量略高",
        "声音提高": "音量略高",
        "slight breathy quality": "slight breath before the emphasized words",
        "breathy quality": "light breath before the emphasized words",
    }
    for a, b in replacements.items():
        value = value.replace(a, b)

    banned_patterns = [
        r"音色", r"声线", r"嗓音", r"发声位置", r"音高", r"口腔位置", r"胸腔共鸣", r"^声音",
        r"男声|女声|少女|少年|御姐|萝莉|大叔|年轻女性|年轻男性|中年女性|中年男性|老年",
        r"(?:声音|声线|嗓音).{0,10}(?:明亮|低沉|沙哑|有磁性|甜美|厚实|清澈|稚嫩|成熟|温暖|冷冽|阴冷|浑厚|尖细|柔和|轻盈|粗糙|厚重|空灵|奶|萝莉|御姐|少年感|少女感)",
        r"(?:明亮|低沉|沙哑|有磁性|甜美|厚实|清澈|稚嫩|成熟|温暖|冷冽|阴冷|浑厚|尖细|柔和|轻盈|粗糙|厚重|空灵)的?(?:声音|声线|嗓音)",
        r"带(?:一点|一些|着)?气声",
        r"(?i)voice\s*(?:quality|color|timbre|texture|type)",
        r"(?i)(?:female|male|girl|boy|woman|man)\s+voice",
        r"(?i)(?:bright|deep|husky|sweet|magnetic|raspy|soft|airy)\s+voice",
    ]

    raw_clauses = [c.strip() for c in re.split(r"[，,；;。]+", value) if c.strip()]
    kept: List[str] = []
    seen: List[str] = []
    for clause in raw_clauses:
        if any(re.search(pat, clause) for pat in banned_patterns):
            continue
        norm = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", clause.lower())
        if not norm:
            continue
        duplicate = False
        for prev in seen:
            if norm == prev or norm in prev or prev in norm:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(clause)
        seen.append(norm)
    return kept


def _join_control_clauses(clauses: List[str], max_chars: int) -> str:
    kept: List[str] = []
    total = 0
    for clause in clauses:
        clause = _sanitize_control_instruction(clause)
        if not clause:
            continue
        add = len(clause) + (1 if kept else 0)
        if kept and total + add > max_chars:
            break
        if not kept and len(clause) > max_chars:
            return clause[:max_chars].rstrip(" ，,。.;；")
        kept.append(clause)
        total += add
    return "，".join(kept).strip(" ，,。.;；")


def _sanitize_reference_control_instruction(value: str, max_chars: int = REFERENCE_CONTROL_MAX_CHARS) -> str:
    """
    参考音频模式下只做最小限制性清理：
    - 保留 LLM 给出的停顿位置、关键词重音、情绪转折、语速、尾音、笑意/叹息等顺序信息；
    - 只移除少数会明显改变说话人身份或音色来源的描述；
    - 不再按逗号拆碎、不再主动压缩 control。
    """
    cleaned = _sanitize_control_instruction(value)
    if not cleaned:
        return "语气贴合正文情绪，语速自然，停连顺畅，关键短语轻微加重"

    # 只处理“换说话人 / 换音色 / 指定性别年龄声线”这类致命音色风险。
    fatal_patterns = [
        r"(?i)\b(?:female|male|girl|boy|woman|man|child|elderly)\s+voice\b",
        r"(?i)\b(?:change|switch)\s+(?:the\s+)?voice\b",
        r"(?i)\bsound\s+like\s+(?:a\s+)?(?:different\s+)?(?:person|speaker|woman|man|girl|boy|child)\b",
        r"(?i)\b(?:voice\s*)?(?:timbre|voice\s*quality|voice\s*texture|voice\s*type)\b",
        r"男声|女声|童声|少女声|少年声|萝莉音|御姐音|大叔音|老人声",
        r"换(?:成|一种)?(?:音色|声线|嗓音|声音)",
        r"(?:模仿|变成|听起来像)(?:另一个人|男性|女性|小孩|老人)",
        r"音色|声线|嗓音|发声位置|口腔位置|胸腔共鸣",
    ]
    for pat in fatal_patterns:
        cleaned = re.sub(pat, "", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"[，,；;。]{2,}", "，", cleaned)
    cleaned = cleaned.strip(" ，,。.;；")
    if not cleaned:
        return "语气贴合正文情绪，语速自然，停连顺畅，关键短语轻微加重"

    if max_chars and len(cleaned) > max_chars:
        # 只有极端过长时才截断，优先在句子边界截，避免半句。
        cut = cleaned[:max_chars]
        boundary = max(cut.rfind("."), cut.rfind("。"), cut.rfind(";"), cut.rfind("；"))
        if boundary >= int(max_chars * 0.65):
            cut = cut[:boundary + 1]
        cleaned = cut.rstrip(" ，,。.;；")
    return cleaned


def _voice_identity_sensitive(policy: Optional[Dict[str, Any]]) -> bool:
    """参考音频版只在有 reference audio 时进入音色优先模式。"""
    if not isinstance(policy, dict):
        return False
    return bool(policy.get("has_reference_audio"))


def _normalize_cfg_value_for_policy(value: Any, default: float, policy: Optional[Dict[str, Any]]) -> float:
    if _voice_identity_sensitive(policy):
        return _safe_float(value, REFERENCE_DEFAULT_CFG_VALUE, REFERENCE_CFG_MIN, REFERENCE_CFG_MAX)
    return _safe_float(value, default, 1.4, 1.9)


def _control_char_len_for_cfg(control: str) -> int:
    control = _sanitize_control_instruction(control)
    cjk = re.findall(r"[一-鿿぀-ヿ가-힯]", control)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", control)
    return len(cjk) + len(words) * 2


def _cfg_cap_by_control_length(control: str) -> float:
    length = _control_char_len_for_cfg(control)
    if length <= 60:
        return REFERENCE_CFG_MAX_SHORT_CONTROL
    if length <= 120:
        return REFERENCE_CFG_MAX_MEDIUM_CONTROL
    if length <= 200:
        return REFERENCE_CFG_MAX_LONG_CONTROL
    return REFERENCE_CFG_MAX_VERY_LONG_CONTROL


def _adaptive_cfg_value_for_segment(value: Any, default: float, seg: Dict[str, Any], policy: Optional[Dict[str, Any]]) -> float:
    """control 越长，参考音频模式下 cfg 越保守，避免音色漂移。"""
    cfg = _normalize_cfg_value_for_policy(value, default, policy)
    if not _voice_identity_sensitive(policy):
        return cfg
    if not bool(seg.get("use_control_instruction", False)) or not _control_instruction_allowed(policy):
        return cfg
    control = _sanitize_reference_control_instruction(str(seg.get("control", "")), max_chars=MERGED_CONTROL_MAX_CHARS)
    cap = _cfg_cap_by_control_length(control)
    adjusted = max(REFERENCE_CFG_MIN, min(cfg, cap))
    if adjusted < cfg:
        print(f"Adaptive cfg for reference audio: {cfg:.2f} -> {adjusted:.2f} because control_len={_control_char_len_for_cfg(control)} control_preview={control[:120]}", flush=True)
    return adjusted


def _normalize_timesteps_for_policy(value: Any, default: int, policy: Optional[Dict[str, Any]]) -> int:
    if _voice_identity_sensitive(policy):
        return _safe_int(value, REFERENCE_DEFAULT_INFERENCE_TIMESTEPS, REFERENCE_INFERENCE_TIMESTEPS_MIN, REFERENCE_INFERENCE_TIMESTEPS_MAX)
    return _safe_int(value, default, 6, 12)


def _sanitize_tts_text_with_official_tags(value: str) -> str:
    """
    允许官方推荐的少量方括号非语言标签；其它疑似标签删除，避免 LLM 自造标签被读出来。
    """
    value = re.sub(r"\s+", " ", str(value or "").strip())

    def _replace_tag(match: re.Match) -> str:
        tag = match.group(0)
        return tag if tag in OFFICIAL_NONVERBAL_TAGS else ""

    value = re.sub(r"\[[A-Za-z][A-Za-z0-9_-]{0,40}\]", _replace_tag, value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _control_instruction_allowed(policy: Dict[str, Any]) -> bool:
    """
    Ultimate/high_similarity 克隆用 prompt_wav_path + prompt_text 做音频续写，
    官方 Demo 也提示该模式会禁用 Control Instruction；默认 style_control 才启用分段控制词。
    """
    if not isinstance(policy, dict):
        return True
    if (
        policy.get("has_reference_audio")
        and policy.get("clone_mode") == "high_similarity"
        and policy.get("prompt_text_is_exact", True)
    ):
        return False
    return True


def _looks_like_voice_description(value: str) -> bool:
    value = (value or "").strip()
    if not value:
        return False
    if _has_cjk(value):
        return True
    return bool(re.search(r"\s|,|，|voice|male|female|young|old|gentle|deep|warm|calm|slow|fast", value, re.I))


# ============================================================
# 模型加载与参考音频音色策略
# ============================================================

@app.on_event("startup")
def load_model():
    global model

    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Model path not found: {MODEL_PATH}")

    model = VoxCPM.from_pretrained(
        MODEL_PATH,
        load_denoiser=LOAD_DENOISER,
        optimize=False,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(
        f"VoxCPM loaded: model_path={MODEL_PATH}, load_denoiser={LOAD_DENOISER}",
        flush=True,
    )
    print("Reference clone mode: voice is provided by request reference audio", flush=True)
    print(f"Generated audio output dir: {OUTPUT_DIR}", flush=True)


def _resolve_voice_for_request(req: OpenAITTSRequest, has_reference_audio: bool) -> Dict[str, Any]:
    """参考音频版不做内置音色选择；音色只来自请求里的 reference audio。"""
    return {
        "voice_name": "reference_audio" if has_reference_audio else "base_model",
        "voice_source": "reference_audio" if has_reference_audio else "base_model_no_reference",
    }


# ============================================================
# 请求/参考音频处理
# ============================================================

def _extract_wav_from_result(result: Any) -> np.ndarray:
    wav = result
    if isinstance(result, tuple):
        wav = result[0]
    elif isinstance(result, dict):
        wav = None
        for key in ("audio", "wav", "waveform"):
            if key in result and result[key] is not None:
                wav = result[key]
                break
    if wav is None:
        raise ValueError("No waveform found in generation result")
    if torch is not None and isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()
    else:
        wav = np.asarray(wav)
    wav = np.squeeze(wav)
    if wav.size == 0:
        raise ValueError("Generated waveform is empty")
    return wav.astype(np.float32, copy=False).reshape(-1)


def _get_sample_rate() -> int:
    if model is None:
        return 48000
    try:
        sample_rate = getattr(model.tts_model, "sample_rate", None)
    except Exception:
        sample_rate = None
    return int(sample_rate or 48000)


def _save_data_uri_audio_to_temp(audio_data_uri: str) -> str:
    if not audio_data_uri:
        raise HTTPException(status_code=400, detail="reference audio is empty")
    m = re.match(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", audio_data_uri, flags=re.DOTALL)
    if not m:
        raise HTTPException(status_code=400, detail="reference audio must be a data URI like data:audio/wav;base64,...")
    mime = m.group("mime").lower().strip()
    b64 = re.sub(r"\s+", "", m.group("data"))
    if not mime.startswith("audio/"):
        raise HTTPException(status_code=400, detail=f"unsupported reference mime: {mime}")
    try:
        audio_bytes = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid reference audio base64: {repr(e)}")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="reference audio decoded to empty bytes")
    if len(audio_bytes) > MAX_REFERENCE_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"reference audio too large: {len(audio_bytes)} bytes")
    suffix = _AUDIO_EXT_BY_MIME.get(mime, ".audio")
    fd, path = tempfile.mkstemp(prefix="voxcpm_ref_raw_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    print(f"Saved reference audio: mime={mime}, path={path}, bytes={len(audio_bytes)}", flush=True)
    return path


def _convert_audio_file_to_wav(input_path: str) -> str:
    output_path = os.path.join(tempfile.gettempdir(), f"voxcpm_ref_wav_{uuid.uuid4().hex}.wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
        "-ac", "1", "-ar", str(REFERENCE_WAV_SAMPLE_RATE), output_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found. Install ffmpeg in the Docker image.")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
        raise HTTPException(status_code=400, detail=f"failed to convert reference audio to wav: {stderr[:1000]}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise HTTPException(status_code=400, detail="converted reference wav is empty")
    return output_path


def _prepare_reference_audio_path(raw_path: str) -> Tuple[str, List[str]]:
    cleanup_files = [raw_path]
    if not CONVERT_REFERENCE_AUDIO_TO_WAV:
        return raw_path, cleanup_files
    wav_path = _convert_audio_file_to_wav(raw_path)
    cleanup_files.append(wav_path)
    return wav_path, cleanup_files


def _extract_reference_from_openai_request(req: OpenAITTSRequest) -> Tuple[Optional[str], Optional[str], List[str]]:
    refs = req.references or (_get_extra(req).get("references") if isinstance(_get_extra(req), dict) else None)
    if not refs:
        return None, None, []
    if not isinstance(refs, list) or len(refs) == 0:
        raise HTTPException(status_code=400, detail="references must be a non-empty list")
    ref0 = refs[0]
    if not isinstance(ref0, dict):
        raise HTTPException(status_code=400, detail="references[0] must be an object")
    ref_audio = ref0.get("audio")
    ref_text = (ref0.get("text") or "").strip() or None
    if not ref_audio:
        return None, ref_text, []
    if not isinstance(ref_audio, str):
        raise HTTPException(status_code=400, detail="references[0].audio must be a string data URI")
    raw_ref_path = _save_data_uri_audio_to_temp(ref_audio)
    usable_ref_path, cleanup_files = _prepare_reference_audio_path(raw_ref_path)
    return usable_ref_path, ref_text, cleanup_files


def _cleanup_files(paths: List[str]):
    for p in paths:
        if not p:
            continue
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception as e:
            print(f"Failed to clean temp file {p}: {repr(e)}", flush=True)


# ============================================================
# 后端 TTS 策略
# ============================================================

def _speed_to_control(speed: float) -> str:
    if speed <= 0.75:
        return "语速较慢"
    if speed < 0.95:
        return "语速稍慢"
    if speed >= 1.25:
        return "语速较快"
    if speed > 1.08:
        return "语速稍快"
    return "语速中等"


def _count_speech_units(text: Optional[str]) -> float:
    if not text:
        return 0.0
    text = re.sub(r"\s+", " ", text.strip())
    cjk = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
    english_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    digits = re.findall(r"\d+", text)
    return float(len(cjk)) + float(len(english_words)) * 1.6 + float(len(digits)) * 1.2


def _audio_duration_sec(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    try:
        info = sf.info(path)
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def _analyze_reference_audio_features(reference_wav_path: Optional[str]) -> Dict[str, Any]:
    """
    轻量参考音频分析：不新增依赖，只用 soundfile + numpy 提取能量、静音比例和粗略节奏。
    这些结果只提供给 LLM 做“表演边界”参考，不直接改变音色。
    """
    base = {
        "reference_rms": None,
        "reference_peak": None,
        "reference_active_ratio": None,
        "reference_silence_ratio": None,
        "reference_mean_silence_ms": None,
        "reference_energy_label": "unknown",
        "reference_pause_label": "unknown",
        "reference_style_hint": "参考音频特征未知，优先保持原音色和自然节奏。",
    }
    if not reference_wav_path or not os.path.exists(reference_wav_path):
        return base
    try:
        wav, sr = sf.read(reference_wav_path, dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = np.nan_to_num(wav.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if wav.size < max(1, int(sr * 0.2)):
            return base

        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
        threshold = max(0.008, rms * 0.45)
        frame = max(1, int(sr * 0.03))
        frame_count = max(1, wav.size // frame)
        trimmed = wav[:frame_count * frame].reshape(frame_count, frame)
        frame_rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
        active = frame_rms > threshold
        active_ratio = float(np.mean(active)) if active.size else None
        silence_ratio = 1.0 - active_ratio if active_ratio is not None else None

        runs = []
        cur = 0
        for is_active in active:
            if not bool(is_active):
                cur += 1
            elif cur > 0:
                runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        silence_runs_ms = [r * 30.0 for r in runs if r * 30.0 >= 90.0]
        mean_silence_ms = float(np.mean(silence_runs_ms)) if silence_runs_ms else 0.0

        if rms < 0.025:
            energy_label = "low"
        elif rms < 0.065:
            energy_label = "medium"
        else:
            energy_label = "high"

        if mean_silence_ms >= 520 or (silence_ratio is not None and silence_ratio > 0.42):
            pause_label = "many_or_long_pauses"
        elif mean_silence_ms >= 260 or (silence_ratio is not None and silence_ratio > 0.25):
            pause_label = "moderate_pauses"
        else:
            pause_label = "tight_pauses"

        if energy_label == "low":
            energy_hint = "参考音频整体能量偏低，后续情绪不要突然变成高亢喊叫。"
        elif energy_label == "high":
            energy_hint = "参考音频整体能量偏高，后续可以保持更外放的表达，但仍需维持同一说话人。"
        else:
            energy_hint = "参考音频能量中等，后续保持自然起伏。"

        if pause_label == "many_or_long_pauses":
            pause_hint = "参考音频停顿较多，长段内可保留自然停连，但不要额外拉长段间静音。"
        elif pause_label == "tight_pauses":
            pause_hint = "参考音频停连较紧，分段和标点都应更连贯。"
        else:
            pause_hint = "参考音频停连适中。"

        return {
            "reference_rms": round(rms, 6),
            "reference_peak": round(peak, 6),
            "reference_active_ratio": round(active_ratio, 4) if active_ratio is not None else None,
            "reference_silence_ratio": round(silence_ratio, 4) if silence_ratio is not None else None,
            "reference_mean_silence_ms": round(mean_silence_ms, 1),
            "reference_energy_label": energy_label,
            "reference_pause_label": pause_label,
            "reference_style_hint": energy_hint + pause_hint,
        }
    except Exception as e:
        print("Reference audio analysis failed:", repr(e), flush=True)
        return base


def _estimate_reference_speech_profile(reference_wav_path: Optional[str], prompt_text: Optional[str], request_speed: float) -> Dict[str, Any]:
    duration = _audio_duration_sec(reference_wav_path)
    units = _count_speech_units(prompt_text)
    units_per_sec = None
    label = "unknown"
    scale = 0.82
    if duration and duration > 0.5 and units > 1:
        units_per_sec = units / duration
        if units_per_sec < 3.2:
            label, scale = "slow", 0.95
        elif units_per_sec < 4.4:
            label, scale = "medium_slow", 0.78
        elif units_per_sec <= 6.2:
            label, scale = "normal", 0.62
        elif units_per_sec <= 7.8:
            label, scale = "medium_fast", 0.50
        else:
            label, scale = "fast", 0.42
    spd = _safe_float(request_speed, 1.0, 0.5, 2.0)
    if spd < 0.85:
        scale *= 1.08
    elif spd < 0.95:
        scale *= 1.03
    elif spd > 1.25:
        scale *= 0.78
    elif spd > 1.08:
        scale *= 0.88
    scale = max(0.32, min(1.05, scale))
    return {
        "reference_duration_sec": duration,
        "reference_text_units": units,
        "reference_units_per_sec": units_per_sec,
        "reference_speech_rate": label,
        "pause_scale": scale,
    }


def _get_backend_tts_policy(req: OpenAITTSRequest, reference_wav_path: Optional[str], prompt_text: Optional[str]) -> Dict[str, Any]:
    extra = _get_extra(req)
    has_reference = bool(reference_wav_path)
    speech_profile = _estimate_reference_speech_profile(reference_wav_path, prompt_text, req.speed)
    audio_features = _analyze_reference_audio_features(reference_wav_path)
    voice_policy = _resolve_voice_for_request(req, has_reference_audio=has_reference)

    clone_mode_default = "high_similarity" if (has_reference and prompt_text and str(prompt_text).strip()) else ("style_control" if has_reference else "normal")
    clone_mode = str(extra.get("clone_mode", clone_mode_default)).strip() or clone_mode_default

    voice_identity_sensitive = bool(has_reference)

    policy = {
        "clone_mode": clone_mode,
        "has_reference_audio": has_reference,
        "voice_identity_sensitive": voice_identity_sensitive,
        "prompt_text_is_exact": bool(extra.get("prompt_text_is_exact", True)),
        "llm_preprocess": bool(extra.get("llm_preprocess", DEFAULT_LLM_PREPROCESS)),
        "auto_emotion": bool(extra.get("auto_emotion", DEFAULT_AUTO_EMOTION)),
        "segment": bool(extra.get("segment", DEFAULT_SEGMENT)),
        "style_prompt": str(extra.get("style_prompt", DEFAULT_STYLE_PROMPT)).strip(),
        "normalize": bool(extra.get("normalize", DEFAULT_NORMALIZE)),
        "denoise": False,
        "retry_badcase": bool(extra.get("retry_badcase", DEFAULT_RETRY_BADCASE)),
        "retry_badcase_max_times": _safe_int(extra.get("retry_badcase_max_times", DEFAULT_RETRY_BADCASE_MAX_TIMES), DEFAULT_RETRY_BADCASE_MAX_TIMES, 0, 10),
        "retry_badcase_ratio_threshold": _safe_float(extra.get("retry_badcase_ratio_threshold", DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD), DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD, 1.0, 20.0),
        "min_len": _safe_int(extra.get("min_len", DEFAULT_MIN_LEN), DEFAULT_MIN_LEN, 0, 100),
        "max_len": _safe_int(extra.get("max_len", DEFAULT_MAX_LEN), DEFAULT_MAX_LEN, 64, 20000),
        "speed": _safe_float(req.speed, 1.0, 0.5, 2.0),
        "target_segment_chars_min": TARGET_SEGMENT_CHARS_MIN,
        "target_segment_chars_max": TARGET_SEGMENT_CHARS_MAX,
        "max_segment_chars": MAX_SEGMENT_CHARS,
        "merge_short_chars": MERGE_SHORT_CHARS,
        "target_segment_words_min": TARGET_SEGMENT_WORDS_MIN,
        "target_segment_words_max": TARGET_SEGMENT_WORDS_MAX,
        "max_segment_words": MAX_SEGMENT_WORDS,
        "merge_short_words": MERGE_SHORT_WORDS,
        "enable_laugh_text_cue": bool(extra.get("enable_laugh_text_cue", ENABLE_LAUGH_TEXT_CUE)),
        "default_pause_ms": DEFAULT_PAUSE_MS,
        "max_pause_ms": MAX_PAUSE_MS,
        "crossfade_ms": CROSSFADE_MS,
        "fade_ms": FADE_MS,
        "trim_silence": TRIM_SILENCE,
        "stream_chunk_ms": STREAM_CHUNK_MS,
        **speech_profile,
        **audio_features,
        **voice_policy,
    }
    return policy


# ============================================================
# LLM 调用和分段
# ============================================================

def _clip_for_prompt(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if TTS_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {TTS_LLM_API_KEY}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"LLM HTTPError {e.code}: {err[:1000]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM URLError: {repr(e)}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM response is not JSON: {repr(e)}")


def _get_json(url: str, timeout: float) -> Dict[str, Any]:
    headers = {}
    if TTS_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {TTS_LLM_API_KEY}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"LLM models HTTPError {e.code}: {err[:1000]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM models URLError: {repr(e)}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM models response is not JSON: {repr(e)}")


def _resolve_llm_model() -> str:
    global _DISCOVERED_LLM_MODEL
    if TTS_LLM_MODEL:
        return TTS_LLM_MODEL
    if _DISCOVERED_LLM_MODEL:
        return _DISCOVERED_LLM_MODEL
    url = f"{TTS_LLM_BASE_URL}/models"
    data = _get_json(url, timeout=min(TTS_LLM_TIMEOUT, 10.0))
    models = data.get("data")
    if not isinstance(models, list) or not models:
        raise RuntimeError(f"No model found from {url}: {repr(data)[:1000]}")
    first = models[0]
    model_id = first.get("id") if isinstance(first, dict) else first
    if not model_id:
        raise RuntimeError(f"Invalid /models response from {url}: {repr(data)[:1000]}")
    _DISCOVERED_LLM_MODEL = str(model_id)
    return _DISCOVERED_LLM_MODEL


def _strip_code_fence(content: str) -> str:
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def _extract_json_array_or_object(content: str) -> Any:
    content = _strip_code_fence(content)
    try:
        return json.loads(content)
    except Exception:
        pass
    for left, right in (("[", "]"), ("{", "}")):
        start = content.find(left)
        end = content.rfind(right)
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
    raise ValueError("No valid JSON array/object found in LLM response")


def _build_llm_messages(text: str, prompt_text: Optional[str], has_reference_audio: bool, policy: Dict[str, Any]) -> List[Dict[str, str]]:
    speed_hint = _speed_to_control(policy["speed"])
    control_allowed = _control_instruction_allowed(policy)

    if has_reference_audio and policy.get("clone_mode") == "high_similarity":
        ref_hint = (
            "当前是 high_similarity / Ultimate 文本引导极致克隆：参考音频和参考文本会作为已说出的前文传入 prompt_wav_path + prompt_text；"
            "目标是优先复刻同一说话人的音色、口音、节奏、情绪底色和发声习惯。"
            "该模式下不要依赖括号 Control Instruction；情绪主要通过正文语义、标点、句内自然停连和少量官方方括号非语言标签体现。"
        )
    elif has_reference_audio:
        ref_hint = (
            "当前是可控克隆 style_control，但音色相似度优先级最高：必须保留参考音频的同一说话人、口音、发声习惯和音色稳定性；"
            "control 只能做轻量表演指导，重点写语气、语速、笑意、停连、重音、尾音和情绪强弱；"
            "不要写任何会改变声线/音色/年龄/性别/音高/声音质感的描述。"
        )
    else:
        ref_hint = (
            "当前没有参考音频，只能使用基础模型直接生成；建议正式使用时传入 reference audio。"
            "control 可以描述语气、语速、停连和情绪，但不要写内置音色身份。"
        )

    nonverbal_tag_guide = _format_nonverbal_tag_guide()

    system_prompt = f"""
你是 VoxCPM2 的 TTS 文本导演。你的任务是把输入文本拆成适合 VoxCPM2 生成的朗读片段，并为每段生成更有表现力的 Control Instruction。

你必须理解 VoxCPM2 的官方控制方式：
- Style Control 都是把自然语言 Control Instruction 放在目标文本开头的英文括号 () 里。
- Control Instruction 不是给人看的短标签，而是给模型的表演指导，应该像配音导演提示演员。
- 有参考音频时，参考音频决定 who speaks；control 只能调整 how to speak。
- 因此参考音频模式禁止重新描述声线、音色、年龄、性别、音高和声音质感。
- Ultimate/high_similarity 克隆更偏复刻参考音频，Control Instruction 会关闭；不要通过括号控制词改变音色。

非常重要：
- 不改变原文事实、人物关系、语义和信息量。
- text 字段是实际要朗读的正文，不能加入括号控制词、音色说明、英文风格词或元信息。
- 但 text 字段允许少量使用官方推荐的方括号非语言标签；只能插入 tag 本身，不要把备注写进 text。
- 可用标签和适用场景：
{nonverbal_tag_guide}
- 方括号标签不是稳定必触发的强指令，尤其 high_similarity/Ultimate 克隆会优先复刻参考音频；如果使用，必须放在句子边界，最好放在它要影响的短句前面，例如：[laughing] I never knew...
- 可以根据情绪来判断加不加方括号标签，有情绪的长段或短句通常可以自然插入 1 个，极强情绪可少量出现 2 个，但不能连续堆叠。
- 不要自造标签；不要使用 [angry]、[sad]、[whisper]、[pause]、[breath] 这类未列入白名单的标签。
- 每段可以明显更长，优先让同一段情绪和上下文连续生成，减少分段重启带来的音色差异。
- 只有当文本真的过长、话题转换明显、或单段会超过后端上限时才拆段；能自然承接的短句必须合并。
- 分段之间的停顿要像真人连续说话，整体紧凑，不能像每段单独录完再拼接。
- 请直接输出 JSON，不要 Markdown，不要解释。

输出格式必须是 JSON 对象：
{{
  "segments": [
    {{
      "text": "实际要朗读的文本，可少量包含官方方括号非语言标签",
      "emotion": "更具体的情绪判断，不要只写开心/难过",
      "control": "适合直接放进 VoxCPM2 括号里的自然语言表演指导",
      "pause_ms": 220,
      "cfg_value": 1.7,
      "inference_timesteps": 9,
      "use_control_instruction": true
    }}
  ]
}}

control 写法要求：
- control 必须有表现力，不能只写“自然、有感情”“平静”“开心”这种短词。
- 有参考音频时推荐结构：情绪底色 + 语气/语速 + 停连/重音/尾音 + 是否带笑意或叹息；不要写声音身份和音色质感。
- 没有参考音频时，只描述基础表演方式，不要写内置音色身份。
- 有参考音频时 control 不要改音色身份，但可以完整写出表演顺序：停顿位置、关键短语重音、情绪转折、语速变化、笑意/叹息、句尾处理都要保留；不要为了简短而丢掉这些信息。
- 如果一段里包含多个自然句，control 要按说话顺序写清楚，例如：先沉思慢速，在 But this 后稍停，随后转为惊叹，在 breathtaking 处轻微加重并留白。
- 中文文本优先输出中文 control；英文文本优先输出英文 control；中英混合时可用中文 control。
- control 不要出现括号，不要出现“读作”“朗读文本”“本段”等元话术。
- control 不要整句重复正文，但可以引用少量关键词来标明停顿或重音位置，例如 after But this、emphasize breathtaking、在“我明白了”处轻微加重。
- control 可以使用这些表达维度：
  - 情绪：克制的失望、压着火气、强装轻松、温柔安抚、惊讶后迅速收住、低落但不崩溃、兴奋但不换声线。
  - 音量/力度：音量偏低、轻声、音量略高、短促有力、关键短语轻微加重。
  - 节奏：语速稍慢、语速略快、节奏逐渐密集、停连自然、句尾轻轻落下。
  - 口吻：像回忆往事、像认真解释、像压抑争吵、像温柔哄人、像忍不住分享好消息。
  - 参考音频模式禁止使用：换声线、换音色、男声、女声、少女声、老人声、另一个人的声音、音高提高、声线更柔、年龄、性别等。
- 对话、小说、剧情类文本要更具体；说明文、技术文本也要给“清晰、专业、稳、关键术语加重”的 control。
- 如果文本情绪强烈，use_control_instruction 必须为 true。
- 如果只是纯信息播报，也建议 use_control_instruction=true，但 control 写成清晰稳重、口吻自然。

分段规则：
- 参考 VoxCPM 官方 split_paragraph 的思路，但本服务为了减少音色重启，会比官方默认更长；同时不要把整篇文章硬合成一个超长段。
- 核心原则：优先“每两个自然句”或“一个完整情绪意群”作为一段；连续铺陈、同一情绪推进、同一对象的描述尽量放在同一段。
- 中文普通叙述：每段目标 {TARGET_SEGMENT_CHARS_MIN} 到 {TARGET_SEGMENT_CHARS_MAX} 个汉字，最多 {MAX_SEGMENT_CHARS} 个汉字；通常 2 到 4 个中文自然句或一个完整情绪意群一段。后端会尊重你输出的 segments，不再替你合并。
- 中文强情绪台词：可以短一些，但通常不要少于 70 个汉字；连续短台词能合在同一段时再合并，不要一两句话就拆一段。
- 英文普通叙述：每段目标约 {TARGET_SEGMENT_WORDS_MIN} 到 {TARGET_SEGMENT_WORDS_MAX} 个英文单词，最多 {MAX_SEGMENT_WORDS} 个英文单词；通常 1 到 2 个英文自然句或一个完整情绪意群一段，长句可 1 句成段。后端会尊重你输出的 segments，不再替你合并。
- 英文短句或对话：不要输出少于 38 个英文单词的孤立短段，除非原文整段本身就是一句短句。
- 不要为了逗号、顿号、短连接词、轻微语气转折单独切段；逗号通常只作为段内停顿，不作为分段依据。
- 如果一个句子本身超过推荐长度，才在自然语义边界拆开；不要把正常长度的句子拆碎。

pause_ms 规则：
- 整体原则：宁可连贯一点，也不要段与段之间空太久。
- 句内承接、逗号、顿号、冒号：120 到 260。
- 普通句末，句号、英文句号、问号、感叹号：360 到 620。
- 情绪转折、省略号、段落切换、明显停顿：560 到 900。
- 除非文本本身有非常明显的大停顿，不要给 900 以上。
- 最后一段 pause_ms 必须为 0。

参数规则：
- cfg_value 用于控制文本/风格约束强度：
  - 有参考音频时，参考音频音色优先，cfg_value 建议 1.35 到 1.55；强情绪也不要超过 1.60。
  - 参考音频模式且 control 较长时，主动降低到 1.35 到 1.45，让音色更收敛到参考音频。
  - 参考音频模式且 control 很短时，可用 1.45 到 1.55。
  - 没有参考音频时，平静旁白、说明文、长句：1.5 到 1.7。
  - 没有参考音频时，普通对话、自然叙述：1.6 到 1.8。
  - 没有参考音频时，激动、愤怒、惊讶、强情绪短句：1.7 到 1.9。
  - 没有参考音频时，悲伤、低语、克制情绪：1.5 到 1.7，避免过度表演。

- inference_timesteps 用于控制生成精细程度和耗时：
  - 有参考音频时，为了减少每段之间的音色差异，统一控制在 7 到 9，通常用 8。
  - 没有参考音频时，普通短句：8 到 9。
  - 没有参考音频时，普通长句、旁白、信息密集句：8 到 10。
  - 没有参考音频时，强情绪、复杂语气、疑问/感叹明显的句子：9 到 11。
  - 没有参考音频时，很短的过渡句可以用 7 到 8。

Control Instruction 可用性：
- 当前 control_allowed={control_allowed}。
- 如果 control_allowed=false，仍然要输出完整的 emotion/control 供后端日志和后处理使用，use_control_instruction 必须为 false；control 里仍要保留停顿位置、关键词重音、情绪转折顺序、笑意/叹息等表演信息。后端不会再把 control 裁成短词，也不会二次合并分段；但不要指望括号控制词生效，应把情绪分析同时转化为标点、正文节奏和少量官方方括号标签。
- 如果 control_allowed=true，除非正文完全中性且极短，否则 use_control_instruction 设为 true。

参考/音色策略：
- {policy['style_prompt']}
- {ref_hint}
- 用户 speed={policy['speed']}，朗读倾向：{speed_hint}。
- reference_speech_rate={policy.get('reference_speech_rate', 'unknown')}
- reference_audio_energy={policy.get('reference_energy_label', 'unknown')}
- reference_audio_pause={policy.get('reference_pause_label', 'unknown')}
- reference_audio_style_hint={policy.get('reference_style_hint', '')}
- pause_scale={policy.get('pause_scale', 1.0)}
""".strip()

    user_obj = {
        "需要处理的文本": text,
        "参考音频对应文本": _clip_for_prompt(prompt_text, 800),
        "参考音频轻量分析": {
            "speech_rate": policy.get("reference_speech_rate"),
            "units_per_sec": policy.get("reference_units_per_sec"),
            "energy": policy.get("reference_energy_label"),
            "pause": policy.get("reference_pause_label"),
            "style_hint": policy.get("reference_style_hint"),
        },
        "是否有参考音频": has_reference_audio,
        "clone_mode": policy.get("clone_mode"),
        "control_instruction_allowed": control_allowed,
        "本次声音来源": {
            "voice_name": policy.get("voice_name"),
            "voice_source": policy.get("voice_source"),
        },
        "优秀输出示例": {
            "segments": [
                {
                    "text": "我真的没想到，你会这么做。",
                    "emotion": "失望压住了，表面冷静但有受伤感",
                    "control": "语气压低，语速稍慢，带一点受伤后的克制，句尾轻轻落下",
                    "pause_ms": 260,
                    "cfg_value": 1.55,
                    "inference_timesteps": 8,
                    "use_control_instruction": control_allowed,
                },
                {
                    "text": "[sigh] 算了，我不想再解释了。",
                    "emotion": "疲惫、放弃争辩、轻微叹息",
                    "control": "先轻轻叹气，随后低声说出，语气疲惫但不崩溃，尾音带一点无奈",
                    "pause_ms": 420,
                    "cfg_value": 1.55,
                    "inference_timesteps": 8,
                    "use_control_instruction": control_allowed,
                },
                {
                    "text": "The result is stable, but we still need to verify the edge cases before shipping.",
                    "emotion": "专业、谨慎、提醒风险",
                    "control": "clear and steady technical narration, slightly serious tone, moderate pace, emphasize stable and edge cases without sounding dramatic",
                    "pause_ms": 360,
                    "cfg_value": 1.55,
                    "inference_timesteps": 8,
                    "use_control_instruction": control_allowed,
                },
            ]
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
    ]


def _normalize_llm_segments(raw: Any, policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        if isinstance(raw.get("segments"), list):
            raw_segments = raw["segments"]
        elif isinstance(raw.get("data"), list):
            raw_segments = raw["data"]
        else:
            raw_segments = [raw]
    elif isinstance(raw, list):
        raw_segments = raw
    else:
        raise ValueError("LLM JSON must be an array or an object")

    segments: List[Dict[str, Any]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = _sanitize_tts_text_with_official_tags(str(item.get("text", "")).strip())
        if not text:
            continue
        control = _sanitize_control_instruction(item.get("control", ""))
        if _voice_identity_sensitive(policy):
            control = _sanitize_reference_control_instruction(control)
        if not control:
            control = "语气贴合正文情绪，语速自然，停连顺畅，关键短语轻微加重" if _voice_identity_sensitive(policy) else "自然清晰，口吻有真实交流感，语速中等，停连顺畅，避免机械平铺"
        raw_use_control = item.get("use_control_instruction", None)
        if raw_use_control is None:
            use_control = _control_instruction_allowed(policy) and bool(control)
        else:
            use_control = bool(raw_use_control) and _control_instruction_allowed(policy)
        segments.append({
            "text": text,
            "emotion": str(item.get("emotion", "自然")).strip() or "自然",
            "control": control,
            "pause_ms": _safe_int(item.get("pause_ms", policy["default_pause_ms"]), policy["default_pause_ms"], 0, MAX_PAUSE_MS),
            "cfg_value": _normalize_cfg_value_for_policy(item.get("cfg_value", DEFAULT_CFG_VALUE), DEFAULT_CFG_VALUE, policy),
            "inference_timesteps": _normalize_timesteps_for_policy(item.get("inference_timesteps", DEFAULT_INFERENCE_TIMESTEPS), DEFAULT_INFERENCE_TIMESTEPS, policy),
            "use_control_instruction": use_control,
        })
    if not segments:
        raise ValueError("LLM returned no usable segments")
    return segments


def _call_llm_for_tts_segments(text: str, prompt_text: Optional[str], has_reference_audio: bool, policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{TTS_LLM_BASE_URL}/chat/completions"
    llm_model = _resolve_llm_model()
    payload = {
        "model": llm_model,
        "messages": _build_llm_messages(text, prompt_text, has_reference_audio, policy),
        "temperature": TTS_LLM_TEMPERATURE,
        "max_tokens": TTS_LLM_MAX_TOKENS,
        "stream": False,
    }
    print("Calling LLM TTS director:", url, "model=", llm_model, flush=True)
    data = _post_json(url, payload, timeout=TTS_LLM_TIMEOUT)
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected LLM response schema: {repr(data)[:1000]}")
    print("LLM TTS director raw content:", content[:3000], flush=True)
    raw = _extract_json_array_or_object(content)
    return _normalize_llm_segments(raw, policy=policy)


def _scaled_range(lo: int, hi: int, scale: float) -> Tuple[int, int]:
    lo2 = int(round(float(lo) * float(scale)))
    hi2 = int(round(float(hi) * float(scale)))
    lo2 = max(50, min(lo2, 1800))
    hi2 = max(lo2, min(hi2, 2000))
    return lo2, hi2


def _clamp_pause_by_text(text: str, pause_ms: int, is_last: bool = False, emotion: str = "", policy: Optional[Dict[str, Any]] = None) -> int:
    if is_last:
        return 0
    text = (text or "").strip()
    emotion = (emotion or "").lower()
    pause_ms = _safe_int(pause_ms, DEFAULT_PAUSE_MS, 0, MAX_PAUSE_MS)
    scale = _safe_float(policy.get("pause_scale", 0.82), 0.82, 0.32, 1.05) if isinstance(policy, dict) else 0.82
    calm_words = ["calm", "reflective", "mesmerized", "awe", "gentle", "平静", "沉思", "着迷", "敬畏", "温柔", "旁白", "低语"]
    fast_words = ["excited", "urgent", "angry", "panic", "tense", "惊慌", "急促", "愤怒", "激动", "紧张"]
    is_calm = any(w in emotion for w in calm_words)
    is_fast = any(w in emotion for w in fast_words)

    if text.endswith(("，", ",", "、", "：", ":", "；", ";")):
        lo, hi = _scaled_range(COMMA_PAUSE_MIN, COMMA_PAUSE_MAX, scale)
        if is_calm:
            lo, hi = _scaled_range(170, 320, scale)
        if is_fast:
            lo, hi = _scaled_range(70, 170, scale)
        return max(lo, min(pause_ms, hi))

    if text.endswith(("……", "...", "…")):
        lo, hi = _scaled_range(LONG_PAUSE_MIN, LONG_PAUSE_MAX, scale)
        if is_calm:
            lo, hi = _scaled_range(560, 900, scale)
        if is_fast:
            lo, hi = _scaled_range(320, 620, scale)
        return max(lo, min(pause_ms, hi))

    if text.endswith(("。", ".", "？", "?", "！", "!")):
        lo, hi = _scaled_range(SENTENCE_PAUSE_MIN, SENTENCE_PAUSE_MAX, scale)
        if is_calm:
            lo, hi = _scaled_range(420, 700, scale)
        if is_fast:
            lo, hi = _scaled_range(180, 360, scale)
        return max(lo, min(pause_ms, hi))

    lo, hi = _scaled_range(180, 360, scale)
    if is_calm:
        lo, hi = _scaled_range(260, 500, scale)
    if is_fast:
        lo, hi = _scaled_range(70, 220, scale)
    return max(lo, min(pause_ms, hi))


def _finalize_segments_no_resplit(segments: List[Dict[str, Any]], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in segments:
        if not isinstance(raw, dict):
            continue
        text = _sanitize_tts_text_with_official_tags(str(raw.get("text", "")).strip())
        if not text:
            continue
        control = _sanitize_control_instruction(raw.get("control", ""))
        if _voice_identity_sensitive(policy):
            control = _sanitize_reference_control_instruction(control)
        if not control:
            control = "语气贴合正文情绪，语速自然，停连顺畅，关键短语轻微加重" if _voice_identity_sensitive(policy) else "自然清晰，口吻有真实交流感，语速中等，停连顺畅，避免机械平铺"
        raw_use_control = raw.get("use_control_instruction", None)
        if raw_use_control is None:
            use_control = _control_instruction_allowed(policy) and bool(control)
        else:
            use_control = bool(raw_use_control) and _control_instruction_allowed(policy)
        out.append({
            "text": text,
            "emotion": str(raw.get("emotion", "自然")).strip() or "自然",
            "control": control,
            "pause_ms": _safe_int(raw.get("pause_ms", policy.get("default_pause_ms", DEFAULT_PAUSE_MS)), policy.get("default_pause_ms", DEFAULT_PAUSE_MS), 0, MAX_PAUSE_MS),
            "cfg_value": _normalize_cfg_value_for_policy(raw.get("cfg_value", DEFAULT_CFG_VALUE), DEFAULT_CFG_VALUE, policy),
            "inference_timesteps": _normalize_timesteps_for_policy(raw.get("inference_timesteps", DEFAULT_INFERENCE_TIMESTEPS), DEFAULT_INFERENCE_TIMESTEPS, policy),
            "use_control_instruction": use_control,
        })

    # 尊重 LLM 输出的分段，不再二次合并或重切。
    # 后端只做字段清理、pause 兜底、参考音频模式下的 cfg/timesteps 收敛。
    for i, seg in enumerate(out):
        if _voice_identity_sensitive(policy):
            seg["control"] = _sanitize_reference_control_instruction(str(seg.get("control", "")), max_chars=MERGED_CONTROL_MAX_CHARS)
        seg["cfg_value"] = _normalize_cfg_value_for_policy(seg.get("cfg_value", DEFAULT_CFG_VALUE), DEFAULT_CFG_VALUE, policy)
        seg["inference_timesteps"] = _normalize_timesteps_for_policy(seg.get("inference_timesteps", DEFAULT_INFERENCE_TIMESTEPS), DEFAULT_INFERENCE_TIMESTEPS, policy)
        seg["pause_ms"] = _clamp_pause_by_text(
            str(seg.get("text", "")),
            _safe_int(seg.get("pause_ms", policy.get("default_pause_ms", DEFAULT_PAUSE_MS)), policy.get("default_pause_ms", DEFAULT_PAUSE_MS), 0, MAX_PAUSE_MS),
            is_last=(i == len(out) - 1),
            emotion=str(seg.get("emotion", "")),
            policy=policy,
        )
    return out


def _split_lang(text: str) -> str:
    return "zh" if _has_cjk(text) else "en"


def _split_len(text: str, lang: Optional[str] = None) -> int:
    text = re.sub(r"\s+", " ", (text or "").strip())
    lang = lang or _split_lang(text)
    if lang == "zh":
        return len(re.sub(r"\s+", "", text))
    return _english_word_count(text)


def _target_min_len(text: str, policy: Dict[str, Any]) -> int:
    if _split_lang(text) == "zh":
        return _safe_int(policy.get("target_segment_chars_min", TARGET_SEGMENT_CHARS_MIN), TARGET_SEGMENT_CHARS_MIN, 20, 420)
    return _safe_int(policy.get("target_segment_words_min", TARGET_SEGMENT_WORDS_MIN), TARGET_SEGMENT_WORDS_MIN, 10, 320)


def _target_max_len(text: str, policy: Dict[str, Any]) -> int:
    if _split_lang(text) == "zh":
        return _safe_int(policy.get("max_segment_chars", MAX_SEGMENT_CHARS), MAX_SEGMENT_CHARS, 40, 520)
    return _safe_int(policy.get("max_segment_words", MAX_SEGMENT_WORDS), MAX_SEGMENT_WORDS, 20, 420)


def _merge_short_len(text: str, policy: Dict[str, Any]) -> int:
    if _split_lang(text) == "zh":
        return _safe_int(policy.get("merge_short_chars", MERGE_SHORT_CHARS), MERGE_SHORT_CHARS, 4, 160)
    return _safe_int(policy.get("merge_short_words", MERGE_SHORT_WORDS), MERGE_SHORT_WORDS, 3, 160)


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


def _merge_field_text(left: str, right: str, default: str) -> str:
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right or default
    if not right or right == left:
        return left
    if left == default:
        return right
    if right == default:
        return left
    return f"{left}、{right}"


def _merge_control_text_ordered(left: Dict[str, Any], right: Dict[str, Any], has_reference: bool) -> str:
    """
    两段合并时，control 不能只取第一段。
    这里按说话顺序保留两段中不重复、且不改变音色的表演动作：
    停顿位置、关键词重音、情绪转折、语速变化、尾音处理、笑意/叹息都会保留。
    """
    clauses: List[str] = []
    seen: List[str] = []
    for raw in (str(left.get("control", "")), str(right.get("control", ""))):
        if has_reference:
            part_clauses = _reference_control_clauses(raw)
        else:
            clean = _sanitize_control_instruction(raw)
            part_clauses = [c.strip() for c in re.split(r"[，,；;。]+", clean) if c.strip()]
        for clause in part_clauses:
            norm = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", clause.lower())
            if not norm:
                continue
            duplicate = False
            for prev in seen:
                if norm == prev or norm in prev or prev in norm:
                    duplicate = True
                    break
            if duplicate:
                continue
            clauses.append(clause)
            seen.append(norm)

    if has_reference:
        control = _join_control_clauses(clauses, MERGED_CONTROL_MAX_CHARS)
    else:
        control = _truncate_control_by_clauses("，".join(clauses), MERGED_CONTROL_MAX_CHARS)
    return control or "语气贴合正文情绪，语速自然，停连顺畅，关键短语轻微加重"


def _merge_segment_pair(left: Dict[str, Any], right: Dict[str, Any], policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = dict(left)
    has_reference = bool(policy.get("has_reference_audio", False)) if isinstance(policy, dict) else False
    merged["text"] = _join_segment_text(str(left.get("text", "")), str(right.get("text", "")))
    merged["emotion"] = _merge_field_text(str(left.get("emotion", "")), str(right.get("emotion", "")), "自然")
    merged["control"] = _merge_control_text_ordered(left, right, has_reference=has_reference)
    # 两段合并后，段间停顿消失；保留右侧段尾停顿作为合并段的最终停顿。
    merged["pause_ms"] = right.get("pause_ms", left.get("pause_ms", DEFAULT_PAUSE_MS))
    merged["cfg_value"] = max(
        _safe_float(left.get("cfg_value", DEFAULT_CFG_VALUE), DEFAULT_CFG_VALUE, 1.3, 1.9),
        _safe_float(right.get("cfg_value", DEFAULT_CFG_VALUE), DEFAULT_CFG_VALUE, 1.3, 1.9),
    )
    merged["inference_timesteps"] = max(
        _safe_int(left.get("inference_timesteps", DEFAULT_INFERENCE_TIMESTEPS), DEFAULT_INFERENCE_TIMESTEPS, 4, 12),
        _safe_int(right.get("inference_timesteps", DEFAULT_INFERENCE_TIMESTEPS), DEFAULT_INFERENCE_TIMESTEPS, 4, 12),
    )
    merged["use_control_instruction"] = bool(left.get("use_control_instruction", False) or right.get("use_control_instruction", False))
    return merged

def _merge_short_segments_by_official_length(segments: List[Dict[str, Any]], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    合并策略按“两个自然句 / 一个完整意群”校准：
    - 当前一段还没达到合适长度时，优先向后合并；
    - 一旦上一段已经达到 target_min，就不再因为下一段偏短而继续吞段；
    - 最后一段如果带有非语言标签，保留独立性，避免笑声/叹气被埋进过长上下文。
    """
    if len(segments) <= 1:
        return segments

    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not out:
            out.append(seg)
            continue

        prev = out[-1]
        prev_text = str(prev.get("text", ""))
        cur_text = str(seg.get("text", ""))
        joined = _join_segment_text(prev_text, cur_text)
        prev_len = _split_len(prev_text)
        cur_len = _split_len(cur_text)
        joined_len = _split_len(joined)
        min_len = _target_min_len(joined, policy)
        max_len = _target_max_len(joined, policy)
        merge_len = _merge_short_len(joined, policy)

        prev_has_nonverbal = any(tag in prev_text for tag in OFFICIAL_NONVERBAL_TAGS)
        cur_has_nonverbal = any(tag in cur_text for tag in OFFICIAL_NONVERBAL_TAGS)

        # 只在“上一段还没到合适长度”时向后合并。
        # 这样会得到类似 1+2、3+4、最后一段保留的结构，避免一直吞成超长段。
        should_merge = (
            joined_len <= max_len
            and not prev_has_nonverbal
            and (
                prev_len < min_len
                or (prev_len <= merge_len and cur_len <= merge_len)
            )
        )

        # 当前段含非语言标签时，如果上一段已经基本够长，不合并，给标签更明确的触发空间。
        if cur_has_nonverbal and prev_len >= int(min_len * 0.85):
            should_merge = False

        if should_merge:
            out[-1] = _merge_segment_pair(prev, seg, policy)
        else:
            out.append(seg)

    # 尾段过短时才并回上一段；但带非语言标签的尾段默认保留，让 [laughing]/[sigh] 更容易生效。
    if len(out) >= 2:
        last = out[-1]
        prev = out[-2]
        last_text = str(last.get("text", ""))
        prev_text = str(prev.get("text", ""))
        joined = _join_segment_text(prev_text, last_text)
        last_len = _split_len(last_text)
        prev_len = _split_len(prev_text)
        joined_len = _split_len(joined)
        min_len = _target_min_len(joined, policy)
        last_has_nonverbal = any(tag in last_text for tag in OFFICIAL_NONVERBAL_TAGS)
        if (
            not last_has_nonverbal
            and prev_len < min_len
            and last_len <= _merge_short_len(joined, policy)
            and joined_len <= _target_max_len(joined, policy)
        ):
            out[-2] = _merge_segment_pair(prev, last, policy)
            out.pop()

    return out

def _sentence_units_for_fallback(text: str, lang: str) -> int:
    return _split_len(text, lang=lang)


def _split_long_unit_by_words(text: str, max_words: int) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    parts = []
    for i in range(0, len(words), max_words):
        part = " ".join(words[i:i + max_words]).strip()
        if part:
            parts.append(part)
    return parts


def _split_long_unit_by_chars(text: str, max_chars: int) -> List[str]:
    compact_len = len(re.sub(r"\s+", "", text))
    if compact_len <= max_chars:
        return [text]
    parts: List[str] = []
    cur = ""
    cur_len = 0
    for ch in text:
        ch_len = 0 if ch.isspace() else 1
        if cur and cur_len + ch_len > max_chars:
            parts.append(cur.strip())
            cur = ch
            cur_len = ch_len
        else:
            cur += ch
            cur_len += ch_len
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _split_paragraph_official_like(text: str, lang: str, token_max_n: int, token_min_n: int, merge_len: int) -> List[str]:
    """
    参考 VoxCPM 官方 TextNormalizer.split_paragraph：
    - 中文长度按字符数；
    - 英文这里按单词数近似 tokenizer 长度；
    - 默认不按逗号切分，避免段落过碎；
    - 最后过短的句子并回上一段。
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    if lang == "zh":
        punc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        punc = [".", "?", "!", ";", ":"]

    st = 0
    units: List[str] = []
    for i, ch in enumerate(text):
        if ch in punc:
            if len(text[st:i].strip()) > 0:
                unit = text[st:i + 1].strip()
                if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                    unit += text[i + 1]
                    st = i + 2
                else:
                    st = i + 1
                units.append(unit)

    rest = text[st:].strip()
    if rest:
        units.append(rest)

    if not units:
        units = [text + ("。" if lang == "zh" else ".")]

    expanded: List[str] = []
    for unit in units:
        if _sentence_units_for_fallback(unit, lang) <= token_max_n:
            expanded.append(unit)
        elif lang == "zh":
            expanded.extend(_split_long_unit_by_chars(unit, token_max_n))
        else:
            expanded.extend(_split_long_unit_by_words(unit, token_max_n))

    final_units: List[str] = []
    cur = ""
    for unit in expanded:
        candidate = _join_segment_text(cur, unit)
        if _sentence_units_for_fallback(candidate, lang) > token_max_n and _sentence_units_for_fallback(cur, lang) > token_min_n:
            final_units.append(cur.strip())
            cur = unit
        else:
            cur = candidate

    if cur.strip():
        if final_units and _sentence_units_for_fallback(cur, lang) < merge_len:
            final_units[-1] = _join_segment_text(final_units[-1], cur)
        else:
            final_units.append(cur.strip())

    return [u for u in final_units if u.strip()]


def _fallback_split_text(text: str, default_pause_ms: int, max_chars: int, max_segments: int, policy: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    policy = policy or {}
    lang = _split_lang(text)
    if lang == "zh":
        token_min_n = _safe_int(policy.get("target_segment_chars_min", TARGET_SEGMENT_CHARS_MIN), TARGET_SEGMENT_CHARS_MIN, 20, 420)
        token_max_n = _safe_int(policy.get("target_segment_chars_max", TARGET_SEGMENT_CHARS_MAX), TARGET_SEGMENT_CHARS_MAX, token_min_n, 480)
        token_max_n = min(token_max_n, _safe_int(policy.get("max_segment_chars", MAX_SEGMENT_CHARS), MAX_SEGMENT_CHARS, token_max_n, 520))
        merge_len = _safe_int(policy.get("merge_short_chars", MERGE_SHORT_CHARS), MERGE_SHORT_CHARS, 4, 160)
    else:
        token_min_n = _safe_int(policy.get("target_segment_words_min", TARGET_SEGMENT_WORDS_MIN), TARGET_SEGMENT_WORDS_MIN, 10, 320)
        token_max_n = _safe_int(policy.get("target_segment_words_max", TARGET_SEGMENT_WORDS_MAX), TARGET_SEGMENT_WORDS_MAX, token_min_n, 380)
        token_max_n = min(token_max_n, _safe_int(policy.get("max_segment_words", MAX_SEGMENT_WORDS), MAX_SEGMENT_WORDS, token_max_n, 420))
        merge_len = _safe_int(policy.get("merge_short_words", MERGE_SHORT_WORDS), MERGE_SHORT_WORDS, 3, 160)

    parts = _split_paragraph_official_like(text, lang=lang, token_max_n=token_max_n, token_min_n=token_min_n, merge_len=merge_len)

    def _new_seg(seg_text: str, pause_ms: int = default_pause_ms) -> Dict[str, Any]:
        return {
            "text": seg_text.strip(),
            "emotion": "自然",
            "control": "自然清晰，口吻有真实交流感，语速中等，关键内容略加重，停连顺畅",
            "pause_ms": pause_ms,
            "cfg_value": DEFAULT_CFG_VALUE,
            "inference_timesteps": DEFAULT_INFERENCE_TIMESTEPS,
            "use_control_instruction": _control_instruction_allowed(policy),
        }

    segments = [_new_seg(p) for p in parts[:max_segments] if p.strip()]
    if not segments:
        segments = [_new_seg(text, pause_ms=0)]
    return segments[:max_segments]


def _prepare_tts_segments(text: str, prompt_text: Optional[str], has_reference_audio: bool, policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    if not policy["segment"]:
        return [{
            "text": text.strip(),
            "emotion": "自然",
            "control": "自然清晰，口吻有真实交流感，语速中等，关键内容略加重，停连顺畅",
            "pause_ms": 0,
            "cfg_value": DEFAULT_CFG_VALUE,
            "inference_timesteps": DEFAULT_INFERENCE_TIMESTEPS,
            "use_control_instruction": _control_instruction_allowed(policy),
        }]

    if not TTS_LLM_ENABLE or not policy["llm_preprocess"]:
        segments = _fallback_split_text(text, policy["default_pause_ms"], policy["max_segment_chars"], 200, policy)
        segments = _finalize_segments_no_resplit(segments, policy)
        print("Fallback TTS segments:\n" + _json_pretty(segments), flush=True)
        return segments

    try:
        segments = _call_llm_for_tts_segments(text, prompt_text, has_reference_audio, policy)
        segments = _finalize_segments_no_resplit(segments, policy)
        print("Prepared TTS segments:\n" + _json_pretty(segments), flush=True)
        return segments
    except Exception as e:
        print("LLM TTS director failed:", repr(e), flush=True)
        traceback.print_exc()
        if not TTS_LLM_FALLBACK_ON_ERROR:
            raise HTTPException(status_code=500, detail=f"LLM TTS director failed: {repr(e)}")
        segments = _fallback_split_text(text, policy["default_pause_ms"], policy["max_segment_chars"], 200, policy)
        segments = _finalize_segments_no_resplit(segments, policy)
        return segments


# ============================================================
# VoxCPM 生成
# ============================================================

def _strip_accidental_control_prefix(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\s*[\(（][^()（）]{1,420}[\)）]\s*", "", text)
    return text.strip()



def _apply_nonverbal_trigger_cues(text: str, policy: Dict[str, Any]) -> str:
    """
    [laughing] 是官方非语言标签，但在 high_similarity/Ultimate 克隆里常被当成弱提示。
    为了提高可听见的触发概率，把标签放到句首，并补一个很短的可朗读笑声提示。
    这会比纯标签更稳定；代价是可能听到轻微的 "Haha," / "哈哈，"。
    """
    text = str(text or "").strip()
    if not text or not policy.get("enable_laugh_text_cue", ENABLE_LAUGH_TEXT_CUE):
        return text
    if "[laughing]" not in text:
        return text

    def repl(match: re.Match) -> str:
        tail = match.group(1) or ""
        after = text[match.end():match.end() + 80]
        cue = "哈哈，" if _has_cjk(after) else "Haha,"
        # 如果后面已经有 haha/哈哈，就不重复补。
        if re.match(r"\s*(haha|ha ha|哈哈|呵呵|嘿嘿)\b", after, flags=re.I):
            return f"{tail} [laughing]"
        return f"{tail} [laughing] {cue}"

    # 优先处理句子边界后的 [laughing]，比如 ". [laughing] I never..."
    text = re.sub(r"(^|[。！？!?\.]+\s*)\[laughing\]\s*", repl, text)
    # 如果 LLM 放在逗号或普通空格后，也保底补 cue。
    text = re.sub(
        r"\[laughing\]\s*(?!\s*(?:Haha|Ha ha|哈哈|呵呵|嘿嘿)\b)",
        lambda m: "[laughing] " + ("哈哈，" if _has_cjk(text[m.end():m.end() + 80]) else "Haha,"),
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _segment_to_generation_text(seg: Dict[str, Any], policy: Dict[str, Any]) -> str:
    raw_text = str(seg.get("text", "")).strip()
    clean_text = _strip_accidental_control_prefix(raw_text)
    clean_text = _sanitize_tts_text_with_official_tags(clean_text)
    clean_text = _apply_nonverbal_trigger_cues(clean_text, policy)
    if not clean_text:
        return ""

    # high_similarity / Ultimate 默认不拼入括号控制词，避免破坏音频续写式复刻。
    if seg.get("use_control_instruction", False) and _control_instruction_allowed(policy):
        segment_control = _sanitize_reference_control_instruction(seg.get("control", ""))
        if segment_control:
            merged_control = f"{REFERENCE_CONTROL_ANCHOR}；{segment_control}"
            merged_control = _truncate_control_by_clauses(
                merged_control,
                len(REFERENCE_CONTROL_ANCHOR) + REFERENCE_CONTROL_MAX_CHARS + 2,
            )
            if merged_control:
                return f"({merged_control}){clean_text}"

    return clean_text


def _build_generate_kwargs(text: str, cfg_value: float, inference_timesteps: int, reference_wav_path: Optional[str], prompt_text: Optional[str], policy: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "text": text,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "normalize": policy.get("normalize", DEFAULT_NORMALIZE),
        "denoise": False,
        "retry_badcase": policy.get("retry_badcase", DEFAULT_RETRY_BADCASE),
        "retry_badcase_max_times": policy.get("retry_badcase_max_times", DEFAULT_RETRY_BADCASE_MAX_TIMES),
        "retry_badcase_ratio_threshold": policy.get("retry_badcase_ratio_threshold", DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD),
        "min_len": policy.get("min_len", DEFAULT_MIN_LEN),
        "max_len": policy.get("max_len", DEFAULT_MAX_LEN),
    }
    if reference_wav_path:
        kwargs["reference_wav_path"] = reference_wav_path
        use_hifi = bool(
            prompt_text and prompt_text.strip()
            and policy.get("prompt_text_is_exact", True)
            and policy.get("clone_mode") == "high_similarity"
        )
        if use_hifi:
            kwargs["prompt_wav_path"] = reference_wav_path
            kwargs["prompt_text"] = prompt_text.strip()
    return kwargs


def _call_model_generate_with_fallback(generate_kwargs: Dict[str, Any]) -> np.ndarray:
    try:
        result = model.generate(**generate_kwargs)
        return _extract_wav_from_result(result)
    except TypeError as e:
        msg = str(e)
        print("VoxCPM generate TypeError, trying fallback:", msg, flush=True)
        fallback_kwargs = dict(generate_kwargs)
        removed = []
        advanced_keys = [
            "normalize", "denoise", "retry_badcase", "retry_badcase_max_times",
            "retry_badcase_ratio_threshold", "min_len", "max_len",
        ]
        if "unexpected keyword" in msg or "got an unexpected" in msg:
            for key in advanced_keys:
                if key in fallback_kwargs:
                    fallback_kwargs.pop(key, None)
                    removed.append(key)
        if "prompt_wav_path" in msg or "prompt_text" in msg:
            for key in ("prompt_wav_path", "prompt_text"):
                if key in fallback_kwargs:
                    fallback_kwargs.pop(key, None)
                    removed.append(key)
        if not removed:
            raise
        print("Retry VoxCPM generate without keys:", removed, flush=True)
        result = model.generate(**fallback_kwargs)
        return _extract_wav_from_result(result)


def _generate_audio(text: str, cfg_value: float, inference_timesteps: int, reference_wav_path: Optional[str] = None, prompt_text: Optional[str] = None, policy: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, int]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    policy = policy or {}
    cfg_value = _normalize_cfg_value_for_policy(cfg_value, DEFAULT_CFG_VALUE, policy)
    inference_timesteps = _normalize_timesteps_for_policy(inference_timesteps, DEFAULT_INFERENCE_TIMESTEPS, policy)
    generate_kwargs = _build_generate_kwargs(text, cfg_value, inference_timesteps, reference_wav_path, prompt_text, policy)
    print("[NON-STREAM] generate_kwargs keys =", list(generate_kwargs.keys()), flush=True)
    print("[NON-STREAM] generate text preview =", text[:300], flush=True)
    wav = _call_model_generate_with_fallback(generate_kwargs)
    return wav, _get_sample_rate()


def _iter_model_generate_streaming_with_fallback(generate_kwargs: Dict[str, Any]) -> Iterator[np.ndarray]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if hasattr(model, "generate_streaming"):
        try:
            for item in model.generate_streaming(**generate_kwargs):
                chunk = _extract_wav_from_result(item)
                if chunk.size > 0:
                    yield chunk
            return
        except TypeError as e:
            msg = str(e)
            print("VoxCPM generate_streaming TypeError, trying fallback:", msg, flush=True)
            fallback_kwargs = dict(generate_kwargs)
            removed = []
            advanced_keys = [
                "normalize", "denoise", "retry_badcase", "retry_badcase_max_times",
                "retry_badcase_ratio_threshold", "min_len", "max_len",
            ]
            if "unexpected keyword" in msg or "got an unexpected" in msg:
                for key in advanced_keys:
                    if key in fallback_kwargs:
                        fallback_kwargs.pop(key, None)
                        removed.append(key)
            if ("prompt_wav_path" in msg or "prompt_text" in msg) and ("prompt_wav_path" in generate_kwargs or "prompt_text" in generate_kwargs):
                # 部分 VoxCPM2 版本的 generate_streaming 可能不支持 Ultimate 克隆参数。
                # 这种情况下不能删掉 prompt_wav_path/prompt_text，否则会退化成普通克隆；
                # 改为非流式 high_similarity 生成，再切块返回，优先保证音色相似度。
                print("Streaming does not support prompt_wav_path/prompt_text; fallback to non-streaming generate with Ultimate cloning args.", flush=True)
                wav = _call_model_generate_with_fallback(generate_kwargs)
                sample_rate = _get_sample_rate()
                chunk_samples = max(1, int(sample_rate * STREAM_CHUNK_MS / 1000))
                for i in range(0, len(wav), chunk_samples):
                    yield wav[i:i + chunk_samples]
                return
            if "prompt_wav_path" in msg or "prompt_text" in msg:
                for key in ("prompt_wav_path", "prompt_text"):
                    if key in fallback_kwargs:
                        fallback_kwargs.pop(key, None)
                        removed.append(key)
            if removed:
                for item in model.generate_streaming(**fallback_kwargs):
                    chunk = _extract_wav_from_result(item)
                    if chunk.size > 0:
                        yield chunk
                return
            raise
    wav = _call_model_generate_with_fallback(generate_kwargs)
    sample_rate = _get_sample_rate()
    chunk_samples = max(1, int(sample_rate * STREAM_CHUNK_MS / 1000))
    for i in range(0, len(wav), chunk_samples):
        yield wav[i:i + chunk_samples]


def _generate_audio_streaming(text: str, cfg_value: float, inference_timesteps: int, reference_wav_path: Optional[str], prompt_text: Optional[str], policy: Dict[str, Any]) -> Iterator[Tuple[np.ndarray, int]]:
    cfg_value = _normalize_cfg_value_for_policy(cfg_value, DEFAULT_CFG_VALUE, policy)
    inference_timesteps = _normalize_timesteps_for_policy(inference_timesteps, DEFAULT_INFERENCE_TIMESTEPS, policy)
    generate_kwargs = _build_generate_kwargs(text, cfg_value, inference_timesteps, reference_wav_path, prompt_text, policy)
    print("[STREAM] generate_kwargs keys =", list(generate_kwargs.keys()), flush=True)
    print("[STREAM] generate text preview =", text[:300], flush=True)
    sample_rate = _get_sample_rate()
    for chunk in _iter_model_generate_streaming_with_fallback(generate_kwargs):
        yield np.asarray(chunk, dtype=np.float32).reshape(-1), sample_rate


# ============================================================
# 音频后处理、拼接、编码、保存
# ============================================================

def _trim_silence_edges(wav: np.ndarray, sample_rate: int, threshold: float = 0.0018, keep_ms: int = 20) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return wav
    idx = np.where(np.abs(wav) > threshold)[0]
    if idx.size == 0:
        return wav
    keep = int(sample_rate * keep_ms / 1000)
    start = max(0, int(idx[0]) - keep)
    end = min(len(wav), int(idx[-1]) + keep)
    return wav[start:end] if end > start else wav


def _fade_edges(wav: np.ndarray, sample_rate: int, fade_ms: int = FADE_MS) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1).copy()
    if fade_ms <= 0 or wav.size < 4:
        return wav
    n = min(int(sample_rate * fade_ms / 1000), wav.size // 2)
    if n <= 1:
        return wav
    wav[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return wav


def _postprocess_segment_wav(wav: np.ndarray, sample_rate: int, policy: Dict[str, Any]) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    if wav.size == 0:
        return wav
    wav = wav - float(np.mean(wav))
    if policy.get("trim_silence", TRIM_SILENCE):
        wav = _trim_silence_edges(wav, sample_rate)
    wav = _fade_edges(wav, sample_rate, int(policy.get("fade_ms", FADE_MS)))
    return np.clip(wav, -1.0, 1.0)


def _crossfade_pair(left: np.ndarray, right: np.ndarray, sample_rate: int, crossfade_ms: int) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    n = min(int(sample_rate * crossfade_ms / 1000), len(left) // 2, len(right) // 2)
    if n <= 1:
        return np.concatenate([left, right])
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    mixed = left[-n:] * fade_out + right[:n] * fade_in
    return np.concatenate([left[:-n], mixed, right[n:]])


def _concat_generated_segments(generated: List[Tuple[np.ndarray, int]], sample_rate: int, policy: Dict[str, Any]) -> np.ndarray:
    if not generated:
        raise HTTPException(status_code=400, detail="no audio segments generated")
    out = np.asarray(generated[0][0], dtype=np.float32).reshape(-1)
    for i in range(1, len(generated)):
        prev_pause_ms = _safe_int(generated[i - 1][1], DEFAULT_PAUSE_MS, 0, MAX_PAUSE_MS)
        cur = np.asarray(generated[i][0], dtype=np.float32).reshape(-1)
        if prev_pause_ms > 0:
            out = np.concatenate([out, np.zeros(int(sample_rate * prev_pause_ms / 1000), dtype=np.float32), cur])
        else:
            out = _crossfade_pair(out, cur, sample_rate, int(policy.get("crossfade_ms", CROSSFADE_MS)))
    return np.clip(out, -1.0, 1.0)


def _float_to_pcm16_bytes(wav: np.ndarray) -> bytes:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


def _wav_stream_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE", b"fmt ", 16, 1, channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b"data", 0xFFFFFFFF,
    )


def _wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, wav, sample_rate, format="WAV")
    audio_buffer.seek(0)
    return audio_buffer.read()


def _mp3_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    tmp_wav = os.path.join(tempfile.gettempdir(), f"voxcpm_out_{uuid.uuid4().hex}.wav")
    tmp_mp3 = os.path.join(tempfile.gettempdir(), f"voxcpm_out_{uuid.uuid4().hex}.mp3")
    try:
        sf.write(tmp_wav, wav, sample_rate, format="WAV")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", tmp_wav, "-codec:a", "libmp3lame", "-b:a", "192k", tmp_mp3]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with open(tmp_mp3, "rb") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found. Install ffmpeg to support mp3.")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
        raise HTTPException(status_code=500, detail=f"failed to encode mp3: {stderr[:1000]}")
    finally:
        for p in (tmp_wav, tmp_mp3):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def _next_output_path(response_format: str) -> str:
    fmt = (response_format or "wav").lower().strip()
    if fmt not in ("mp3", "wav"):
        fmt = "wav"
    date_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y-%m-%d"))
    with _SAVE_LOCK:
        os.makedirs(date_dir, exist_ok=True)
        max_num = 0
        for name in os.listdir(date_dir):
            stem = os.path.splitext(name)[0]
            if re.fullmatch(r"\d{5}", stem):
                try:
                    max_num = max(max_num, int(stem))
                except Exception:
                    pass
        return os.path.join(date_dir, f"{max_num + 1:05d}.{fmt}")


def _save_generated_audio(wav: np.ndarray, sample_rate: int, response_format: str) -> Optional[str]:
    try:
        fmt = (response_format or "wav").lower().strip()
        if fmt not in ("mp3", "wav"):
            fmt = "wav"
        out_path = _next_output_path(fmt)
        tmp_path = out_path + ".tmp"
        data = _mp3_bytes(wav, sample_rate) if fmt == "mp3" else _wav_bytes(wav, sample_rate)
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, out_path)
        print("Saved generated audio:", out_path, flush=True)
        return out_path
    except Exception as e:
        print("Failed to save generated audio:", repr(e), flush=True)
        traceback.print_exc()
        return None


def _audio_response(wav: np.ndarray, sample_rate: int, response_format: str = "wav") -> Response:
    fmt = (response_format or "wav").lower().strip()
    save_path = _save_generated_audio(wav, sample_rate, fmt)
    if fmt == "mp3":
        content = _mp3_bytes(wav, sample_rate)
        headers = {"Content-Disposition": 'attachment; filename="speech.mp3"'}
        if save_path:
            headers["X-Saved-Audio-Path"] = save_path
        return Response(content=content, media_type="audio/mpeg", headers=headers)
    content = _wav_bytes(wav, sample_rate)
    headers = {"Content-Disposition": 'attachment; filename="speech.wav"'}
    if save_path:
        headers["X-Saved-Audio-Path"] = save_path
    return Response(content=content, media_type="audio/wav", headers=headers)


# ============================================================
# 非流式 / 流式生成
# ============================================================

def _generate_segments_audio(segments: List[Dict[str, Any]], fallback_cfg_value: float, fallback_inference_timesteps: int, reference_wav_path: Optional[str], prompt_text: Optional[str], policy: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    generated: List[Tuple[np.ndarray, int]] = []
    sample_rate: Optional[int] = None
    with _MODEL_LOCK:
        for idx, seg in enumerate(segments):
            gen_text = _segment_to_generation_text(seg, policy)
            if not gen_text:
                continue
            cfg_value = _adaptive_cfg_value_for_segment(seg.get("cfg_value", fallback_cfg_value), fallback_cfg_value, seg, policy)
            inference_timesteps = _normalize_timesteps_for_policy(seg.get("inference_timesteps", fallback_inference_timesteps), fallback_inference_timesteps, policy)
            print(
                f"Generating segment {idx + 1}/{len(segments)}: voice={policy.get('voice_name')} "
                f"source={policy.get('voice_source')} emotion={seg.get('emotion')} pause_ms={seg.get('pause_ms')} "
                f"cfg={cfg_value} steps={inference_timesteps}",
                flush=True,
            )
            wav, sr = _generate_audio(gen_text, cfg_value, inference_timesteps, reference_wav_path, prompt_text, policy)
            wav = _postprocess_segment_wav(wav, sr, policy)
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                raise RuntimeError(f"sample_rate changed across segments: {sample_rate} vs {sr}")
            pause_ms = _safe_int(seg.get("pause_ms", 0), DEFAULT_PAUSE_MS, 0, MAX_PAUSE_MS)
            generated.append((wav, pause_ms))
    if not generated:
        raise HTTPException(status_code=400, detail="no audio segments generated")
    final_wav = _concat_generated_segments(generated, sample_rate or 48000, policy)
    return final_wav, sample_rate or 48000


def _stream_float_chunks(segments: List[Dict[str, Any]], fallback_cfg_value: float, fallback_inference_timesteps: int, reference_wav_path: Optional[str], prompt_text: Optional[str], policy: Dict[str, Any]) -> Iterator[Tuple[np.ndarray, int]]:
    sample_rate = _get_sample_rate()
    with _MODEL_LOCK:
        for idx, seg in enumerate(segments):
            gen_text = _segment_to_generation_text(seg, policy)
            if not gen_text:
                continue
            cfg_value = _adaptive_cfg_value_for_segment(seg.get("cfg_value", fallback_cfg_value), fallback_cfg_value, seg, policy)
            inference_timesteps = _normalize_timesteps_for_policy(seg.get("inference_timesteps", fallback_inference_timesteps), fallback_inference_timesteps, policy)
            print(
                f"Streaming segment {idx + 1}/{len(segments)}: voice={policy.get('voice_name')} "
                f"source={policy.get('voice_source')} emotion={seg.get('emotion')} pause_ms={seg.get('pause_ms')} "
                f"cfg={cfg_value} steps={inference_timesteps}",
                flush=True,
            )
            first_chunk = True
            for chunk, sr in _generate_audio_streaming(gen_text, cfg_value, inference_timesteps, reference_wav_path, prompt_text, policy):
                if sr != sample_rate:
                    raise RuntimeError(f"sample_rate changed across stream: {sample_rate} vs {sr}")
                chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if chunk.size == 0:
                    continue
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                chunk = np.clip(chunk, -1.0, 1.0)
                if first_chunk:
                    chunk = _fade_edges(chunk, sample_rate, min(FADE_MS, 4))
                    first_chunk = False
                yield chunk, sample_rate
            pause_ms = _safe_int(seg.get("pause_ms", 0), DEFAULT_PAUSE_MS, 0, MAX_PAUSE_MS)
            if pause_ms > 0 and idx < len(segments) - 1:
                yield np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32), sample_rate


def _wav_stream_iterator(segments: List[Dict[str, Any]], fallback_cfg_value: float, fallback_inference_timesteps: int, reference_wav_path: Optional[str], prompt_text: Optional[str], policy: Dict[str, Any], response_format_for_save: str, cleanup_files: List[str]):
    collected: List[np.ndarray] = []
    sample_rate = _get_sample_rate()
    try:
        yield _wav_stream_header(sample_rate)
        for chunk, sr in _stream_float_chunks(segments, fallback_cfg_value, fallback_inference_timesteps, reference_wav_path, prompt_text, policy):
            sample_rate = sr
            collected.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            yield _float_to_pcm16_bytes(chunk)
        if collected:
            _save_generated_audio(np.concatenate(collected), sample_rate, response_format_for_save)
    except Exception as e:
        print("WAV stream generator failed:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        _cleanup_files(cleanup_files)


def _mp3_stream_iterator(segments: List[Dict[str, Any]], fallback_cfg_value: float, fallback_inference_timesteps: int, reference_wav_path: Optional[str], prompt_text: Optional[str], policy: Dict[str, Any], cleanup_files: List[str]):
    sample_rate = _get_sample_rate()
    collected: List[np.ndarray] = []
    err_box: List[BaseException] = []
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "192k", "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    except FileNotFoundError:
        _cleanup_files(cleanup_files)
        raise HTTPException(status_code=500, detail="ffmpeg not found. Install ffmpeg to support mp3 streaming.")

    def producer():
        try:
            assert proc.stdin is not None
            for chunk, sr in _stream_float_chunks(segments, fallback_cfg_value, fallback_inference_timesteps, reference_wav_path, prompt_text, policy):
                if sr != sample_rate:
                    raise RuntimeError(f"sample_rate changed across stream: {sample_rate} vs {sr}")
                chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if chunk.size == 0:
                    continue
                collected.append(chunk)
                proc.stdin.write(_float_to_pcm16_bytes(chunk))
            try:
                proc.stdin.close()
            except Exception:
                pass
            if collected:
                _save_generated_audio(np.concatenate(collected), sample_rate, "mp3")
        except BaseException as e:
            err_box.append(e)
            print("MP3 stream producer failed:", repr(e), flush=True)
            traceback.print_exc()
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    try:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read(4096)
            if data:
                yield data
                continue
            if proc.poll() is not None:
                break
            time.sleep(0.01)
        thread.join(timeout=5)
        if err_box:
            print("MP3 stream finished with producer error:", repr(err_box[0]), flush=True)
    except Exception as e:
        print("MP3 stream generator failed:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        _cleanup_files(cleanup_files)


def _streaming_audio_response(segments: List[Dict[str, Any]], fallback_cfg_value: float, fallback_inference_timesteps: int, reference_wav_path: Optional[str], prompt_text: Optional[str], policy: Dict[str, Any], response_format: str, cleanup_files: List[str]) -> StreamingResponse:
    fmt = (response_format or "wav").lower().strip()
    if fmt == "mp3":
        return StreamingResponse(
            _mp3_stream_iterator(segments, fallback_cfg_value, fallback_inference_timesteps, reference_wav_path, prompt_text, policy, cleanup_files),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="speech.mp3"', "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return StreamingResponse(
        _wav_stream_iterator(segments, fallback_cfg_value, fallback_inference_timesteps, reference_wav_path, prompt_text, policy, "wav", cleanup_files),
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"', "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# API
# ============================================================

def _summarize_openai_request(req: OpenAITTSRequest) -> Dict[str, Any]:
    extra = _get_extra(req)
    refs = req.references or extra.get("references")
    ref_summary = []
    if isinstance(refs, list):
        for idx, ref in enumerate(refs[:3]):
            if not isinstance(ref, dict):
                ref_summary.append({"index": idx, "type": type(ref).__name__})
                continue
            audio = ref.get("audio")
            audio_summary = {"present": bool(audio), "type": type(audio).__name__}
            if isinstance(audio, str):
                audio_summary = {"present": True, "prefix": audio[:48], "chars": len(audio)}
            ref_text = ref.get("text")
            ref_summary.append({
                "index": idx,
                "text_preview": str(ref_text or "")[:200],
                "text_chars": len(str(ref_text or "")),
                "audio": audio_summary,
            })
    return {
        "model": req.model,
        "input_preview": (req.input or "")[:300],
        "input_chars": len(req.input or ""),
        "voice": req.voice,
        "speed": req.speed,
        "response_format": req.response_format,
        "stream_top_level_raw": req.stream,
        "extra_body_keys": sorted(list(extra.keys())) if isinstance(extra, dict) else [],
        "extra_body_stream_raw": extra.get("stream") if isinstance(extra, dict) else None,
        "reference_voice": req.reference_voice,
        "speaker": req.speaker,
        "cfg_value": req.cfg_value,
        "inference_timesteps": req.inference_timesteps,
        "references_count": len(refs) if isinstance(refs, list) else 0,
        "references": ref_summary,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_name": OPENAI_MODEL_NAME,
        "load_denoiser": LOAD_DENOISER,
        "llm": {
            "enable": TTS_LLM_ENABLE,
            "base_url": TTS_LLM_BASE_URL,
            "model": TTS_LLM_MODEL or _DISCOVERED_LLM_MODEL or "auto",
            "timeout": TTS_LLM_TIMEOUT,
            "fallback_on_error": TTS_LLM_FALLBACK_ON_ERROR,
        },
        "reference_audio": {
            "voice_source": "request reference audio",
            "default_clone_mode": "high_similarity when reference text is provided",
        },
        "storage": {"output_dir": OUTPUT_DIR, "layout": "YYYY-MM-DD/00001.wav or 00001.mp3"},
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": OPENAI_MODEL_NAME, "object": "model", "owned_by": "local"}]}


@app.post("/debug/tts_plan")
def debug_tts_plan(req: TTSPlanDebugRequest):
    fake_req = OpenAITTSRequest(
        model=OPENAI_MODEL_NAME,
        input=req.text,
        voice=req.voice,
        speed=req.speed,
        response_format="wav",
        extra_body=req.extra_body or {},
    )
    policy = _get_backend_tts_policy(fake_req, reference_wav_path="dummy.wav" if req.has_reference_audio else None, prompt_text=req.prompt_text)
    segments = _prepare_tts_segments(req.text, req.prompt_text, req.has_reference_audio, policy)
    return {"policy": policy, "segments": segments}


@app.post("/tts")
def tts(req: TTSRequest):
    try:
        fake_req = OpenAITTSRequest(
            model=OPENAI_MODEL_NAME,
            input=req.text,
            voice=req.voice or DEFAULT_VOICE_NAME,
            speed=1.0,
            response_format="wav",
            extra_body={"segment": False},
        )
        policy = _get_backend_tts_policy(fake_req, reference_wav_path=None, prompt_text=None)
        print("/tts policy:", _json_preview({k: policy.get(k) for k in ("voice_name", "voice_source", "clone_mode")}), flush=True)
        with _MODEL_LOCK:
            gen_text = _segment_to_generation_text({"text": req.text}, policy)
            wav, sample_rate = _generate_audio(gen_text, req.cfg_value, req.inference_timesteps, policy=policy)
        save_path = _save_generated_audio(wav, sample_rate, "wav")
        out_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
        sf.write(out_path, wav, sample_rate)
        headers = {}
        if save_path:
            headers["X-Saved-Audio-Path"] = save_path
        return FileResponse(out_path, media_type="audio/wav", filename="output.wav", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        print("TTS ERROR repr(e):", repr(e), flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/v1/audio/speech")
def openai_audio_speech(req: OpenAITTSRequest, authorization: Optional[str] = Header(default=None)):
    cleanup_files: List[str] = []
    try:
        if authorization:
            print("Authorization header received", flush=True)
        if req.model and req.model != OPENAI_MODEL_NAME:
            print(f"Requested model '{req.model}' != server model '{OPENAI_MODEL_NAME}', continue anyway", flush=True)

        stream_enabled = _is_stream_request(req)
        request_summary = _summarize_openai_request(req)
        print("Incoming OpenAI TTS request summary:\n" + _json_pretty(request_summary), flush=True)

        extra = _get_extra(req)
        fallback_cfg_value = _safe_float(
            extra.get("cfg_value", req.cfg_value if req.cfg_value is not None else DEFAULT_CFG_VALUE),
            DEFAULT_CFG_VALUE,
            1.3,
            1.9,
        )
        fallback_inference_timesteps = _safe_int(
            extra.get("inference_timesteps", req.inference_timesteps if req.inference_timesteps is not None else DEFAULT_INFERENCE_TIMESTEPS),
            DEFAULT_INFERENCE_TIMESTEPS,
            4,
            12,
        )

        reference_wav_path, prompt_text, cleanup_files = _extract_reference_from_openai_request(req)
        print("Reference audio enabled" if reference_wav_path else "No reference audio", flush=True)

        policy = _get_backend_tts_policy(req=req, reference_wav_path=reference_wav_path, prompt_text=prompt_text)
        print("Backend TTS policy:", _json_preview(policy), flush=True)
        print(
            f"Voice source: name={policy.get('voice_name')} source={policy.get('voice_source')} clone_mode={policy.get('clone_mode')}",
            flush=True,
        )

        segments = _prepare_tts_segments(req.input, prompt_text, bool(reference_wav_path), policy)
        print("REQUEST MODE =", "STREAMING" if stream_enabled else "NON_STREAMING", flush=True)

        if stream_enabled:
            return _streaming_audio_response(
                segments=segments,
                fallback_cfg_value=fallback_cfg_value,
                fallback_inference_timesteps=fallback_inference_timesteps,
                reference_wav_path=reference_wav_path,
                prompt_text=prompt_text,
                policy=policy,
                response_format=req.response_format,
                cleanup_files=cleanup_files,
            )

        wav, sample_rate = _generate_segments_audio(
            segments=segments,
            fallback_cfg_value=fallback_cfg_value,
            fallback_inference_timesteps=fallback_inference_timesteps,
            reference_wav_path=reference_wav_path,
            prompt_text=prompt_text,
            policy=policy,
        )
        return _audio_response(wav, sample_rate, response_format=req.response_format)
    except HTTPException:
        raise
    except Exception as e:
        print("OPENAI TTS ERROR repr(e):", repr(e), flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=repr(e))
    finally:
        try:
            if not _is_stream_request(req):
                _cleanup_files(cleanup_files)
        except Exception:
            _cleanup_files(cleanup_files)


@app.get("/")
def root():
    return JSONResponse({
        "message": "VoxCPM2 reference-audio clone server is running",
        "health": "/health",
        "tts": "/tts",
        "openai_speech": "/v1/audio/speech",
        "models": "/v1/models",
        "debug_tts_plan": "/debug/tts_plan",
        "model_name": OPENAI_MODEL_NAME,
        "reference_audio": {
            "supports_data_uri": True,
            "field_nested": "extra_body.references[0].audio",
            "field_flat": "references[0].audio",
            "text_field_nested": "extra_body.references[0].text",
            "text_field_flat": "references[0].text",
            "default_clone_mode": "high_similarity when reference text is provided",
        },
        "streaming": {
            "same_endpoint": "/v1/audio/speech",
            "enable_by": "stream=true or extra_body.stream=true",
            "mp3_streaming_uses_ffmpeg_pipe": True,
        },
        "storage": {"output_dir": OUTPUT_DIR, "layout": "YYYY-MM-DD/00001.wav or 00001.mp3"},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
