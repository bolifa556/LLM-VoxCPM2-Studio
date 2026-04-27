from __future__ import annotations

import base64
import io
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .config import APP_ROOT, AppConfig, load_config, merge_config_override, save_config
from .llm import segment_text_with_llm
from .models import GenerateRequest, SaveVoiceRequest, SegmentRequest
from .prompts import load_prompt_markdown, save_prompt_markdown
from .tts_adapters.registry import get_tts_adapter
from .voice_library import ensure_voice_library, get_voice, list_voices, save_generated_voice

templates = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))


def _decode_audio_data_url(data_url: str) -> tuple[bytes, str]:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="无效的 data URL。")
    header, payload = data_url.split(",", 1)
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="只支持 base64 编码的音频 data URL。")
    mime = header.split(":")[-1].split(";")[0]
    ext = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
    }.get(mime, ".bin")
    return base64.b64decode(payload), ext


def _resolve_request_config(config_override: dict | None) -> AppConfig:
    return merge_config_override(load_config(), config_override)


def _resolve_reference_assets(req: GenerateRequest):
    temp_files: list[Path] = []
    reference_audio_path: Path | None = None
    reference_text = (req.reference_text or "").strip() or None

    chosen_voice = get_voice(req.selected_voice_id)
    if chosen_voice and chosen_voice.reference_audio_path and not req.reference_audio_data_url:
        reference_audio_path = APP_ROOT / chosen_voice.reference_audio_path
        reference_text = reference_text or chosen_voice.reference_text

    if req.reference_audio_data_url:
        if not reference_text:
            raise HTTPException(status_code=400, detail="上传参考音频时，必须同时提供参考文字。")
        data, ext = _decode_audio_data_url(req.reference_audio_data_url)
        temp_path = Path(tempfile.gettempdir()) / f"voxcpm_ref_{uuid.uuid4().hex}{ext}"
        temp_path.write_bytes(data)
        temp_files.append(temp_path)
        reference_audio_path = temp_path

    return chosen_voice, reference_audio_path, reference_text, temp_files


def _read_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav_file:
        return {
            "channels": wav_file.getnchannels(),
            "sample_width": wav_file.getsampwidth(),
            "sample_rate": wav_file.getframerate(),
            "frames": wav_file.readframes(wav_file.getnframes()),
        }


def _silence_frames(sample_rate: int, sample_width: int, channels: int, pause_ms: int) -> bytes:
    frame_count = max(0, int(sample_rate * pause_ms / 1000.0))
    return b"\x00" * frame_count * sample_width * channels


def _concat_segments_audio(chunks: list[tuple[Path, int]]) -> bytes:
    if not chunks:
        raise HTTPException(status_code=400, detail="没有可拼接的音频片段。")

    first = _read_wav(chunks[0][0])
    channels = first["channels"]
    sample_width = first["sample_width"]
    sample_rate = first["sample_rate"]

    merged = bytearray()
    for index, (path, pause_ms) in enumerate(chunks):
        current = _read_wav(path)
        if current["sample_rate"] != sample_rate or current["channels"] != channels or current["sample_width"] != sample_width:
            raise HTTPException(status_code=500, detail="音频片段格式不一致，无法拼接。")

        merged.extend(current["frames"])
        if pause_ms > 0 and index < len(chunks) - 1:
            merged.extend(_silence_frames(sample_rate, sample_width, channels, pause_ms))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(merged))
    return buffer.getvalue()


def _pick_local_folder(title: str = "选择文件夹") -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=title)
        root.destroy()
        selected = (selected or "").strip()
        return selected or None
    except Exception:
        pass

    ps_script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
        f"$dialog.Description = '{title}'; "
        "$dialog.UseDescriptionForTitle = $true; "
        "$result = $dialog.ShowDialog(); "
        "if ($result -eq [System.Windows.Forms.DialogResult]::OK) { "
        "  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "  Write-Output $dialog.SelectedPath "
        "}"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
        selected = (result.stdout or "").strip()
        return selected or None
    except Exception:
        return None


def _download_hf_model(repo_id: str, target_dir: str) -> str:
    repo_id = (repo_id or "").strip()
    target_dir = (target_dir or "").strip()
    if not repo_id:
        raise HTTPException(status_code=400, detail="请先填写 Hugging Face 仓库名。")
    if not target_dir:
        target_dir = str(APP_ROOT)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise HTTPException(status_code=500, detail="缺少 huggingface_hub 依赖，请先安装 requirements-app.txt。") from exc

    repo_leaf = repo_id.split("/")[-1].strip() or "model"
    local_dir = Path(target_dir) / repo_leaf
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return str(Path(downloaded))


def create_app() -> FastAPI:
    ensure_voice_library()

    app = FastAPI(title="LLM VoxCPM2 Studio")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=str(APP_ROOT / "app" / "static")), name="static")
    app.mount("/voice-files", StaticFiles(directory=str(APP_ROOT / "voice")), name="voice-files")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(
            name="index.html",
            request=request,
            context={"request": request, "app_name": "LLM VoxCPM2 Studio"},
        )

    @app.get("/api/config")
    async def get_config():
        return load_config().model_dump(mode="json")

    @app.put("/api/config")
    async def update_config(new_config: AppConfig):
        saved = save_config(new_config)
        return {"ok": True, "config": saved.model_dump(mode="json")}

    @app.get("/api/prompt")
    async def get_prompt():
        return {"content": load_prompt_markdown()}

    @app.put("/api/prompt")
    async def update_prompt(payload: dict):
        save_prompt_markdown(str(payload.get("content", "")))
        return {"ok": True}

    @app.get("/api/voices")
    async def voices():
        return {"voices": [voice.model_dump(mode="json") for voice in list_voices()]}

    @app.post("/api/system/pick-model-folder")
    async def pick_model_folder(payload: dict | None = None):
        title = "选择 VoxCPM 模型文件夹"
        if isinstance(payload, dict) and payload.get("title"):
            title = str(payload.get("title"))
        selected = _pick_local_folder(title)
        return {"selected_path": selected, "cancelled": not bool(selected)}

    @app.post("/api/system/download-model")
    async def download_model(payload: dict):
        cfg = _resolve_request_config(payload.get("config_override"))
        adapter_id = payload.get("tts_adapter_id") or cfg.tts.active_adapter
        adapter_cfg = cfg.tts.adapters[adapter_id]
        repo_id = str(payload.get("repo_id") or adapter_cfg.model_path or "").strip()
        target_dir = str(payload.get("download_dir") or adapter_cfg.download_dir or APP_ROOT).strip()
        local_path = _download_hf_model(repo_id, target_dir)
        return {
            "ok": True,
            "repo_id": repo_id,
            "download_dir": target_dir,
            "local_model_path": local_path,
        }

    @app.post("/api/segment")
    async def segment(req: SegmentRequest):
        cfg = _resolve_request_config(req.config_override)
        voice = get_voice(req.selected_voice_id)
        reference_text = req.reference_text or (voice.reference_text if voice else None)
        segments = segment_text_with_llm(
            config=cfg,
            text=req.text,
            llm_provider_id=req.llm_provider_id,
            tts_adapter_id=req.tts_adapter_id,
            control_hint=req.control_hint,
            soul_override=req.soul_override,
            reference_mode=bool(req.reference_mode or (voice and voice.reference_audio_path)),
            reference_text=reference_text,
        )
        return {"segments": [item.model_dump(mode="json") for item in segments]}

    @app.post("/api/generate")
    async def generate(req: GenerateRequest):
        cfg = _resolve_request_config(req.config_override)
        adapter = get_tts_adapter(cfg, req.tts_adapter_id)
        chosen_voice, reference_audio_path, reference_text, temp_files = _resolve_reference_assets(req)

        if not req.segments:
            raise HTTPException(status_code=400, detail="没有可用于生成的分段。")

        output_dir = Path(tempfile.gettempdir()) / "llm-voxcpm2-preview"
        output_dir.mkdir(parents=True, exist_ok=True)
        segment_paths: list[Path] = []

        try:
            final_text_parts: list[str] = []
            generated_chunks: list[tuple[Path, int]] = []
            fallback_cfg = req.cfg_value
            fallback_steps = req.inference_timesteps

            for index, segment in enumerate(req.segments, start=1):
                result = adapter.synthesize(
                    text=segment.text,
                    output_stem=f"{uuid.uuid4().hex}_{index}",
                    control_hint=segment.control or (req.control_hint if len(req.segments) == 1 else ""),
                    reference_audio_path=reference_audio_path,
                    reference_text=reference_text,
                    cfg_value=segment.cfg_value if segment.cfg_value is not None else fallback_cfg,
                    inference_timesteps=segment.inference_timesteps if segment.inference_timesteps is not None else fallback_steps,
                    output_dir=output_dir,
                )
                audio_path = Path(result["audio_path"])
                segment_paths.append(audio_path)
                generated_chunks.append((audio_path, max(0, int(segment.pause_ms or 0))))
                final_text_parts.append(result.get("final_text", segment.text))

            audio_bytes = _concat_segments_audio(generated_chunks)
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        finally:
            for path in temp_files:
                if path.exists():
                    path.unlink(missing_ok=True)
            for path in segment_paths:
                if path.exists():
                    path.unlink(missing_ok=True)

        return {
            "audio_data_url": f"data:audio/wav;base64,{audio_base64}",
            "reference_text": reference_text or req.text,
            "final_text": "\n".join(final_text_parts).strip() or req.text,
            "can_save_voice": True,
        }

    @app.post("/api/voices/save")
    async def save_voice(req: SaveVoiceRequest):
        audio_bytes, ext = _decode_audio_data_url(req.audio_data_url)
        temp_path = Path(tempfile.gettempdir()) / f"voice_save_{uuid.uuid4().hex}{ext}"
        temp_path.write_bytes(audio_bytes)
        try:
            voice = save_generated_voice(
                name=req.name,
                audio_source=temp_path,
                reference_text=req.reference_text,
                source=req.source,
            )
        finally:
            temp_path.unlink(missing_ok=True)
        return {"voice": voice.model_dump(mode="json")}

    return app
