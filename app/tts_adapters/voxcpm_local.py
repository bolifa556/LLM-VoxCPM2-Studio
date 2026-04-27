from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .base import BaseTTSAdapter


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "audio"


class VoxCPMLocalAdapter(BaseTTSAdapter):
    adapter_id = "voxcpm_local"
    format_name = "VoxCPM2 Local Request"

    def __init__(self, settings):
        self.settings = settings
        self._model = None

    def request_format_markdown(self) -> str:
        return (
            "当前适配器是 `voxcpm_local`\n\n"
            "- 输入文本最终会送进 `VoxCPM2`，支持在正文前拼接 `control`\n"
            "- 分段结果会逐段进入 TTS\n"
            "- `control` 适合写说话方式、情绪方向、节奏、轻重音，不适合塞太多设定解释\n"
            "- 参考模式优先使用 `reference_wav_path` + `prompt_text`\n"
            "- 如果带参考音频，模型会更贴近参考音色，control 只做轻量补充\n"
            "- 你输出的 JSON 应包含 `text`、`control`、`pause_ms`、`emotion`、`cfg_value`、`inference_timesteps`\n"
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from voxcpm import VoxCPM
        except Exception as exc:
            raise RuntimeError("未安装 `voxcpm`，请先按 README 安装 VoxCPM2 本地依赖。") from exc

        model_path = self.settings.model_path
        if hasattr(VoxCPM, "from_pretrained"):
            self._model = VoxCPM.from_pretrained(
                model_path,
                load_denoiser=bool(self.settings.load_denoiser),
            )
        else:
            self._model = VoxCPM(
                model_path,
                zipenhancer_model_path=None if not self.settings.load_denoiser else "iic/speech_zipenhancer_ans_multiloss_16k_base",
                enable_denoiser=bool(self.settings.load_denoiser),
            )
        return self._model

    def synthesize(
        self,
        text: str,
        output_stem: str,
        control_hint: str,
        reference_audio_path: Optional[Path],
        reference_text: Optional[str],
        cfg_value: Optional[float],
        inference_timesteps: Optional[int],
        output_dir: Path,
    ) -> dict:
        try:
            import numpy as np
            import soundfile as sf
        except Exception as exc:
            raise RuntimeError("当前环境缺少 `numpy` 或 `soundfile`，请先安装应用依赖。") from exc

        model = self._load_model()
        output_dir.mkdir(parents=True, exist_ok=True)
        final_text = text.strip()
        use_prompt_text_mode = bool(reference_audio_path and reference_text and reference_text.strip())
        if control_hint.strip() and not use_prompt_text_mode:
            final_text = f"({control_hint.strip()}){final_text}"

        kwargs = {
            "text": final_text,
            "cfg_value": cfg_value if cfg_value is not None else self.settings.default_cfg_value,
            "inference_timesteps": inference_timesteps if inference_timesteps is not None else self.settings.default_inference_timesteps,
            "denoise": False,
        }
        if reference_audio_path:
            kwargs["reference_wav_path"] = str(reference_audio_path)
            if reference_text and reference_text.strip():
                kwargs["prompt_wav_path"] = str(reference_audio_path)
                kwargs["prompt_text"] = reference_text.strip()

        try:
            wav = model.generate(**kwargs)
        except TypeError:
            kwargs.pop("prompt_wav_path", None)
            kwargs.pop("prompt_text", None)
            wav = model.generate(**kwargs)

        wav_array = np.asarray(wav, dtype=np.float32).reshape(-1)
        sample_rate = int(getattr(model.tts_model, "sample_rate", 24000))
        filename = f"{_safe_slug(output_stem)}.wav"
        output_path = output_dir / filename
        sf.write(output_path, wav_array, sample_rate)
        return {
            "audio_path": str(output_path),
            "sample_rate": sample_rate,
            "final_text": final_text,
        }
