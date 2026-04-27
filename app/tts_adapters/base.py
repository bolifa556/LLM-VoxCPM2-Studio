from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseTTSAdapter(ABC):
    adapter_id: str
    format_name: str

    @abstractmethod
    def request_format_markdown(self) -> str:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
