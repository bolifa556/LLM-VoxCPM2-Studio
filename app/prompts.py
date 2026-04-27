from __future__ import annotations

from .config import PROMPT_PATH, PROMPT_RUNTIME_PATH, SOUL_PATH


def load_prompt_markdown() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def save_prompt_markdown(content: str) -> None:
    PROMPT_PATH.write_text(content, encoding="utf-8")


def load_soul_markdown() -> str:
    return SOUL_PATH.read_text(encoding="utf-8")


def load_runtime_template() -> str:
    return PROMPT_RUNTIME_PATH.read_text(encoding="utf-8")


def build_segmentation_prompt(
    user_text: str,
    request_format_name: str,
    request_format_markdown: str,
    soul_markdown: str,
    control_hint: str,
    reference_mode: bool,
    reference_text: str | None,
) -> str:
    rules = load_prompt_markdown().strip()
    template = load_runtime_template()
    replacements = {
        "{{SEGMENTATION_RULES}}": rules,
        "{{REQUEST_FORMAT_NAME}}": request_format_name,
        "{{REQUEST_FORMAT_MARKDOWN}}": request_format_markdown.strip(),
        "{{SOUL_MARKDOWN}}": (soul_markdown or "").strip(),
        "{{CONTROL_HINT}}": (control_hint or "").strip() or "无额外 control",
        "{{REFERENCE_MODE}}": "是" if reference_mode else "否",
        "{{REFERENCE_TEXT}}": (reference_text or "").strip() or "无",
        "{{USER_TEXT}}": user_text.strip(),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt
