# Text cleanup (e.g. strip <think> blocks).

import re


def clean_thought_content(content: str) -> str:
    """Remove <think>...</think> block from model output, keep the rest (e.g. JSON action)."""
    if not isinstance(content, str):
        return content
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return cleaned.strip()
