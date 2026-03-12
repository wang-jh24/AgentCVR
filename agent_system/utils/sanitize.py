# String truncation and log sanitization (e.g. strip Base64).

import json
from typing import Any


def safe_truncate(s: str, hard_limit: int = 500) -> str:
    """Truncate string to hard_limit with trailing ellipsis."""
    if s is None:
        return ""
    if len(s) <= hard_limit:
        return s
    return s[:hard_limit] + "... (Result truncated)"


def ensure_str_content(content: Any) -> str:
    """Coerce content to string; serialize dict/list as JSON with ensure_ascii=False."""
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    return str(content)


def sanitize_for_log(data: Any) -> Any:
    """Recursively replace long non-space strings (e.g. Base64) with '<frame>'; keep normal text."""
    if isinstance(data, dict):
        return {k: sanitize_for_log(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_log(i) for i in data]
    if isinstance(data, str):
        if len(data) > 1000 and " " not in data[:100]:
            return "<frame>"
        return data
    return data
