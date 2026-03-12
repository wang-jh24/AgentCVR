# Answer parsing for multi-choice, sort, interval, open-ended, UAV, etc.

import re
import json
from typing import Any, List, Optional, Tuple


def normalize_letters(seq, num_options: int) -> List[str]:
    """Normalize option sequence to uppercase letters, preserve order, dedupe."""
    if not seq:
        return []
    valid = {chr(65 + i) for i in range(num_options)}
    seen, out = set(), []
    for x in seq:
        if not x:
            continue
        c = str(x).strip().upper()
        if c in valid and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def extract_correct_list(answer_field: Any, num_options: int) -> List[str]:
    """Extract ground-truth answer list from field (list, nested list, or str)."""
    if answer_field is None:
        return []
    if isinstance(answer_field, list) and all(isinstance(x, str) for x in answer_field):
        return normalize_letters(answer_field, num_options)
    if isinstance(answer_field, list) and any(isinstance(x, list) for x in answer_field):
        for inner in answer_field:
            if isinstance(inner, list) and inner:
                return normalize_letters(inner, num_options)
        return []
    if isinstance(answer_field, str):
        return normalize_letters([answer_field], num_options)
    return []


def normalize_string_to_list(text_input: Any, num_options: int) -> List[str]:
    """Convert string or list to option letters (e.g. 'AC' -> ['A','C']) for multi-select."""
    if not text_input:
        return []
    valid_chars = {chr(65 + i) for i in range(num_options)}
    found = set()
    if isinstance(text_input, list):
        for item in text_input:
            if isinstance(item, str):
                for char in item.upper():
                    if char in valid_chars:
                        found.add(char)
    elif isinstance(text_input, str):
        for char in text_input.upper():
            if char in valid_chars:
                found.add(char)
    return sorted(list(found))


def parse_agent_output_multi(final_answer_text: Any, num_options: int) -> List[str]:
    """Parse multi/single-choice agent output (PEA, CC, NC, PI)."""
    if isinstance(final_answer_text, list):
        return normalize_letters(final_answer_text, num_options)
    if isinstance(final_answer_text, dict):
        return normalize_letters(final_answer_text.get("content"), num_options)
    if isinstance(final_answer_text, str):
        s = final_answer_text.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return normalize_letters(obj, num_options)
            if isinstance(obj, dict):
                return normalize_letters(obj.get("content"), num_options)
            if isinstance(obj, str):
                return normalize_letters([obj], num_options)
        except Exception:
            pass
        m = re.search(r'"content"\s*:\s*(\[[^\]]*\])', s, flags=re.IGNORECASE)
        if m:
            try:
                return normalize_letters(json.loads(m.group(1)), num_options)
            except Exception:
                pass
        m2 = re.search(r'\[\s*"(?:[A-Za-z])"(?:\s*,\s*"(?:[A-Za-z])")*\s*\]', s)
        if m2:
            try:
                return normalize_letters(json.loads(m2.group(0)), num_options)
            except Exception:
                pass
        matches = re.findall(r'(?<!Video\s)\b([A-E])\b', s)
        if matches:
            return normalize_letters([matches[-1]], num_options)
    return []


def parse_agent_output_multi_BU(final_answer_text: Any, num_options: int) -> List[str]:
    """Parse BU (behavior) task multi-select output."""
    if isinstance(final_answer_text, dict):
        val = final_answer_text.get("final_answer") or final_answer_text.get("content")
        return normalize_string_to_list(val, num_options)
    if isinstance(final_answer_text, list):
        return normalize_string_to_list(final_answer_text, num_options)
    if isinstance(final_answer_text, str):
        s = final_answer_text.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                val = obj.get("final_answer") or obj.get("content")
                return normalize_string_to_list(val, num_options)
            if isinstance(obj, list):
                return normalize_string_to_list(obj, num_options)
            if isinstance(obj, str):
                return normalize_string_to_list(obj, num_options)
        except json.JSONDecodeError:
            pass
        m_key = re.search(r'"final_answer"\s*:\s*"([^"]+)"', s)
        if m_key:
            return normalize_string_to_list(m_key.group(1), num_options)
        matches = re.findall(r'"final_answer"\s*:\s*([A-Za-z,\s]+)', s)
        if matches:
            return normalize_string_to_list(matches[0], num_options)
        if len(s) < 10:
            return normalize_string_to_list(s, num_options)
    return []


def parse_agent_output_sort(final_answer_text: Any) -> Optional[str]:
    """Parse PSS (sort) task output, e.g. '2->3->1->4'."""
    def _clean(s):
        return s.strip().replace('"', '').replace("'", "")

    if isinstance(final_answer_text, list) and len(final_answer_text) > 0:
        return _clean(str(final_answer_text[0]))
    if isinstance(final_answer_text, dict):
        content = final_answer_text.get("content")
        if isinstance(content, list) and len(content) > 0:
            return _clean(str(content[0]))
        if isinstance(content, str):
            return _clean(content)
    if isinstance(final_answer_text, str):
        s = final_answer_text.strip()
        match = re.search(r'(\d+\s*->\s*\d+(?:\s*->\s*\d+)*)', s)
        if match:
            return match.group(1).replace(" ", "")
        try:
            obj = json.loads(s)
            if isinstance(obj, list) and len(obj) > 0:
                return _clean(str(obj[0]))
            if isinstance(obj, dict):
                return _clean(str(obj.get("content", "")))
        except Exception:
            pass
    return None


def parse_agent_output_json_list(final_answer_text: Any, num_options: int) -> List[str]:
    """Parse single-choice list output (e.g. UAV), e.g. ['A']."""
    if not isinstance(final_answer_text, str):
        if isinstance(final_answer_text, list) and len(final_answer_text) > 0:
            return [str(final_answer_text[0])]
        return []
    s = final_answer_text.strip()
    try:
        m = re.search(r'\[\s*"([A-Z])"\s*\]', s, re.DOTALL)
        if m:
            return [m.group(1)]
        m2 = re.search(r'"final_answer"\s*:\s*"([A-Z])"', s)
        if m2:
            return [m2.group(1)]
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            val = parsed.get("final_answer")
            if isinstance(val, list) and val:
                val = val[0]
            return [str(val)] if val is not None and val != "" else []
    except Exception:
        pass
    valid = {chr(65 + i) for i in range(num_options)}
    for char in s:
        if char.upper() in valid:
            return [char.upper()]
    return []


def interval_iou(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> float:
    """Intersection-over-union for two time intervals."""
    if not interval1 or not interval2:
        return 0.0
    start1, end1 = interval1
    start2, end2 = interval2
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)
    union = max(end1, end2) - min(start1, start2)
    if union == 0:
        return 1.0 if intersection > 0 else 0.0
    return intersection / union


def parse_json_interval_string(text: str) -> Optional[Tuple[float, float]]:
    """Parse [start, end] time interval from LLM output."""
    if not text:
        return None
    p_list = re.compile(r'\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]')
    p_str_val = re.compile(r'"([\d\.]+,\s*[\d\.]+)"')

    def _parse(t):
        m = p_list.search(t)
        if m:
            return (float(m.group(1)), float(m.group(2)))
        m2 = p_str_val.search(t)
        if m2:
            parts = m2.group(1).split(',')
            return (float(parts[0]), float(parts[1]))
        return None

    if '</think>' in text:
        res = _parse(text.rsplit('</think>', 1)[-1])
        if res:
            return res
    return _parse(text)


def extract_answer_text(final_answer: str) -> str:
    """Extract answer text from CCQA (open) task agent final_answer for scoring."""
    if not final_answer:
        return ""
    try:
        if isinstance(final_answer, dict):
            return final_answer.get("answer", str(final_answer))
        data = json.loads(final_answer)
        if isinstance(data, dict) and "answer" in data:
            return data["answer"]
    except Exception:
        pass
    match = re.search(r'\{\s*"answer"\s*:\s*"([^"]+)"\s*\}', final_answer, re.IGNORECASE)
    if match:
        return match.group(1)
    return final_answer if isinstance(final_answer, str) else str(final_answer)
