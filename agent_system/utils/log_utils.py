# Logging: redact base64 image data, setup file + console logger.

import json
import re
import logging
from datetime import datetime

_DATA_URI_HEADER_RE = re.compile(r'data:image/[^;]+;base64,', re.IGNORECASE)


class RedactingFormatter(logging.Formatter):
    """Replaces base64 image placeholders in log messages with a short description."""

    def __init__(self, fmt: str = None, images_placeholder: str = None):
        super().__init__(fmt)
        self.images_placeholder = images_placeholder or "frames attached (hidden)"

    def format(self, record: logging.LogRecord) -> str:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        matches = _DATA_URI_HEADER_RE.findall(msg)
        if matches:
            record.msg = json.dumps(
                {"images": f"{len(matches)} {self.images_placeholder}"},
                ensure_ascii=False,
            )
            record.args = ()
        return super().format(record)


def setup_logger(
    task_name: str,
    start_question: int = None,
    end_question: int = None,
    logger_name: str = "AgentWorkflowLogger",
) -> logging.Logger:
    """Creates logger with file + console handlers and base64 redaction; log file named by task and range."""
    formatter = RedactingFormatter('%(asctime)s - %(levelname)s - %(message)s')
    if start_question is not None and end_question is not None:
        log_filename = f"log_{task_name}_main_q{start_question}_to_{end_question}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_filename = f"log_{task_name}_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger
