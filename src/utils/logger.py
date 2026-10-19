"""Structured logging setup for the whole project."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.utils.config import get_config_value, resolve_path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def setup_logging(level: str | None = None, log_file: str = "logs/app.log") -> None:
    """Configure the root logger once (console + rotating file)."""
    global _configured
    if _configured:
        return

    level_name = (level or get_config_value("logging.level", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    log_path = resolve_path(log_file, create=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)