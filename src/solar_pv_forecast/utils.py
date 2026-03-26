"""Logging setup for the pipeline."""

import sys
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from solar_pv_forecast.config import OUTPUT_DIR


def setup_logger(log_file: str | None = None) -> None:
    """Configure loguru with console + file sinks."""
    logger.remove()  # remove default
    logger.add(sys.stderr, level="INFO", format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    ))
    if log_file:
        path = Path(log_file)
    else:
        path = OUTPUT_DIR / "runtime.log"
    logger.add(str(path), level="DEBUG", rotation="10 MB")


@contextmanager
def log_step(name: str):
    """Context manager to time and log a pipeline step."""
    logger.info(f"▶ Starting: {name}")
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    logger.info(f"✓ Completed: {name} ({elapsed:.1f}s)")
