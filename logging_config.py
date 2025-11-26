# logging_config.py
import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "debug.log"


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given name.
    Logs INFO to console and DEBUG to file.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers on reload
        logger.setLevel(logging.DEBUG)

        # --- File handler ---
        fh = logging.FileHandler(LOG_FILE)
        fh.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        fh.setFormatter(file_fmt)

        # --- Console handler ---
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        console_fmt = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(console_fmt)

        # Attach handlers
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
