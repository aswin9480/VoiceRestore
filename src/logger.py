import logging
from logging.handlers import RotatingFileHandler

from .config import ROOT


def configure_logging() -> None:
    ROOT.joinpath("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("voicerestore")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            ROOT / "logs/application.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
