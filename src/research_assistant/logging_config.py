"""Application logging configuration: a single console handler on the root logger."""

import logging
import logging.config

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a console handler.

    Safe to call more than once; dictConfig replaces the previous config.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):  # e.g. an invalid LOG_LEVEL value
        numeric_level = logging.INFO

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"console": {"format": _LOG_FORMAT}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": numeric_level, "handlers": ["console"]},
        }
    )
