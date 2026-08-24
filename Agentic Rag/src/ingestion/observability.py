from dataclasses import asdict, dataclass

import structlog


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )


def get_logger(**initial_values: object) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(**initial_values)


@dataclass
class RunCounters:
    docs_processed: int = 0
    docs_failed: int = 0
    chunks_created: int = 0
    chunks_skipped_idempotent: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
