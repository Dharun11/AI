import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class DeadLetterRecord(BaseModel):
    source_uri: str
    stage: str
    exception_type: str
    message: str
    traceback: str
    timestamp: datetime


class DeadLetterSink:
    """Writes one JSON file per failed document so a bad file can't stop the batch."""

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def write(self, record: DeadLetterRecord) -> Path:
        path = self._directory / f"{uuid.uuid4()}.json"
        path.write_text(record.model_dump_json(indent=2))
        return path
