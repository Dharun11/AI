from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """API-layer settings, distinct from IngestionSettings: these govern the
    service itself (where per-project state lives), not any one pipeline run.
    """

    model_config = SettingsConfigDict(env_prefix="AGENTIC_RAG_API_")

    # Per-project dead-letter dir / default Qdrant storage live under
    # data_root/<project_id>/... . When hash_store_name is "sqlite" (default),
    # each project's idempotency db also lives there as its own file; when
    # "mysql", every project instead shares one table (rows scoped by
    # project_id) in the database named by the MYSQL_* env vars - see
    # dependencies.py::project_paths / _get_hash_store.
    data_root: Path = Path("./data")
    hash_store_name: str = "sqlite"
