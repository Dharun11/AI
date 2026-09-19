from typing import Any

from ingestion.store.base import VectorStoreWriter

STORE_REGISTRY: dict[str, type[VectorStoreWriter]] = {}


def register_store(name: str):
    def decorator(cls: type[VectorStoreWriter]) -> type[VectorStoreWriter]:
        STORE_REGISTRY[name] = cls
        return cls

    return decorator


def get_store(name: str, **kwargs: Any) -> VectorStoreWriter:
    try:
        cls = STORE_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown store '{name}'. Registered stores: {sorted(STORE_REGISTRY)}"
        ) from None
    return cls(**kwargs)


def list_registered() -> list[str]:
    return sorted(STORE_REGISTRY)
