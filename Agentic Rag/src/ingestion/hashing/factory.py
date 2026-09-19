from typing import Any

from ingestion.hashing.base import HashStore

HASH_STORE_REGISTRY: dict[str, type[HashStore]] = {}


def register_hash_store(name: str):
    def decorator(cls: type[HashStore]) -> type[HashStore]:
        HASH_STORE_REGISTRY[name] = cls
        return cls

    return decorator


def get_hash_store(name: str, **kwargs: Any) -> HashStore:
    try:
        cls = HASH_STORE_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown hash store '{name}'. Registered hash stores: {sorted(HASH_STORE_REGISTRY)}"
        ) from None
    return cls(**kwargs)


def list_registered() -> list[str]:
    return sorted(HASH_STORE_REGISTRY)
