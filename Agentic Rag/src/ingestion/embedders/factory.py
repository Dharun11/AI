from typing import Any

from ingestion.embedders.base import Embedder

EMBEDDER_REGISTRY: dict[str, type[Embedder]] = {}


def register_embedder(name: str):
    def decorator(cls: type[Embedder]) -> type[Embedder]:
        EMBEDDER_REGISTRY[name] = cls
        return cls

    return decorator


def get_embedder(name: str, **kwargs: Any) -> Embedder:
    try:
        cls = EMBEDDER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedder '{name}'. Registered embedders: {sorted(EMBEDDER_REGISTRY)}"
        ) from None
    return cls(**kwargs)


def list_registered() -> list[str]:
    return sorted(EMBEDDER_REGISTRY)
