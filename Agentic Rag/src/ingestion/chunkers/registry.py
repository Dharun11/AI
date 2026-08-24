from ingestion.chunkers.base import Chunker

CHUNKER_REGISTRY: dict[str, type[Chunker]] = {}


def register_chunker(name: str):
    def decorator(cls: type[Chunker]) -> type[Chunker]:
        CHUNKER_REGISTRY[name] = cls
        return cls

    return decorator


def get_chunker(name: str) -> Chunker:
    try:
        cls = CHUNKER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown chunker '{name}'. Registered chunkers: {sorted(CHUNKER_REGISTRY)}"
        ) from None
    return cls()
