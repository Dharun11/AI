from ingestion.parsers.base import Parser

PARSER_REGISTRY: dict[str, type[Parser]] = {}


def register_parser(name: str):
    def decorator(cls: type[Parser]) -> type[Parser]:
        PARSER_REGISTRY[name] = cls
        return cls

    return decorator


def get_parser(name: str) -> Parser:
    try:
        cls = PARSER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown parser '{name}'. Registered parsers: {sorted(PARSER_REGISTRY)}"
        ) from None
    return cls()
