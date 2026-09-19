"""Send policy documents to the internal ingestion pipeline, which chunks, embeds and indexes them in Pinecone.

PLACEHOLDER: the endpoint contract (multipart vs JSON, auth) is not final. Adjust `upload()` when it is.
Usage: python scripts/ingest_policy_client.py [file ...]   (defaults to everything in data/policy/)
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def upload(path: Path) -> None:
    resp = requests.post(
        os.environ["INGEST_API_URL"],
        headers={"Authorization": f"Bearer {os.getenv('INGEST_API_KEY', '')}"},
        files={"file": (path.name, path.read_bytes())},
        timeout=120,
    )
    resp.raise_for_status()
    print(f"Ingested {path.name}: {resp.status_code}")


if __name__ == "__main__":
    files = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "data" / "policy").glob("*.*"))
    for f in files:
        upload(f)
