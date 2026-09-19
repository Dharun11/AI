from fastapi import APIRouter

from ingestion.chunkers.registry import list_registered as list_chunkers
from ingestion.embedders.factory import list_registered as list_embedders
from ingestion.parsers.factory import list_registered as list_parsers
from ingestion.store.factory import list_registered as list_stores

from api.schemas import HealthResponse, StrategiesResponse

router = APIRouter()


@router.get("/strategies", response_model=StrategiesResponse)
def strategies() -> StrategiesResponse:
    return StrategiesResponse(
        parsers=list_parsers(),
        chunkers=list_chunkers(),
        embedders=list_embedders(),
        stores=list_stores(),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()
