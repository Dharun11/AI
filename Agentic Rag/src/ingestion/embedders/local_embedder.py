from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.embedders.factory import register_embedder

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@register_embedder("sentence_transformer")
class SentenceTransformerEmbedder:
    """Local, no-API-key embedder backed by a sentence-transformers model."""

    name = "sentence_transformer"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = 32) -> None:
        # Imported lazily so importing this module (e.g. via the embedders
        # package __init__ for registration) doesn't pull in torch/sentence-
        # transformers just to register the class.
        from sentence_transformers import SentenceTransformer

        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_embedding_dimension()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _encode_batch(self, batch: list[str]) -> list[list[float]]:
        # bge models are trained/benchmarked for cosine similarity — normalizing
        # here means a plain dot product downstream is equivalent to cosine.
        vectors = self._model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._encode_batch(batch))
        return vectors
