"""Turn text into embeddings with the local model (FR-19; decision recorded in T1.5).

    embedder = get_embedder()
    vectors = embedder.embed_documents(["Laptop sales in EMEA ...", ...])   # summaries
    vector = embedder.embed_query("How did laptops do in EMEA?")            # a question

BGE models are trained so that a *search question* starts with a short instruction, and the
documents being searched do not. `embed_query` adds that instruction; `embed_documents`
never does. Leaving it off makes search noticeably worse.

The model is loaded once, on first use, and then reused.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Protocol

from sales_poc.config import EmbeddingSettings, get_embedding_settings

# The instruction BGE English models expect in front of a search question.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEFAULT_BATCH_SIZE = 64


class EmbeddingError(RuntimeError):
    """The embedding model could not be used as configured."""


class TextModel(Protocol):
    """The part of fastembed's TextEmbedding we use (a fake one is used in tests)."""

    def embed(self, documents: Sequence[str], batch_size: int = ...) -> Iterable: ...


def query_instruction(model_name: str) -> str:
    return BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""


class Embedder:
    def __init__(self, settings: EmbeddingSettings, model: TextModel | None = None):
        if settings.embedding_provider != "local":
            raise EmbeddingError(
                f"EMBEDDING_PROVIDER is '{settings.embedding_provider}', but only 'local' is "
                "built. Set EMBEDDING_PROVIDER=local in .env."
            )
        self.settings = settings
        self.dim = settings.embedding_dim
        self._model = model
        self._instruction = query_instruction(settings.embedding_model)

    @property
    def model(self) -> TextModel:
        if self._model is None:
            # Windows can't make the file shortcuts the model cache prefers; it works
            # without them, so hide the long warning about it.
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from fastembed import TextEmbedding  # imported here: loading takes a moment

            try:
                self._model = TextEmbedding(
                    model_name=self.settings.embedding_model,
                    cache_dir=str(self.settings.embedding_cache_dir),
                )
            except Exception as exc:  # noqa: BLE001 - download and file errors vary
                raise EmbeddingError(
                    f"Could not load '{self.settings.embedding_model}' into "
                    f"{self.settings.embedding_cache_dir}/ (the first run needs internet): {exc}"
                ) from exc
        return self._model

    def _embed(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        if not texts:
            return []
        vectors = [[float(x) for x in v] for v in self.model.embed(list(texts), batch_size=batch_size)]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Asked for {len(texts)} vectors, got {len(vectors)}.")
        sizes = {len(v) for v in vectors}
        if sizes != {self.dim}:
            raise EmbeddingError(
                f"The model makes vectors of size {sorted(sizes)}, but EMBEDDING_DIM is "
                f"{self.dim}. Make them match in .env."
            )
        return vectors

    def embed_documents(self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
        """Vectors for texts that will be stored and searched (no instruction added)."""
        return self._embed(texts, batch_size)

    def embed_query(self, question: str) -> list[float]:
        """The vector for one search question (instruction added for BGE models)."""
        return self._embed([self._instruction + question], 1)[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(get_embedding_settings())
