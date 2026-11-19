import pytest

from sales_poc.config import EmbeddingSettings
from sales_poc.vectors.embeddings import BGE_QUERY_INSTRUCTION, Embedder, EmbeddingError


class FakeModel:
    """Records what it was asked to embed; returns vectors of a chosen size."""

    def __init__(self, size=4):
        self.size, self.seen, self.batch_sizes = size, [], []

    def embed(self, documents, batch_size=64):
        self.seen.extend(documents)
        self.batch_sizes.append(batch_size)
        for i, _ in enumerate(documents):
            yield [float(i)] * self.size


def settings(**changes):
    values = {"embedding_provider": "local", "embedding_model": "BAAI/bge-small-en-v1.5",
              "embedding_dim": 4, **changes}
    return EmbeddingSettings(_env_file=None, **values)


def test_documents_are_embedded_without_the_instruction():
    model = FakeModel()
    vectors = Embedder(settings(), model).embed_documents(["a", "b", "c"], batch_size=2)

    assert model.seen == ["a", "b", "c"]
    assert model.batch_sizes == [2]
    assert len(vectors) == 3 and all(len(v) == 4 for v in vectors)


def test_questions_get_the_bge_instruction():
    model = FakeModel()
    Embedder(settings(), model).embed_query("laptop sales in EMEA")

    assert model.seen == [BGE_QUERY_INSTRUCTION + "laptop sales in EMEA"]


def test_other_models_get_no_instruction():
    model = FakeModel()
    Embedder(settings(embedding_model="some/other-model"), model).embed_query("q")

    assert model.seen == ["q"]


def test_vectors_are_plain_python_floats():
    [vector] = Embedder(settings(), FakeModel()).embed_documents(["a"])
    assert all(type(x) is float for x in vector)


def test_wrong_vector_size_is_caught():
    with pytest.raises(EmbeddingError, match="size \\[3\\].*EMBEDDING_DIM is 4"):
        Embedder(settings(), FakeModel(size=3)).embed_documents(["a"])


def test_only_the_local_provider_is_built():
    with pytest.raises(EmbeddingError, match="EMBEDDING_PROVIDER=local"):
        Embedder(settings(embedding_provider="openrouter"), FakeModel())


def test_nothing_to_embed():
    model = FakeModel()
    assert Embedder(settings(), model).embed_documents([]) == []
    assert model.seen == []
