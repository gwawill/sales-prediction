import math

import pytest
from botocore.exceptions import ClientError

from sales_poc.config import EmbeddingSettings
from sales_poc.vectors.embeddings import Embedder
from sales_poc.vectors.store import (
    PUT_BATCH,
    VectorIndex,
    VectorIndexError,
    filters,
    load_summaries,
    search,
)
from sales_poc.vectors.summaries import Summary


class FakeS3Vectors:
    """Enough of the S3 Vectors API for these tests, with real cosine search."""

    def __init__(self):
        self.indexes, self.calls = {}, []

    def get_index(self, vectorBucketName, indexName):
        if indexName not in self.indexes:
            raise ClientError({"Error": {"Code": "NotFoundException"}}, "GetIndex")
        return {"index": {"dimension": self.indexes[indexName]["dim"]}}

    def create_index(self, vectorBucketName, indexName, dataType, dimension, distanceMetric, metadataConfiguration):
        assert (dataType, distanceMetric) == ("float32", "cosine")
        assert metadataConfiguration == {"nonFilterableMetadataKeys": ["text"]}
        self.indexes[indexName] = {"dim": dimension, "vectors": {}}

    def put_vectors(self, vectorBucketName, indexName, vectors):
        assert len(vectors) <= PUT_BATCH
        self.calls.append(("put", len(vectors)))
        for v in vectors:
            self.indexes[indexName]["vectors"][v["key"]] = v

    def delete_vectors(self, vectorBucketName, indexName, keys):
        for k in keys:
            self.indexes[indexName]["vectors"].pop(k, None)

    def list_vectors(self, vectorBucketName, indexName, nextToken=None, returnMetadata=False):
        vectors = self.indexes[indexName]["vectors"]
        keys = sorted(vectors)
        start = int(nextToken or 0)
        page = keys[start:start + 3]  # tiny pages, to exercise paging
        more = start + 3 < len(keys)
        items = [{"key": k, **({"metadata": vectors[k]["metadata"]} if returnMetadata else {})} for k in page]
        return {"vectors": items, **({"nextToken": str(start + 3)} if more else {})}

    def query_vectors(self, vectorBucketName, indexName, queryVector, topK, returnMetadata, returnDistance, filter=None):
        def matches(meta, f):
            if f is None:
                return True
            if "$and" in f:
                return all(matches(meta, part) for part in f["$and"])
            [(key, cond)] = f.items()
            return meta.get(key) == cond["$eq"]

        q = queryVector["float32"]
        scored = []
        for v in self.indexes[indexName]["vectors"].values():
            if matches(v["metadata"], filter):
                d = v["data"]["float32"]
                cos = sum(a * b for a, b in zip(q, d)) / (math.hypot(*q) * math.hypot(*d))
                scored.append({"key": v["key"], "distance": 1 - cos, "metadata": v["metadata"]})
        return {"vectors": sorted(scored, key=lambda x: x["distance"])[:topK]}


class WordModel:
    """A tiny fake embedding: counts of a few words, so meaning is easy to predict."""

    WORDS = ["laptop", "server", "emea", "apj", "customer"]

    def embed(self, documents, batch_size=64):
        for text in documents:
            low = text.lower()
            yield [float(low.count(w)) + 0.01 for w in self.WORDS]


def embedder():
    settings = EmbeddingSettings(_env_file=None, embedding_provider="local",
                                 embedding_model="fake/words", embedding_dim=5)
    return Embedder(settings, WordModel())


SUMMARIES = [
    Summary("catreg:Laptop:EMEA:2026-03", "Laptop sales in EMEA, March: laptop laptop",
            {"kind": "catreg", "category": "Laptop", "region_id": "EMEA", "month": "2026-03"}),
    Summary("catreg:Server:APJ:2026-03", "Server sales in APJ, March: server server",
            {"kind": "catreg", "category": "Server", "region_id": "APJ", "month": "2026-03"}),
    Summary("catreg:Laptop:APJ:2026-03", "Laptop sales in APJ, March",
            {"kind": "catreg", "category": "Laptop", "region_id": "APJ", "month": "2026-03"}),
    Summary("customer:C00001", "Customer C00001 in EMEA",
            {"kind": "customer", "customer_id": "C00001", "region_id": "EMEA"}),
]


@pytest.fixture
def index():
    return VectorIndex(FakeS3Vectors(), "bucket", "sales-summaries", dim=5)


def test_first_load_creates_the_index_and_stores_everything(index):
    result = load_summaries(index, SUMMARIES, embedder())

    assert result.created_index
    assert (result.stored, result.deleted, result.total) == (4, 0, 4)
    stored = index.client.indexes["sales-summaries"]["vectors"]["catreg:Laptop:EMEA:2026-03"]
    assert stored["metadata"]["text"].startswith("Laptop sales in EMEA")
    assert stored["metadata"]["region_id"] == "EMEA"


def test_loading_again_adds_no_copies(index):
    load_summaries(index, SUMMARIES, embedder())
    result = load_summaries(index, SUMMARIES, embedder())

    assert not result.created_index
    assert result.total == 4


def test_summaries_that_disappear_are_deleted(index):
    load_summaries(index, SUMMARIES, embedder())
    result = load_summaries(index, SUMMARIES[:2], embedder())

    assert result.deleted == 2 and result.total == 2


def test_stored_returns_every_key_with_its_text_and_labels(index):
    load_summaries(index, SUMMARIES, embedder())

    stored = index.stored()

    assert set(stored) == {s.summary_id for s in SUMMARIES}   # across several pages of 3
    assert stored["customer:C00001"] == {"kind": "customer", "customer_id": "C00001", "region_id": "EMEA",
                                         "text": "Customer C00001 in EMEA"}


def test_large_loads_are_sent_in_batches_of_500(index):
    many = [Summary(f"customer:C{i:05d}", f"Customer {i}", {"kind": "customer"}) for i in range(1200)]
    load_summaries(index, many, embedder())

    assert [n for kind, n in index.client.calls if kind == "put"] == [500, 500, 200]


def test_an_index_with_a_different_vector_size_is_refused():
    client = FakeS3Vectors()
    VectorIndex(client, "b", "sales-summaries", dim=384).ensure()

    with pytest.raises(VectorIndexError, match="size 384.*EMBEDDING_DIM is 5"):
        VectorIndex(client, "b", "sales-summaries", dim=5).ensure()


def test_search_finds_the_closest_summary_and_returns_its_text(index):
    load_summaries(index, SUMMARIES, embedder())

    [best, *_] = search(index, embedder(), "laptop sales in EMEA", top_k=3)

    assert best.summary_id == "catreg:Laptop:EMEA:2026-03"
    assert best.text.startswith("Laptop sales in EMEA")
    assert "text" not in best.metadata
    assert 0 < best.score <= 1


def test_search_can_be_narrowed_by_labels(index):
    load_summaries(index, SUMMARIES, embedder())

    apj = search(index, embedder(), "laptop", top_k=5, region="APJ")
    apj_laptops = search(index, embedder(), "sales", top_k=5, region="APJ", category="Laptop")

    assert {r.metadata["region_id"] for r in apj} == {"APJ"}
    assert [r.summary_id for r in apj_laptops] == ["catreg:Laptop:APJ:2026-03"]


def test_filter_building():
    assert filters() is None
    assert filters(region="EMEA") == {"region_id": {"$eq": "EMEA"}}
    assert filters(region="EMEA", category="Laptop") == {
        "$and": [{"region_id": {"$eq": "EMEA"}}, {"category": {"$eq": "Laptop"}}]
    }
