"""Store and search the summaries in Amazon S3 Vectors (FR-19, used by RAG in FR-26).

    vector-load                                   lake -> summaries -> vectors -> index
    vector-search "laptop sales in EMEA"          try a search (hybrid, see hybrid.py)
    vector-search "desktops" --region APJ         search only APJ summaries
    vector-search "laptop sales in EMEA" --plain  meaning only, to compare

Each stored vector:
    key        the summary ID, e.g. "catreg:Laptop:EMEA:2026-03"
    data       384 numbers from the local embedding model
    metadata   the search labels (kind, region_id, category, month, segment, customer_id),
               which a search can filter on, plus "text", the summary itself, so a search
               can return it straight away. "text" is stored but not filterable.

Storing a key that exists replaces it, so re-loading overwrites instead of adding copies.
Keys that are no longer produced are deleted, so the index always matches the lake.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from botocore.exceptions import ClientError

from sales_poc.vectors.embeddings import Embedder, get_embedder
from sales_poc.vectors.hybrid import CANDIDATES, labels_in, rerank
from sales_poc.vectors.summaries import Summary

PUT_BATCH = 500          # S3 Vectors accepts at most 500 vectors per put request
DELETE_BATCH = 500
TEXT_KEY = "text"        # stored with each vector, but not filterable
CUSTOMER_MATCHES = 3     # summaries fetched directly for a customer ID named in a question


class VectorIndexError(RuntimeError):
    """The index exists but does not fit (for example, a different vector size)."""


@dataclass(frozen=True)
class SearchResult:
    summary_id: str
    score: float             # meaning only: 1 = same meaning, 0 = unrelated (1 - cosine distance)
    text: str
    metadata: dict
    matched_labels: int = 0  # hybrid search: labels named in the question that this result has


class VectorIndex:
    """One S3 Vectors index."""

    def __init__(self, client, bucket: str, index: str, dim: int):
        self.client, self.bucket, self.index, self.dim = client, bucket, index, dim

    def _ids(self) -> dict:
        return {"vectorBucketName": self.bucket, "indexName": self.index}

    def ensure(self) -> bool:
        """Create the index if it is missing. True if it was created now."""
        try:
            found = self.client.get_index(**self._ids())["index"]
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "NotFoundException":
                raise
            self.client.create_index(
                **self._ids(),
                dataType="float32",
                dimension=self.dim,
                distanceMetric="cosine",
                metadataConfiguration={"nonFilterableMetadataKeys": [TEXT_KEY]},
            )
            return True
        if found.get("dimension") != self.dim:
            raise VectorIndexError(
                f"Index '{self.index}' holds vectors of size {found.get('dimension')}, but "
                f"EMBEDDING_DIM is {self.dim}. Use a new S3_VECTOR_INDEX name, or delete the old index."
            )
        return False

    def keys(self) -> set[str]:
        keys, token = set(), None
        while True:
            page = self.client.list_vectors(**self._ids(), **({"nextToken": token} if token else {}))
            keys |= {v["key"] for v in page.get("vectors", [])}
            token = page.get("nextToken")
            if not token:
                return keys

    def stored(self) -> dict[str, dict]:
        """{key: metadata} for every stored vector. The metadata includes the summary text,
        so a caller can tell which summaries have changed without embedding them again."""
        stored, token = {}, None
        while True:
            page = self.client.list_vectors(**self._ids(), returnMetadata=True,
                                            **({"nextToken": token} if token else {}))
            stored.update({v["key"]: dict(v.get("metadata", {})) for v in page.get("vectors", [])})
            token = page.get("nextToken")
            if not token:
                return stored

    def put(self, summaries: list[Summary], vectors: list[list[float]]) -> None:
        if len(summaries) != len(vectors):
            raise ValueError("One vector per summary is needed.")
        items = [
            {"key": s.summary_id, "data": {"float32": v}, "metadata": {**s.metadata, TEXT_KEY: s.text}}
            for s, v in zip(summaries, vectors)
        ]
        for start in range(0, len(items), PUT_BATCH):
            self.client.put_vectors(**self._ids(), vectors=items[start:start + PUT_BATCH])

    def delete(self, keys: list[str]) -> None:
        for start in range(0, len(keys), DELETE_BATCH):
            self.client.delete_vectors(**self._ids(), keys=keys[start:start + DELETE_BATCH])

    def query(self, vector: list[float], top_k: int = 5, where: dict | None = None) -> list[SearchResult]:
        response = self.client.query_vectors(
            **self._ids(),
            queryVector={"float32": vector},
            topK=top_k,
            returnMetadata=True,
            returnDistance=True,
            **({"filter": where} if where else {}),
        )
        results = []
        for item in response.get("vectors", []):
            metadata = dict(item.get("metadata", {}))
            text = metadata.pop(TEXT_KEY, "")
            results.append(SearchResult(item["key"], 1 - float(item.get("distance", 1.0)), text, metadata))
        return results


@dataclass(frozen=True)
class LoadResult:
    created_index: bool
    stored: int
    deleted: int
    total: int


def load_summaries(index: VectorIndex, summaries: list[Summary], embedder: Embedder) -> LoadResult:
    """Make the index hold exactly these summaries."""
    created = index.ensure()
    before = index.keys()
    vectors = embedder.embed_documents([s.text for s in summaries])
    index.put(summaries, vectors)
    stale = sorted(before - {s.summary_id for s in summaries})
    index.delete(stale)
    return LoadResult(created_index=created, stored=len(summaries), deleted=len(stale), total=len(index.keys()))


def filters(region: str | None = None, category: str | None = None, month: str | None = None,
            kind: str | None = None) -> dict | None:
    """An S3 Vectors filter from the labels that were given, or None."""
    conditions = [{k: {"$eq": v}} for k, v in
                  (("region_id", region), ("category", category), ("month", month), ("kind", kind)) if v]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(
    index: VectorIndex,
    embedder: Embedder,
    question: str,
    top_k: int = 5,
    hybrid: bool = True,
    **labels,
) -> list[SearchResult]:
    """The summaries that best answer `question`.

    hybrid=True (default): take the CANDIDATES closest by meaning, then move up results whose
    labels the question names (see hybrid.py). A customer ID in the question (C00012) also
    fetches that customer's summary directly by its label, so it can't be missed.
    hybrid=False: plain search by meaning only.
    Keyword labels (region=..., category=..., month=..., kind=...) narrow the search first.
    """
    vector = embedder.embed_query(question)
    where = filters(**labels)
    if not hybrid:
        return index.query(vector, top_k, where)
    candidates = index.query(vector, max(top_k, CANDIDATES), where)
    # A customer ID in the question names one exact summary. All 500 customer summaries read
    # alike, so it is often not among the closest by meaning; fetch it directly instead.
    for customer_id in labels_in(question).get("customer_id", []):
        exact = {"customer_id": {"$eq": customer_id}}
        candidates += index.query(vector, CUSTOMER_MATCHES, exact if where is None else {"$and": [where, exact]})
    unique = list({r.summary_id: r for r in reversed(candidates)}.values())[::-1]   # first copy of each
    return rerank(question, unique, top_k)


def default_index() -> VectorIndex:
    from sales_poc.aws import s3vectors_client
    from sales_poc.config import get_embedding_settings, get_vector_store_settings

    vs = get_vector_store_settings()
    return VectorIndex(s3vectors_client(), vs.s3_vector_bucket, vs.s3_vector_index,
                       get_embedding_settings().embedding_dim)


def load_main(argv: list[str] | None = None) -> int:
    """vector-load: rebuild the index from the clean lake."""
    import time

    from sales_poc.lake.clean import read_all_clean
    from sales_poc.lake.store import s3_store
    from sales_poc.vectors.summaries import build_summaries

    argparse.ArgumentParser(prog="vector-load", description="Load lake summaries into S3 Vectors.").parse_args(
        sys.argv[1:] if argv is None else argv)
    sys.stdout.reconfigure(line_buffering=True)
    started = time.monotonic()
    summaries = build_summaries(read_all_clean(s3_store()))
    print(f"Built {len(summaries)} summaries from the lake. Embedding and storing...")
    index = default_index()
    result = load_summaries(index, summaries, get_embedder())
    print(f"Index '{index.index}': {'created, ' if result.created_index else ''}stored {result.stored}, "
          f"deleted {result.deleted} old, now holds {result.total}. Done in {time.monotonic() - started:.0f}s")
    return 0 if result.total == len(summaries) else 1


def search_main(argv: list[str] | None = None) -> int:
    """vector-search: try a search from the command line."""
    parser = argparse.ArgumentParser(prog="vector-search", description="Search the sales summaries.")
    parser.add_argument("question")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--region")
    parser.add_argument("--category")
    parser.add_argument("--month", help="YYYY-MM")
    parser.add_argument("--kind", choices=["catreg", "region", "customer"])
    parser.add_argument("--plain", action="store_true", help="meaning only, no label re-ranking")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    results = search(default_index(), get_embedder(), args.question, args.top, hybrid=not args.plain,
                     region=args.region, category=args.category, month=args.month, kind=args.kind)
    for r in results:
        labels = "" if args.plain else f"  (labels matched: {r.matched_labels})"
        print(f"{r.score:.3f}  {r.summary_id}{labels}\n       {r.text[:200]}")
    return 0
