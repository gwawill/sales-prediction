"""The embedding wrapper with the real local model (bge-small-en-v1.5)."""

import pytest

from sales_poc.vectors.embeddings import get_embedder


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


DOCS = [
    "Laptop sales in Europe, Middle East and Africa (EMEA), March 2026: 210 orders, $1.20M revenue.",
    "Server sales in Asia Pacific and Japan (APJ), March 2026: 12 orders, $95.0K revenue.",
    "Customer C00013, Carter, Fuller and Mcclure: Enterprise customer in Netherlands.",
]


def test_real_vectors_have_the_configured_size(embedder):
    vectors = embedder.embed_documents(DOCS)
    assert len(vectors) == 3
    assert all(len(v) == embedder.dim == 384 for v in vectors)


def test_a_question_is_closest_to_the_matching_summary(embedder):
    docs = embedder.embed_documents(DOCS)
    for question, best in [
        ("How did laptops sell in Europe in March?", 0),
        ("server revenue in Asia", 1),
        ("Tell me about the Dutch enterprise customer", 2),
    ]:
        q = embedder.embed_query(question)
        scores = [cosine(q, d) for d in docs]
        assert scores.index(max(scores)) == best, (question, scores)


def test_the_instruction_does_not_hurt_the_right_match(embedder):
    """A light safety net: adding the instruction must not make the right summary score
    clearly worse. (It does not prove the instruction helps; that shows over many questions.)"""
    docs = embedder.embed_documents(DOCS)
    question = "How did laptops sell in Europe in March?"
    with_instruction = cosine(embedder.embed_query(question), docs[0])
    without = cosine(embedder.embed_documents([question])[0], docs[0])
    assert with_instruction >= without - 0.05
