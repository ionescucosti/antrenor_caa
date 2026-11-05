from rokid_backend.core.text import normalize_text, question_hash, strip_option_label
from rokid_backend.ingestion.chunking import chunk_blocks
from rokid_backend.ingestion.parsers import Block
from rokid_backend.rag.embeddings import HashEmbedding


def test_normalize_and_hash():
    assert normalize_text("  Care este  Răspunsul? ") == "care este raspunsul"
    assert question_hash("Which Python type is immutable?") == question_hash("which python type is IMMUTABLE")
    assert strip_option_label("C. Tuple") == "Tuple"
    assert strip_option_label("b) dict()") == "dict()"
    assert strip_option_label("Tuple") == "Tuple"


def test_chunking_respects_size_and_keeps_metadata():
    text = " ".join(f"Sentence number {i} is here." for i in range(120))
    chunks = chunk_blocks([Block(text=text, page=3, section="Intro")], size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.page == 3 and c.section == "Intro" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text.split()[0] == "Sentence"
    # overlap: the start of chunk 2 repeats the tail of chunk 1
    assert chunks[1].text[:20] in chunks[0].text + " " + chunks[1].text


def test_chunking_splits_on_block_boundaries():
    chunks = chunk_blocks([Block("alpha beta", page=1), Block("gamma delta", page=2)], size=900, overlap=100)
    assert [(c.page, c.text) for c in chunks] == [(1, "alpha beta"), (2, "gamma delta")]


def test_hash_embedding_is_deterministic_and_normalised():
    emb = HashEmbedding()
    a, b = emb.embed(["Which Python type is immutable?", "Which Python type is immutable?"])
    assert a == b and len(a) == emb.dimension
    assert abs(sum(v * v for v in a) - 1.0) < 1e-6


def test_hash_embedding_similarity_ordering():
    emb = HashEmbedding()
    q, near, far = emb.embed(["Which Python type is immutable?", "Which Python type is immutable", "How do I bake sourdough bread?"])
    dot = lambda x, y: sum(a * b for a, b in zip(x, y))
    assert dot(q, near) > 0.9 > dot(q, far)
