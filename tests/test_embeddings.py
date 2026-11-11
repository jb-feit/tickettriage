import numpy as np
from conftest import FakeEmbedder

from tickettriage.embeddings import embed_split


def test_embed_split_caches_and_detects_changes(tmp_path):
    emb = FakeEmbedder()
    texts = ["intent_1 example a", "intent_2 example b"]

    first = embed_split("train", texts, emb, tmp_path)
    again = embed_split("train", texts, emb, tmp_path)
    assert emb.calls == 1  # second call served from cache
    assert np.array_equal(first, again)

    embed_split("train", texts + ["intent_3 example c"], emb, tmp_path)
    assert emb.calls == 2  # texts changed -> recomputed


def test_embed_split_recomputes_when_model_changes(tmp_path):
    a, b = FakeEmbedder(), FakeEmbedder()
    b.model_id = "bedrock:something-else:512"
    texts = ["intent_1 example a"]
    embed_split("train", texts, a, tmp_path)
    embed_split("train", texts, b, tmp_path)
    assert b.calls == 1
