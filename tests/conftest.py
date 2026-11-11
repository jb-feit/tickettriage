import zlib

import numpy as np
import pandas as pd
import pytest

from tickettriage.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        seed=7,
        dataset_name="synthetic",
        text_col="instruction",
        label_col="intent",
        launch_fraction=0.75,
        embedding_backend="local",
        local_embed_model="fake/test-model",
        bedrock_embed_model_id="amazon.titan-embed-text-v2:0",
        bedrock_embed_dims=512,
        aws_region="eu-central-1",
    )


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rows = [
        {"text": f"intent_{k} example number {i}", "intent": f"intent_{k}"}
        for k in range(8)
        for i in range(120)
    ]
    return pd.DataFrame(rows)


class FakeEmbedder:
    """Deterministic embedder: each intent gets its own direction, plus per-text noise."""

    model_id = "local:fake/test-model"
    dim = 16

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            intent = t.split(" example")[0]
            base = np.zeros(self.dim, dtype=np.float32)
            base[zlib.crc32(intent.encode()) % self.dim] = 3.0
            noise = np.random.default_rng(zlib.crc32(t.encode())).normal(0, 0.3, self.dim)
            out.append(base + noise.astype(np.float32))
        return np.asarray(out, dtype=np.float32)
