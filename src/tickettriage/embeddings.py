"""Turn ticket text into vectors, with an on-disk cache.

Two interchangeable backends:
  local   - sentence-transformers on CPU (free; fine for week 1)
  bedrock - Amazon Titan Text Embeddings V2

Rule: one embedding model per vector store. Each model writes to its own folder
(data/embeddings/<tag>/) and every file records the model id, so mixing is detectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tickettriage.config import Settings, embedding_tag, get_settings
from tickettriage.data import SPLIT_NAMES


class LocalEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = f"local:{model_name}"
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vecs, dtype=np.float32)


class BedrockEmbedder:
    def __init__(self, model_id: str, dims: int, region: str, workers: int = 8) -> None:
        import boto3
        from botocore.config import Config

        self._bedrock_model = model_id
        self.dim = dims
        self.workers = workers
        self.model_id = f"bedrock:{model_id}:{dims}"
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),
        )

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True})
        resp = self._client.invoke_model(
            modelId=self._bedrock_model,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(resp["body"].read())["embedding"]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            vecs = list(pool.map(self._embed_one, texts))
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(s: Settings):
    if s.embedding_backend == "local":
        return LocalEmbedder(s.local_embed_model)
    if s.embedding_backend == "bedrock":
        return BedrockEmbedder(s.bedrock_embed_model_id, s.bedrock_embed_dims, s.aws_region)
    raise ValueError(f"EMBEDDING_BACKEND must be 'local' or 'bedrock', got {s.embedding_backend!r}")


def _fingerprint(texts: Sequence[str]) -> str:
    return hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()


def embed_split(
    name: str, texts: Sequence[str], embedder, out_dir: Path, force: bool = False
) -> np.ndarray:
    """Embed `texts`, or load the cached vectors if the texts and model are unchanged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npy, meta_path = out_dir / f"{name}.npy", out_dir / f"{name}.meta.json"
    fingerprint = _fingerprint(texts)

    if npy.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("model_id") == embedder.model_id and meta.get("fingerprint") == fingerprint:
            return np.load(npy)

    vecs = embedder.embed(texts)
    np.save(npy, vecs)
    meta_path.write_text(
        json.dumps(
            {
                "model_id": embedder.model_id,
                "dim": int(vecs.shape[1]),
                "rows": int(vecs.shape[0]),
                "fingerprint": fingerprint,
            },
            indent=2,
        )
    )
    return vecs


def embed_all(s: Settings, embedder, force: bool = False) -> Path:
    processed = s.data_dir / "processed"
    out_dir = s.data_dir / "embeddings" / embedding_tag(s)
    for name in SPLIT_NAMES:
        path = processed / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make data` first.")
        texts = pd.read_parquet(path)["text"].tolist()
        vecs = embed_split(name, texts, embedder, out_dir, force=force)
        print(f"  {name:7s} {vecs.shape[0]:6d} x {vecs.shape[1]}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embed all dataset splits.")
    parser.add_argument("--force", action="store_true", help="re-embed even if cached")
    args = parser.parse_args(argv)

    s = get_settings()
    embedder = get_embedder(s)
    print(f"Embedding with {embedder.model_id} (dim {embedder.dim})")
    out_dir = embed_all(s, embedder, force=args.force)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
