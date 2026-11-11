"""Central configuration, read from environment variables (a .env file is loaded if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is only a convenience
    pass

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    artifacts_dir: Path
    seed: int
    dataset_name: str
    text_col: str
    label_col: str
    launch_fraction: float
    embedding_backend: str  # "local" or "bedrock"
    local_embed_model: str
    bedrock_embed_model_id: str
    bedrock_embed_dims: int
    aws_region: str


def get_settings() -> Settings:
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", ROOT / "data")),
        artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts")),
        seed=int(os.getenv("SEED", "42")),
        dataset_name=os.getenv(
            "DATASET_NAME", "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
        ),
        text_col=os.getenv("TEXT_COL", "instruction"),
        label_col=os.getenv("LABEL_COL", "intent"),
        launch_fraction=float(os.getenv("LAUNCH_FRACTION", "0.75")),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "local").lower(),
        local_embed_model=os.getenv(
            "LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        bedrock_embed_model_id=os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
        bedrock_embed_dims=int(os.getenv("BEDROCK_EMBED_DIMS", "512")),
        aws_region=os.getenv("AWS_REGION", "eu-central-1"),
    )


def embedding_tag(s: Settings) -> str:
    """Folder-safe name that identifies the embedding model.

    Embeddings from different models live in different folders, so they can never be mixed.
    """
    if s.embedding_backend == "bedrock":
        model = s.bedrock_embed_model_id.replace(":", "-")
        return f"bedrock__{model}__{s.bedrock_embed_dims}"
    if s.embedding_backend == "local":
        return "local__" + s.local_embed_model.replace("/", "_")
    raise ValueError(f"EMBEDDING_BACKEND must be 'local' or 'bedrock', got {s.embedding_backend!r}")
