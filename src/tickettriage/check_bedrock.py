"""Smoke-test Bedrock access early, so a permissions problem doesn't surprise you in week 3.

    python -m tickettriage.check_bedrock            # embeddings (Titan Text Embeddings V2)
    python -m tickettriage.check_bedrock --judge    # also test an LLM (set BEDROCK_JUDGE_MODEL_ID)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from tickettriage.config import get_settings
from tickettriage.embeddings import BedrockEmbedder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Bedrock access.")
    parser.add_argument("--judge", action="store_true", help="also test the LLM judge model")
    args = parser.parse_args(argv)

    s = get_settings()
    print(f"Region: {s.aws_region}")
    embedder = BedrockEmbedder(s.bedrock_embed_model_id, s.bedrock_embed_dims, s.aws_region)
    vec = embedder.embed(["I want to cancel my order"])[0]
    print(f"Embedding OK: dim={len(vec)}, norm={np.linalg.norm(vec):.3f}")

    if args.judge:
        model_id = os.getenv("BEDROCK_JUDGE_MODEL_ID", "").strip()
        if not model_id:
            print("Set BEDROCK_JUDGE_MODEL_ID in .env to a model/inference profile id.")
            return 1
        import boto3

        client = boto3.client("bedrock-runtime", region_name=s.aws_region)
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Reply with the single word: OK"}]}],
            inferenceConfig={"maxTokens": 10, "temperature": 0},
        )
        print("LLM OK:", resp["output"]["message"]["content"][0]["text"].strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
