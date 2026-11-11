"""Dataset preparation: download, clean, and split.

Design (explained in docs/WEEK1.md):
- Intents are split into "launch" intents (known when the model first ships) and
  "future" intents (they appear after launch; only used by the optional new-intents stage).
- Rows of launch intents are split into train / val / test / stream:
    * test   is frozen and used by the evaluation gate in later weeks;
    * stream is replayed as simulated production traffic in week 3.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split

from tickettriage.config import Settings, get_settings

SPLIT_NAMES = ("train", "val", "test", "stream", "future")
# Fractions of the launch-intent rows. train gets the remainder (0.60).
TEST_FRAC, STREAM_FRAC, VAL_FRAC = 0.15, 0.15, 0.10

_PLACEHOLDER = re.compile(r"\{\{(.*?)\}\}")


def clean_text(text: str) -> str:
    """Turn template placeholders like {{Order Number}} into [Order Number], fix whitespace."""
    text = _PLACEHOLDER.sub(r"[\1]", str(text))
    return re.sub(r"\s+", " ", text).strip()


def prepare_frame(raw: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    missing = [c for c in (text_col, label_col) if c not in raw.columns]
    if missing:
        raise KeyError(
            f"Columns {missing} not found. The dataset has {list(raw.columns)}. "
            "Fix TEXT_COL / LABEL_COL in your .env."
        )
    df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "intent"}).copy()
    df["text"] = df["text"].map(clean_text)
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    # Drop exact duplicates (case-insensitive) so no text can leak across splits.
    df = df.loc[~df["text"].str.lower().duplicated()].reset_index(drop=True)
    return df


def split_intents(
    intents: Iterable[str], launch_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministically choose which intents are 'launch' and which are 'future'."""
    ordered = sorted(set(intents))
    if len(ordered) < 2:
        raise ValueError("Need at least two intents to split into launch/future.")
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    n_launch = min(max(round(len(shuffled) * launch_fraction), 1), len(shuffled) - 1)
    return sorted(shuffled[:n_launch]), sorted(shuffled[n_launch:])


def make_splits(
    df: pd.DataFrame, launch_intents: list[str], seed: int
) -> dict[str, pd.DataFrame]:
    launch = df[df["intent"].isin(launch_intents)].reset_index(drop=True)
    future = df[~df["intent"].isin(launch_intents)].reset_index(drop=True)

    rest, test = train_test_split(
        launch, test_size=TEST_FRAC, stratify=launch["intent"], random_state=seed
    )
    rest, stream = train_test_split(
        rest, test_size=STREAM_FRAC / (1 - TEST_FRAC), stratify=rest["intent"], random_state=seed
    )
    train, val = train_test_split(
        rest,
        test_size=VAL_FRAC / (1 - TEST_FRAC - STREAM_FRAC),
        stratify=rest["intent"],
        random_state=seed,
    )
    splits = {"train": train, "val": val, "test": test, "stream": stream, "future": future}
    return {name: frame.reset_index(drop=True) for name, frame in splits.items()}


def save_splits(
    splits: dict[str, pd.DataFrame], settings: Settings, launch: list[str], future: list[str]
) -> None:
    out = settings.data_dir / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    meta = {
        "dataset": settings.dataset_name,
        "seed": settings.seed,
        "launch_fraction": settings.launch_fraction,
        "launch_intents": launch,
        "future_intents": future,
        "rows": {name: len(frame) for name, frame in splits.items()},
        "created": datetime.now(timezone.utc).isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download, clean and split the dataset.")
    parser.add_argument("--force", action="store_true", help="redo even if already prepared")
    args = parser.parse_args(argv)

    s = get_settings()
    out = s.data_dir / "processed"
    if (out / "meta.json").exists() and not args.force:
        print(f"Already prepared in {out}. Use --force to redo (this changes the splits!).")
        return

    from datasets import load_dataset  # imported here so tests don't need the download

    print(f"Loading {s.dataset_name} ...")
    raw = load_dataset(s.dataset_name, split="train").to_pandas()
    df = prepare_frame(raw, s.text_col, s.label_col)
    launch, future = split_intents(df["intent"].unique(), s.launch_fraction, s.seed)
    splits = make_splits(df, launch, s.seed)
    save_splits(splits, s, launch, future)

    print(f"{len(df)} unique tickets, {df['intent'].nunique()} intents")
    print(f"launch intents ({len(launch)}): {', '.join(launch)}")
    print(f"future intents ({len(future)}): {', '.join(future)}")
    for name, frame in splits.items():
        print(f"  {name:7s} {len(frame):6d} rows")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
