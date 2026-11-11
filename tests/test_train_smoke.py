import json

from conftest import FakeEmbedder

from tickettriage.data import make_splits, save_splits, split_intents
from tickettriage.embeddings import embed_all
from tickettriage.train import train_baseline


def test_pipeline_end_to_end_on_synthetic_data(settings, synthetic_df):
    launch, future = split_intents(synthetic_df["intent"].unique(), 0.75, settings.seed)
    save_splits(make_splits(synthetic_df, launch, settings.seed), settings, launch, future)
    embed_all(settings, FakeEmbedder())

    m = train_baseline(settings)

    assert m["test"]["accuracy"] > 0.9
    assert 0.0 <= m["test"]["auto_route_rate"] <= 1.0
    assert "unseen_intent_probe" in m

    out = settings.artifacts_dir / "baseline"
    assert list(out.rglob("model.joblib")) and list(out.rglob("report.txt"))
    saved = json.loads(next(out.rglob("metrics.json")).read_text())
    assert saved["n_classes"] == len(launch)
