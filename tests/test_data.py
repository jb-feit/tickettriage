from tickettriage.data import clean_text, make_splits, prepare_frame, split_intents


def test_clean_text_replaces_placeholders_and_whitespace():
    assert clean_text("cancel  order {{Order Number}} \n now") == "cancel order [Order Number] now"


def test_prepare_frame_drops_duplicates_and_empty():
    import pandas as pd

    raw = pd.DataFrame(
        {"instruction": ["Hello", "hello ", "", "Other"], "intent": ["a", "a", "a", "b"]}
    )
    df = prepare_frame(raw, "instruction", "intent")
    assert df["text"].tolist() == ["Hello", "Other"]


def test_split_intents_is_deterministic_and_disjoint():
    intents = [f"i{k}" for k in range(27)]
    a = split_intents(intents, 0.75, seed=1)
    b = split_intents(intents, 0.75, seed=1)
    assert a == b
    launch, future = a
    assert len(launch) == 20 and len(future) == 7
    assert not set(launch) & set(future)


def test_make_splits_have_no_overlap_and_keep_intent_roles(synthetic_df):
    launch, future = split_intents(synthetic_df["intent"].unique(), 0.75, seed=3)
    splits = make_splits(synthetic_df, launch, seed=3)

    for name in ("train", "val", "test", "stream"):
        assert set(splits[name]["intent"]) <= set(launch)
    assert set(splits["future"]["intent"]) == set(future)

    texts = [set(splits[n]["text"]) for n in ("train", "val", "test", "stream", "future")]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            assert not texts[i] & texts[j]

    total_launch = sum(len(splits[n]) for n in ("train", "val", "test", "stream"))
    assert abs(len(splits["test"]) / total_launch - 0.15) < 0.02
    assert abs(len(splits["stream"]) / total_launch - 0.15) < 0.02
    assert abs(len(splits["val"]) / total_launch - 0.10) < 0.02
