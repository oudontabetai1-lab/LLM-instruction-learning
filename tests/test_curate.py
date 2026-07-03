import json

from itl.config import CurationConfig
from itl.generation.curate import (
    curate,
    deduplicate,
    passes_quality_filter,
    rouge_l_f1,
)


def make_record(instruction="これは十分な長さの指示文です。", output="こちらは十分な長さの模範解答テキストです。" * 2, input_=""):
    return {"instruction": instruction, "input": input_, "output": output}


class TestRougeL:
    def test_identical_texts_score_one(self):
        assert rouge_l_f1("同じテキストです", "同じテキストです") == 1.0

    def test_disjoint_texts_score_zero(self):
        assert rouge_l_f1("あいうえお", "xyzqw") == 0.0

    def test_similar_texts_score_between(self):
        score = rouge_l_f1("次の文章を要約してください", "次の文章を翻訳してください")
        assert 0.5 < score < 1.0

    def test_empty_text(self):
        assert rouge_l_f1("", "テキスト") == 0.0


class TestQualityFilter:
    def config(self):
        return CurationConfig(min_instruction_chars=8, min_output_chars=20)

    def test_good_record_passes(self):
        assert passes_quality_filter(make_record(), self.config())

    def test_short_instruction_rejected(self):
        assert not passes_quality_filter(make_record(instruction="短い"), self.config())

    def test_short_output_rejected(self):
        assert not passes_quality_filter(make_record(output="短い解答"), self.config())

    def test_refusal_rejected(self):
        record = make_record(output="わかりません。その質問には答えられないためです。")
        assert not passes_quality_filter(record, self.config())


class TestDeduplicate:
    def test_near_duplicates_removed(self):
        records = [
            make_record(instruction="次の文章を三行で要約してください"),
            make_record(instruction="次の文章を三行で要約してください。"),
            make_record(instruction="Pythonでソートアルゴリズムを実装してください"),
        ]
        kept = deduplicate(records, threshold=0.7)
        assert len(kept) == 2

    def test_distinct_records_kept(self):
        records = [
            make_record(instruction="次の英文を日本語に翻訳してください"),
            make_record(instruction="フィボナッチ数列を計算するコードを書いてください"),
        ]
        assert len(deduplicate(records, threshold=0.7)) == 2


def test_curate_end_to_end(tmp_path):
    input_path = tmp_path / "generated.jsonl"
    records = [make_record(instruction=f"タスク番号{i}についての十分に長い指示文です") for i in range(20)]
    records.append(make_record(instruction="短い"))  # filtered out
    with open(input_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    config = CurationConfig(
        input_path=str(input_path),
        output_dir=str(tmp_path / "curated"),
        similarity_threshold=0.95,
        val_ratio=0.1,
    )
    stats = curate(config)

    assert stats["total"] == 21
    assert stats["after_quality_filter"] == 20
    assert stats["train"] + stats["val"] == stats["after_dedup"]
    assert stats["val"] >= 1
    assert (tmp_path / "curated" / "train.jsonl").exists()
    assert (tmp_path / "curated" / "val.jsonl").exists()


def test_curate_tiny_dataset_keeps_at_least_one_train_record(tmp_path):
    """With only 2 records and a val_ratio that would consume both, at least
    one record must remain in train (regression test for the empty-train bug).
    """
    input_path = tmp_path / "generated.jsonl"
    records = [
        make_record(instruction="次の英文を日本語に翻訳してください、お願いします"),
        make_record(instruction="フィボナッチ数列を計算するコードを書いてください"),
    ]
    with open(input_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    config = CurationConfig(
        input_path=str(input_path),
        output_dir=str(tmp_path / "curated"),
        similarity_threshold=0.95,
        val_ratio=1.0,
    )
    stats = curate(config)

    assert stats["after_dedup"] == 2
    assert stats["train"] >= 1
    assert stats["train"] + stats["val"] == stats["after_dedup"]
