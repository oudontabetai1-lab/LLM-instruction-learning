import pytest

from itl.config import PipelineConfig, load_config
from itl.evaluation.evaluate import parse_judgement
from itl.teachers import create_teacher
from itl.training.train_qlora import build_chat_records


def test_default_config_loads():
    config = load_config("config/default.yaml")
    assert config.teacher.provider in ("ollama", "bedrock")
    assert config.generation.num_instructions > 0
    assert 0 < config.curation.val_ratio < 1


def test_empty_yaml_uses_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    config = load_config(path)
    assert isinstance(config, PipelineConfig)
    assert config.student.ollama_name == "itl-student"


def test_create_teacher_rejects_unknown_provider():
    config = PipelineConfig().teacher
    config.provider = "openai"
    with pytest.raises(ValueError, match="Unknown teacher provider"):
        create_teacher(config)


class TestParseJudgement:
    def test_valid_judgement(self):
        result = parse_judgement('{"score": 8, "reason": "おおむね正確"}')
        assert result == {"score": 8.0, "reason": "おおむね正確"}

    def test_judgement_with_surrounding_prose(self):
        result = parse_judgement('採点結果:\n{"score": 5, "reason": "不完全"}\n以上')
        assert result is not None
        assert result["score"] == 5.0

    def test_out_of_range_score_rejected(self):
        assert parse_judgement('{"score": 15, "reason": "x"}') is None
        assert parse_judgement('{"score": "high", "reason": "x"}') is None

    def test_garbage_rejected(self):
        assert parse_judgement("採点できません") is None


def test_build_chat_records_merges_input():
    records = [
        {"instruction": "翻訳して", "input": "hello", "output": "こんにちは"},
        {"instruction": "俳句を詠んで", "input": "", "output": "古池や"},
    ]
    rows = build_chat_records(records)
    assert rows[0]["messages"][0]["content"] == "翻訳して\n\nhello"
    assert rows[1]["messages"][0]["content"] == "俳句を詠んで"
    assert rows[1]["messages"][1] == {"role": "assistant", "content": "古池や"}
