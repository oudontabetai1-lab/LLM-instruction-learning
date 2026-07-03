import json

import pytest

import itl.evaluation.evaluate as evaluate_module
from itl.config import EvaluationConfig, PipelineConfig, StudentConfig, load_config
from itl.evaluation.evaluate import evaluate, parse_judgement
from itl.teachers import create_teacher
from itl.teachers.base import TeacherClient
from itl.training.train_qlora import build_chat_records


class FakeTeacherClient(TeacherClient):
    """A test double for TeacherClient with a canned reply."""

    def __init__(self, reply: str = '{"score": 8, "reason": "ok"}'):
        self.reply = reply

    def chat(self, messages, temperature=None, max_tokens=None):
        return self.reply


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


class TestEvaluateNumSamplesValidation:
    def test_non_positive_num_samples_raises_value_error(self, tmp_path):
        dataset_path = tmp_path / "val.jsonl"
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"instruction": "指示", "input": "", "output": "解答"}) + "\n")

        config = EvaluationConfig(
            dataset_path=str(dataset_path),
            num_samples=0,
            report_path=str(tmp_path / "report.json"),
        )
        with pytest.raises(ValueError, match="num_samples must be positive"):
            evaluate(teacher=None, student=StudentConfig(), config=config)

    def test_missing_dataset_still_raises_file_not_found(self, tmp_path):
        config = EvaluationConfig(
            dataset_path=str(tmp_path / "does_not_exist.jsonl"),
            num_samples=5,
            report_path=str(tmp_path / "report.json"),
        )
        with pytest.raises(FileNotFoundError):
            evaluate(teacher=None, student=StudentConfig(), config=config)


class TestEvaluateAllSamplesFail:
    def _write_dataset(self, tmp_path, n=3):
        dataset_path = tmp_path / "val.jsonl"
        with open(dataset_path, "w", encoding="utf-8") as f:
            for i in range(n):
                f.write(
                    json.dumps({"instruction": f"指示{i}", "input": "", "output": f"解答{i}"})
                    + "\n"
                )
        return dataset_path

    def test_raises_when_all_student_inferences_fail(self, tmp_path, monkeypatch):
        dataset_path = self._write_dataset(tmp_path, n=3)

        def always_fail(student, prompt, timeout=300.0):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(evaluate_module, "ask_student", always_fail)

        config = EvaluationConfig(
            dataset_path=str(dataset_path),
            num_samples=3,
            report_path=str(tmp_path / "report.json"),
        )
        with pytest.raises(RuntimeError, match="全 3 サンプルの評価に失敗した"):
            evaluate(teacher=FakeTeacherClient(), student=StudentConfig(), config=config)
        assert not (tmp_path / "report.json").exists()

    def test_raises_when_all_judge_replies_unparseable(self, tmp_path, monkeypatch):
        dataset_path = self._write_dataset(tmp_path, n=3)

        monkeypatch.setattr(evaluate_module, "ask_student", lambda student, prompt, timeout=300.0: "回答")

        config = EvaluationConfig(
            dataset_path=str(dataset_path),
            num_samples=3,
            report_path=str(tmp_path / "report.json"),
        )
        with pytest.raises(RuntimeError, match="全 3 サンプルの評価に失敗した"):
            evaluate(
                teacher=FakeTeacherClient(reply="採点できません"),
                student=StudentConfig(),
                config=config,
            )
        assert not (tmp_path / "report.json").exists()


def test_build_chat_records_merges_input():
    records = [
        {"instruction": "翻訳して", "input": "hello", "output": "こんにちは"},
        {"instruction": "俳句を詠んで", "input": "", "output": "古池や"},
    ]
    rows = build_chat_records(records)
    assert rows[0]["messages"][0]["content"] == "翻訳して\n\nhello"
    assert rows[1]["messages"][0]["content"] == "俳句を詠んで"
    assert rows[1]["messages"][1] == {"role": "assistant", "content": "古池や"}
