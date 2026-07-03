import json

import pytest

from itl.config import GenerationConfig
from itl.generation.generate import (
    format_examples,
    generate_instructions,
    load_jsonl,
    parse_generated_tasks,
    validate_seeds,
)
from itl.teachers.base import TeacherClient


class TestParseGeneratedTasks:
    def test_plain_json_array(self):
        text = json.dumps(
            [{"instruction": "翻訳して", "input": "hello", "output": "こんにちは"}],
            ensure_ascii=False,
        )
        tasks = parse_generated_tasks(text)
        assert len(tasks) == 1
        assert tasks[0]["output"] == "こんにちは"

    def test_json_in_code_fence_with_prose(self):
        text = '以下が生成結果です:\n```json\n[{"instruction": "要約して", "input": "", "output": "要約結果"}]\n```\n以上です。'
        tasks = parse_generated_tasks(text)
        assert len(tasks) == 1
        assert tasks[0]["instruction"] == "要約して"

    def test_missing_fields_skipped(self):
        text = json.dumps(
            [
                {"instruction": "有効なタスク", "output": "有効な解答"},
                {"instruction": "outputなし"},
                {"output": "instructionなし"},
                "not a dict",
            ],
            ensure_ascii=False,
        )
        tasks = parse_generated_tasks(text)
        assert len(tasks) == 1
        assert tasks[0]["input"] == ""

    def test_invalid_json_returns_empty(self):
        assert parse_generated_tasks("[{broken json") == []
        assert parse_generated_tasks("配列がありません") == []


class FakeTeacher(TeacherClient):
    """Returns a fixed batch of tasks per call."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        batch = [
            {
                "instruction": f"生成されたタスク {self.calls}-{i}",
                "input": "",
                "output": f"生成された解答 {self.calls}-{i}",
            }
            for i in range(3)
        ]
        return json.dumps(batch, ensure_ascii=False)


def test_generate_instructions_loop(tmp_path):
    seed_path = tmp_path / "seeds.jsonl"
    with open(seed_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"instruction": "シード指示", "input": "", "output": "シード解答"}) + "\n")

    output_path = tmp_path / "generated.jsonl"
    config = GenerationConfig(
        seed_path=str(seed_path),
        output_path=str(output_path),
        num_instructions=10,
        batch_size=3,
        num_prompt_examples=2,
    )
    teacher = FakeTeacher()
    new_count = generate_instructions(teacher, config)

    generated = load_jsonl(output_path)
    assert new_count == len(generated)
    assert len(generated) == 10  # truncated to exactly num_instructions
    assert teacher.calls == 4  # ceil(10 / 3)


def test_format_examples_contains_all_tasks():
    tasks = [
        {"instruction": "指示A", "input": "", "output": "解答A"},
        {"instruction": "指示B", "input": "入力B", "output": "解答B"},
    ]
    text = format_examples(tasks)
    assert "指示A" in text
    assert "入力B" in text
    assert "### 例 2" in text


class OversizedBatchTeacher(TeacherClient):
    """Always returns a batch far larger than requested."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        batch = [
            {"instruction": f"タスク{i}", "input": "", "output": f"解答{i}"} for i in range(50)
        ]
        return json.dumps(batch, ensure_ascii=False)


def test_generate_instructions_stops_exactly_at_num_instructions(tmp_path):
    seed_path = tmp_path / "seeds.jsonl"
    with open(seed_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"instruction": "シード指示", "input": "", "output": "シード解答"}) + "\n")

    output_path = tmp_path / "generated.jsonl"
    config = GenerationConfig(
        seed_path=str(seed_path),
        output_path=str(output_path),
        num_instructions=7,
        batch_size=50,
        num_prompt_examples=1,
    )
    teacher = OversizedBatchTeacher()
    new_count = generate_instructions(teacher, config)

    generated = load_jsonl(output_path)
    assert new_count == 7
    assert len(generated) == 7
    assert teacher.calls == 1


class AlwaysRaisingTeacher(TeacherClient):
    """Raises an exception on every call."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        raise RuntimeError("teacher backend unavailable")


class AlwaysUnparseableTeacher(TeacherClient):
    """Returns text that never parses into tasks."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        return "no valid json here"


def _make_config(tmp_path, num_instructions=10):
    seed_path = tmp_path / "seeds.jsonl"
    with open(seed_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"instruction": "シード指示", "input": "", "output": "シード解答"}) + "\n")
    return GenerationConfig(
        seed_path=str(seed_path),
        output_path=str(tmp_path / "generated.jsonl"),
        num_instructions=num_instructions,
        batch_size=3,
        num_prompt_examples=1,
    )


def test_generate_instructions_gives_up_after_repeated_exceptions(tmp_path):
    config = _make_config(tmp_path)
    teacher = AlwaysRaisingTeacher()
    sleeps = []

    with pytest.raises(RuntimeError, match="10 times in a row"):
        generate_instructions(
            teacher,
            config,
            max_consecutive_failures=10,
            sleep_fn=sleeps.append,
        )

    assert teacher.calls == 10
    # No sleep after the 10th (final) failure: it raises immediately instead.
    assert len(sleeps) == 9


def test_generate_instructions_gives_up_after_repeated_unparseable_replies(tmp_path):
    config = _make_config(tmp_path)
    teacher = AlwaysUnparseableTeacher()
    sleeps = []

    with pytest.raises(RuntimeError, match="10 times in a row"):
        generate_instructions(
            teacher,
            config,
            max_consecutive_failures=10,
            sleep_fn=sleeps.append,
        )

    assert teacher.calls == 10
    assert len(sleeps) == 9


def test_generate_instructions_resets_failure_counter_on_success(tmp_path):
    """A teacher that fails a few times then succeeds should not raise."""

    class FlakyTeacher(TeacherClient):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, temperature=None, max_tokens=None):
            self.calls += 1
            if self.calls % 3 != 0:
                raise RuntimeError("transient failure")
            batch = [
                {"instruction": f"タスク{self.calls}-{i}", "input": "", "output": f"解答{i}"}
                for i in range(3)
            ]
            return json.dumps(batch, ensure_ascii=False)

    config = _make_config(tmp_path, num_instructions=6)
    teacher = FlakyTeacher()
    sleeps = []

    new_count = generate_instructions(
        teacher,
        config,
        max_consecutive_failures=10,
        sleep_fn=sleeps.append,
    )

    assert new_count == 6
    # Never accumulated 10 consecutive failures, so no RuntimeError raised.


class TestValidateSeeds:
    def test_valid_seeds_kept(self):
        seeds = [{"instruction": "指示", "input": "", "output": "解答"}]
        assert validate_seeds(seeds) == seeds

    def test_missing_or_empty_fields_skipped(self):
        seeds = [
            {"instruction": "有効な指示", "output": "有効な解答"},
            {"instruction": "", "output": "解答"},
            {"instruction": "指示のみ", "output": ""},
            {"instruction": "指示だけあってoutputキーなし"},
            {"output": "outputだけ"},
        ]
        valid = validate_seeds(seeds)
        assert len(valid) == 1
        assert valid[0]["instruction"] == "有効な指示"

    def test_invalid_seeds_skipped_but_valid_remain(self, tmp_path, caplog):
        seed_path = tmp_path / "seeds.jsonl"
        with open(seed_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"instruction": "有効指示", "output": "有効解答"}) + "\n")
            f.write(json.dumps({"instruction": "", "output": "解答のみ"}) + "\n")
            f.write(json.dumps({"instruction": "指示のみ", "output": ""}) + "\n")

        config = GenerationConfig(
            seed_path=str(seed_path),
            output_path=str(tmp_path / "generated.jsonl"),
            num_instructions=3,
            batch_size=3,
            num_prompt_examples=1,
        )
        teacher = FakeTeacher()
        generate_instructions(teacher, config)
        generated = load_jsonl(tmp_path / "generated.jsonl")
        assert len(generated) == 3

    def test_all_invalid_seeds_raises(self, tmp_path):
        seed_path = tmp_path / "seeds.jsonl"
        with open(seed_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"instruction": "", "output": ""}) + "\n")

        config = GenerationConfig(
            seed_path=str(seed_path),
            output_path=str(tmp_path / "generated.jsonl"),
            num_instructions=3,
        )
        teacher = FakeTeacher()
        with pytest.raises(FileNotFoundError):
            generate_instructions(teacher, config)
