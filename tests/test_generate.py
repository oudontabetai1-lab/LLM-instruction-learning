import json

from itl.config import GenerationConfig
from itl.generation.generate import (
    format_examples,
    generate_instructions,
    load_jsonl,
    parse_generated_tasks,
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
    assert len(generated) >= 10
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
