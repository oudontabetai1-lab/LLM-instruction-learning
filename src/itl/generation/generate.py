"""Self-Instruct style instruction-data generation using the teacher LLM.

The teacher is shown a few example tasks and asked to produce new
instruction/input/output triples as JSON. Results are appended to a JSONL
file so generation can be resumed.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path

from itl.config import GenerationConfig
from itl.teachers.base import TeacherClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_JA = (
    "あなたは言語モデルの指示学習用データセットを作成する専門家です。"
    "多様で具体的、かつ高品質な指示と模範解答を日本語で作成してください。"
)
SYSTEM_PROMPT_EN = (
    "You are an expert at creating instruction-tuning datasets for language models. "
    "Produce diverse, concrete, high-quality instructions with exemplary answers in English."
)

GENERATION_PROMPT = """以下は指示学習データセットのタスク例です:

{examples}

上記とは重複しない新しいタスクを {batch_size} 件作成してください。
- ジャンル(要約、翻訳、コード、推論、創作、質問応答など)を分散させること
- "input" は指示に付随する入力テキスト。不要な場合は空文字列にすること
- "output" は指示への完全で正確な模範解答にすること

次の形式の JSON 配列のみを出力してください(説明文は不要):
[{{"instruction": "...", "input": "...", "output": "..."}}]
"""


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    p = Path(path)
    if not p.exists():
        return records
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, records: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_examples(tasks: list[dict]) -> str:
    blocks = []
    for i, task in enumerate(tasks, 1):
        blocks.append(
            f"### 例 {i}\n"
            + json.dumps(
                {
                    "instruction": task["instruction"],
                    "input": task.get("input", ""),
                    "output": task["output"],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(blocks)


def parse_generated_tasks(text: str) -> list[dict]:
    """Extract instruction/input/output records from the teacher's reply.

    The model may wrap the JSON array in prose or a code fence, so find the
    outermost array and validate each element.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    tasks = []
    for item in data:
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        if not instruction or not output:
            continue
        tasks.append(
            {
                "instruction": instruction,
                "input": str(item.get("input", "")).strip(),
                "output": output,
            }
        )
    return tasks


def validate_seeds(seeds: list[dict]) -> list[dict]:
    """Drop seed records with missing or empty instruction/output fields."""
    valid = []
    for i, seed in enumerate(seeds):
        instruction = str(seed.get("instruction", "")).strip()
        output = str(seed.get("output", "")).strip()
        if not instruction or not output:
            logger.warning(
                "Skipping seed record %d: missing or empty 'instruction'/'output'", i
            )
            continue
        valid.append(seed)
    return valid


def generate_instructions(
    teacher: TeacherClient,
    config: GenerationConfig,
    rng: random.Random | None = None,
    max_consecutive_failures: int = 10,
    sleep_fn=time.sleep,
) -> int:
    """Run the generation loop until ``num_instructions`` records exist.

    Returns the number of newly generated records.

    Raises ``RuntimeError`` if the teacher fails (exception or unparseable
    reply) ``max_consecutive_failures`` times in a row, to avoid retrying
    forever on a persistently broken teacher.
    """
    rng = rng or random.Random()
    seeds = load_jsonl(config.seed_path)
    if not seeds:
        raise FileNotFoundError(f"No seed tasks found at {config.seed_path}")
    seeds = validate_seeds(seeds)
    if not seeds:
        raise FileNotFoundError(
            f"No valid seed tasks (with non-empty 'instruction' and 'output') found at "
            f"{config.seed_path}"
        )

    generated = load_jsonl(config.output_path)
    system_prompt = SYSTEM_PROMPT_JA if config.language == "ja" else SYSTEM_PROMPT_EN
    new_count = 0

    consecutive_failures = 0
    backoff = 1.0
    last_failure_kind = None

    while len(generated) < config.num_instructions:
        pool = seeds + generated
        examples = rng.sample(pool, min(config.num_prompt_examples, len(pool)))
        prompt = GENERATION_PROMPT.format(
            examples=format_examples(examples), batch_size=config.batch_size
        )
        try:
            reply = teacher.complete(prompt, system=system_prompt)
        except Exception:
            logger.exception("Teacher call failed; retrying with a new sample")
            consecutive_failures += 1
            last_failure_kind = "teacher call exception"
        else:
            tasks = parse_generated_tasks(reply)
            if not tasks:
                logger.warning("Could not parse any tasks from teacher reply; skipping batch")
                consecutive_failures += 1
                last_failure_kind = "unparseable teacher reply"
            else:
                consecutive_failures = 0
                backoff = 1.0
                tasks = tasks[: config.num_instructions - len(generated)]
                append_jsonl(config.output_path, tasks)
                generated.extend(tasks)
                new_count += len(tasks)
                logger.info(
                    "Generated %d/%d instructions", len(generated), config.num_instructions
                )
                continue

        if consecutive_failures >= max_consecutive_failures:
            raise RuntimeError(
                f"Teacher failed {consecutive_failures} times in a row "
                f"(last failure: {last_failure_kind}); giving up."
            )
        sleep_fn(backoff)
        backoff = min(backoff * 2, 30.0)

    return new_count
