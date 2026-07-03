"""LLM-as-judge evaluation: the teacher scores the student's answers."""

from __future__ import annotations

import json
import logging
import re
import statistics
from pathlib import Path

import httpx

from itl.config import EvaluationConfig, StudentConfig
from itl.generation.generate import load_jsonl
from itl.teachers.base import TeacherClient

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "あなたは言語モデルの応答を採点する厳格な審査員です。"
    "正確性・指示への忠実さ・有用性の観点で 1〜10 点で採点してください。"
)

JUDGE_PROMPT = """次のタスクに対するモデルの応答を採点してください。

## 指示
{instruction}

## 入力
{input}

## 参考解答(教師モデルによる模範解答)
{reference}

## 採点対象の応答
{answer}

1(全く不正確)〜10(完璧)で採点し、次の JSON のみを出力してください:
{{"score": <1-10>, "reason": "<短い理由>"}}
"""


def ask_student(student: StudentConfig, prompt: str, timeout: float = 300.0) -> str:
    """Query the fine-tuned student model through Ollama."""
    with httpx.Client(base_url=student.ollama_host, timeout=timeout) as client:
        response = client.post(
            "/api/chat",
            json={
                "model": student.ollama_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def parse_judgement(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    score = data.get("score")
    if not isinstance(score, (int, float)) or not 1 <= score <= 10:
        return None
    return {"score": float(score), "reason": str(data.get("reason", ""))}


def evaluate(
    teacher: TeacherClient,
    student: StudentConfig,
    config: EvaluationConfig,
) -> dict:
    """Score the student on the validation set and write a JSON report."""
    if config.num_samples <= 0:
        raise ValueError("evaluation.num_samples must be positive")
    records = load_jsonl(config.dataset_path)[: config.num_samples]
    if not records:
        raise FileNotFoundError(f"No evaluation data at {config.dataset_path} — run 'itl curate' first")

    results = []
    for i, record in enumerate(records, 1):
        prompt = record["instruction"]
        if record.get("input"):
            prompt += "\n\n" + record["input"]
        try:
            answer = ask_student(student, prompt)
        except Exception:
            logger.exception("Student inference failed on sample %d", i)
            continue
        judge_reply = teacher.complete(
            JUDGE_PROMPT.format(
                instruction=record["instruction"],
                input=record.get("input", "") or "(なし)",
                reference=record["output"],
                answer=answer,
            ),
            system=JUDGE_SYSTEM,
            temperature=0.0,
        )
        judgement = parse_judgement(judge_reply)
        if judgement is None:
            logger.warning("Could not parse judgement for sample %d", i)
            continue
        results.append({**record, "student_answer": answer, **judgement})
        logger.info("Evaluated %d/%d (score=%.0f)", i, len(records), judgement["score"])

    if not results:
        raise RuntimeError(
            f"全 {len(records)} サンプルの評価に失敗した"
            "(生徒推論失敗または judge 応答パース不能)。詳細はログを参照。"
        )

    scores = [r["score"] for r in results]
    report = {
        "model": student.ollama_name,
        "num_evaluated": len(results),
        "mean_score": statistics.mean(scores) if scores else None,
        "median_score": statistics.median(scores) if scores else None,
        "results": results,
    }
    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
