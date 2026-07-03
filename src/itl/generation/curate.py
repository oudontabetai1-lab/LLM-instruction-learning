"""Curation: quality filtering, near-duplicate removal, and train/val split."""

from __future__ import annotations

import json
import random
from pathlib import Path

from itl.config import CurationConfig
from itl.generation.generate import load_jsonl


def _tokenize(text: str) -> list[str]:
    """Language-agnostic tokenization: character bigrams work for Japanese too."""
    text = "".join(text.split()).lower()
    if len(text) < 2:
        return [text] if text else []
    return [text[i : i + 2] for i in range(len(text) - 1)]


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (used for a ROUGE-L style score)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b):
            curr.append(prev[j] + 1 if token_a == token_b else max(prev[j + 1], curr[j]))
        prev = curr
    return prev[-1]


def rouge_l_f1(text_a: str, text_b: str) -> float:
    tokens_a, tokens_b = _tokenize(text_a), _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    lcs = lcs_length(tokens_a, tokens_b)
    if lcs == 0:
        return 0.0
    precision = lcs / len(tokens_a)
    recall = lcs / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


def passes_quality_filter(record: dict, config: CurationConfig) -> bool:
    instruction = record.get("instruction", "")
    output = record.get("output", "")
    if len(instruction) < config.min_instruction_chars:
        return False
    if len(output) < config.min_output_chars:
        return False
    # Reject refusals / meta answers that poison training data.
    refusal_markers = ("わかりません", "できません", "I cannot", "I can't", "As an AI")
    return not any(output.startswith(marker) for marker in refusal_markers)


def _dedup_key(record: dict) -> str:
    """Text used for near-duplicate comparison: instruction *and* input.

    A generic instruction (e.g. "次の英文を日本語に翻訳してください") paired with
    different inputs represents distinct training examples and must not be
    collapsed into a single record just because the instruction repeats.
    """
    return record["instruction"] + "\n" + record.get("input", "")


def deduplicate(records: list[dict], threshold: float) -> list[dict]:
    """Drop records whose instruction+input is near-duplicate of an earlier one."""
    kept: list[dict] = []
    for record in records:
        key = _dedup_key(record)
        if any(rouge_l_f1(key, _dedup_key(k)) >= threshold for k in kept):
            continue
        kept.append(record)
    return kept


def curate(config: CurationConfig, seed: int = 42) -> dict[str, int]:
    """Filter, dedupe and split the generated data. Returns record counts."""
    records = load_jsonl(config.input_path)
    if not records:
        raise FileNotFoundError(f"No generated data at {config.input_path} — run 'itl generate' first")
    total = len(records)
    records = [r for r in records if passes_quality_filter(r, config)]
    filtered = len(records)
    records = deduplicate(records, config.similarity_threshold)
    deduped = len(records)

    rng = random.Random(seed)
    rng.shuffle(records)
    val_size = max(1, int(len(records) * config.val_ratio)) if records else 0
    # Never let the val split consume every record — keep at least one for train.
    val_size = min(val_size, max(0, len(records) - 1))
    val, train = records[:val_size], records[val_size:]

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for record in split:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "total": total,
        "after_quality_filter": filtered,
        "after_dedup": deduped,
        "train": len(train),
        "val": len(val),
    }
