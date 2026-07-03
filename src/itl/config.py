"""Typed pipeline configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TeacherConfig(BaseModel):
    provider: str = "ollama"  # "ollama" | "bedrock"
    model: str = "llama3.3:70b"
    ollama_host: str = "http://localhost:11434"
    bedrock_model_id: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    aws_region: str = "ap-northeast-1"
    temperature: float = 0.9
    max_tokens: int = 2048


class StudentConfig(BaseModel):
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ollama_name: str = "itl-student"
    ollama_host: str = "http://localhost:11434"


class GenerationConfig(BaseModel):
    seed_path: str = "data/seed_tasks.jsonl"
    output_path: str = "data/generated/instructions.jsonl"
    num_instructions: int = 1000
    batch_size: int = 5
    num_prompt_examples: int = 3
    language: str = "ja"


class CurationConfig(BaseModel):
    input_path: str = "data/generated/instructions.jsonl"
    output_dir: str = "data/curated"
    similarity_threshold: float = 0.7
    min_instruction_chars: int = 8
    min_output_chars: int = 20
    val_ratio: float = 0.05


class TrainingConfig(BaseModel):
    dataset_dir: str = "data/curated"
    output_dir: str = "outputs/checkpoints"
    epochs: int = 3
    learning_rate: float = 2.0e-4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class ExportConfig(BaseModel):
    llama_cpp_dir: str = "~/llama.cpp"
    quantization: str = "q4_k_m"
    output_dir: str = "outputs/gguf"


class EvaluationConfig(BaseModel):
    dataset_path: str = "data/curated/val.jsonl"
    num_samples: int = 50
    report_path: str = "outputs/eval/report.json"


class AwsConfig(BaseModel):
    s3_bucket: str = ""
    s3_prefix: str = "llm-itl"


class PipelineConfig(BaseModel):
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    aws: AwsConfig = Field(default_factory=AwsConfig)


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return PipelineConfig.model_validate(raw)
