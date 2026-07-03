"""LLM Instruction Learning Platform.

Teacher/student instruction-tuning pipeline: a large "teacher" LLM generates
instruction data, a small "student" model is fine-tuned on it (QLoRA) and
served via Ollama, then evaluated by the teacher (LLM-as-judge).
"""

__version__ = "0.1.0"
