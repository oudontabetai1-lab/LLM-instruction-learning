"""Teacher (upper-tier) LLM clients."""

from itl.config import TeacherConfig
from itl.teachers.base import TeacherClient


def create_teacher(config: TeacherConfig) -> TeacherClient:
    """Instantiate the teacher client selected by ``config.provider``."""
    if config.provider == "ollama":
        from itl.teachers.ollama_teacher import OllamaTeacher

        return OllamaTeacher(config)
    if config.provider == "bedrock":
        from itl.teachers.bedrock_teacher import BedrockTeacher

        return BedrockTeacher(config)
    raise ValueError(f"Unknown teacher provider: {config.provider!r} (expected 'ollama' or 'bedrock')")


__all__ = ["TeacherClient", "create_teacher"]
