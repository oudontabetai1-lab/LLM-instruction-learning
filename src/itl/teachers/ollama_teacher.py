"""Teacher backend served by Ollama (e.g. a 70B-class local model)."""

from __future__ import annotations

import httpx

from itl.config import TeacherConfig
from itl.teachers.base import TeacherClient


class OllamaTeacher(TeacherClient):
    def __init__(self, config: TeacherConfig, timeout: float = 300.0):
        self.config = config
        self._client = httpx.Client(base_url=config.ollama_host, timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
            },
        }
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
