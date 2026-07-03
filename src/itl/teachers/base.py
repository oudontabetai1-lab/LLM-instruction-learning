"""Abstract interface for teacher LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TeacherClient(ABC):
    """A chat-completion client backed by an upper-tier LLM."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send chat messages ([{"role": ..., "content": ...}]) and return the reply text."""

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Convenience wrapper for a single-turn prompt."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)
