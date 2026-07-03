"""Teacher backend on AWS Bedrock (e.g. Claude) via the Converse API."""

from __future__ import annotations

from itl.config import TeacherConfig
from itl.teachers.base import TeacherClient


class BedrockTeacher(TeacherClient):
    def __init__(self, config: TeacherConfig):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "boto3 is required for the Bedrock teacher. Install with: pip install -e '.[aws]'"
            ) from exc
        self.config = config
        self._client = boto3.client("bedrock-runtime", region_name=config.aws_region)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        system_blocks = [
            {"text": m["content"]} for m in messages if m["role"] == "system"
        ]
        converse_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]
        kwargs = {
            "modelId": self.config.bedrock_model_id,
            "messages": converse_messages,
            "inferenceConfig": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "maxTokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            },
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        response = self._client.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"]
