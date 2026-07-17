"""
Thin wrapper around OpenRouter's OpenAI-compatible chat completions API.

Every agent stage (classifier, feature extractor, matcher, judge) goes through
this one call site so the model, prompt format, and API key handling stay in
a single place.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Minimal OpenAI-compatible client pointed at OpenRouter."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    def complete(
        self,
        model: str,
        prompt: str,
        image_base64: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Send `prompt` (and optional base64-encoded PNG/JPEG) to `model`.

        Returns the model's text response.
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_base64:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ]
        else:
            content = prompt

        messages.append({"role": "user", "content": content})

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
