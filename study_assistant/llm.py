"""
LLM access through Groq API using the OpenAI-compatible
Chat Completions endpoint.

Configuration:
    GROQ_API_KEY=your_groq_api_key
    LLM_MODEL=llama-3.3-70b-versatile  # Optional

Only the Python standard library is used (urllib).
"""

from __future__ import annotations
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


class LLMError(Exception):
    pass


def load_dotenv(path: str | Path = ".env") -> None:
    """Load environment variables from .env without overriding existing variables."""

    for p in {
        Path(path),
        Path(__file__).resolve().parent.parent / ".env"
    }:
        if not p.is_file():
            continue

        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            k, v = line.split("=", 1)
            os.environ.setdefault(
                k.strip(),
                v.strip().strip('"').strip("'")
            )


class GroqLLM:

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 180
    ):
        self.provider = "groq"
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @property
    def label(self) -> str:
        return f"groq:{self.model}"

    def _post(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "study-assistant/1.0 (+python)",
            "Authorization": f"Bearer {self.api_key}"
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(
            req, timeout=self.timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 3000,
        json_mode: bool = True
    ) -> str:

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens
        }

        if json_mode:
            payload["response_format"] = {
                "type": "json_object"
            }

        last = "unknown error"

        for attempt in range(4):
            try:
                data = self._post(payload)

                response = (
                    data["choices"][0]["message"]["content"] or ""
                )

                return re.sub(
                    r"<think>.*?</think>",
                    "",
                    response,
                    flags=re.S
                ).strip()

            except urllib.error.HTTPError as exc:

                body = exc.read().decode(
                    "utf-8", errors="ignore"
                )[:300]

                last = f"HTTP {exc.code}: {body}"

                # Retry without JSON mode if unsupported.
                if exc.code == 400 and "response_format" in payload:
                    payload.pop("response_format")
                    continue

                # Handle rate limits and temporary server errors.
                if exc.code in (429, 500, 502, 503, 504):

                    wait = exc.headers.get("Retry-After")

                    time.sleep(
                        min(float(wait), 30)
                        if wait and wait.replace(".", "").isdigit()
                        else 2 ** attempt
                    )

                    continue

                raise LLMError(last) from exc

            except (
                urllib.error.URLError,
                TimeoutError,
                OSError
            ) as exc:

                last = f"Network error: {exc}"
                time.sleep(1 + attempt)

            except (
                KeyError,
                IndexError,
                ValueError
            ) as exc:

                raise LLMError(
                    f"Unexpected response format: {exc}"
                ) from exc

        raise LLMError(last)


def get_llm():
    """
    Initialize and return the Groq LLM client.

    Requires GROQ_API_KEY in the environment or .env file.
    """

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise LLMError(
            "GROQ_API_KEY not found. "
            "Add your Groq API key to the .env file."
        )

    model = os.getenv(
        "LLM_MODEL",
        "llama-3.3-70b-versatile"
    )

    return GroqLLM(
        api_key=api_key,
        model=model
    )


def extract_json(text: str):
    """Extract the first JSON object from a model reply."""

    start = text.find("{")

    if start == -1:
        raise LLMError("No JSON object found in model reply")

    try:
        obj, _ = json.JSONDecoder().raw_decode(
            text[start:]
        )

    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Invalid JSON from model: {exc}"
        ) from exc

    return obj
