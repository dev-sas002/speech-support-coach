import os
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from groq import Groq

from .local_backend import (
    LOCAL_MODEL_NAME,
    LocalGroqClient,
    request_timeout_s,
    usable_api_key,
)


def approx_tokens(text: str) -> int:
    """
    Roughly estimate the number of tokens in a string.

    This uses a simple heuristic of ~4 characters per token and always
    returns at least 1 to avoid downstream division-by-zero issues.
    """
    return max(1, int(len(text) / 4))


@dataclass(slots=True)
class LLMConfig:
    """Configuration for LLM calls."""

    model: str
    temperature: float
    max_tokens: int
    seed: int | None


def _usage_as_dict(usage: Any) -> dict[str, Any] | None:
    """Normalise a provider usage object to a plain dict."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    for method in ("model_dump", "dict", "to_dict"):
        converter = getattr(usage, method, None)
        if callable(converter):
            try:
                return dict(converter())
            except Exception:
                pass
    # Last resort: pick off the fields the SDK documents.
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    collected = {f: getattr(usage, f) for f in fields if hasattr(usage, f)}
    return collected or None


class LLMClient:
    """
    Thin wrapper around the Groq chat completions API.

    The underlying Groq client can be injected for testing by passing a
    preconfigured instance via the `client` parameter.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 180,
        client: Groq | None = None,
    ) -> None:
        api_key = usable_api_key("GROQ_API_KEY")
        if client is not None:
            self.client = client
        elif api_key:
            self.client = Groq(api_key=api_key, timeout=request_timeout_s())
        else:
            # No usable key: run the scripted local agent rather than failing
            # in the constructor. Without this the whole assistant — state
            # machine, personas, logging, metrics — could not be started.
            self.client = LocalGroqClient()

        self.is_local = api_key is None and client is None

        # The default was "penai/gpt-oss-20b": a missing leading "o". The
        # README documents "openai/gpt-oss-20b", so anyone relying on the
        # default got a model-not-found error from the vendor.
        default_model = LOCAL_MODEL_NAME if self.is_local else "openai/gpt-oss-20b"
        model_name = model or os.getenv("GROQ_LLM_MODEL", default_model)
        seed_str = os.getenv("SEED", "")
        try:
            seed = int(seed_str) if seed_str.strip() != "" else None
        except Exception:
            seed = None

        self.config = LLMConfig(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        #: Time to first token of the most recent stream_chat call, in ms.
        #: None until that stream has actually yielded something.
        self.last_first_token_ms: float | None = None

    def _common_params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the parameter dictionary shared by streaming and non-streaming calls."""
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
        }

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> tuple[Generator[str, None, None], float]:
        """
        Stream tokens from the LLM for a chat-style conversation.

        Returns (token_generator, open_stream_latency_ms).

        The second element used to be documented as the first-token latency,
        which it could never be: the tuple is built before the caller has
        consumed a single token, so `first_latency` was always still None and
        the fallback — the time to open the stream — was what came back every
        time. Time to first token is only knowable once the generator runs, so
        it is recorded on `last_first_token_ms` for callers that want it.
        """
        t0 = time.perf_counter()
        params = self._common_params(messages)
        try:
            stream = self.client.chat.completions.create(
                **params,
                stream=True,
            )
        except Exception as exc:
            raise RuntimeError("Failed to start streaming chat completion") from exc

        self.last_first_token_ms = None

        def gen() -> Generator[str, None, None]:
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if not delta:
                    continue
                if self.last_first_token_ms is None:
                    self.last_first_token_ms = (time.perf_counter() - t0) * 1000
                yield delta

        return gen(), (time.perf_counter() - t0) * 1000

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, float, dict[str, Any] | None]:
        """
        Get a full (non-streaming) chat completion response from the LLM.

        Returns a tuple of (text, latency_ms, usage_metadata_dict_or_none).
        """
        t0 = time.perf_counter()
        params = self._common_params(messages)
        try:
            resp = self.client.chat.completions.create(
                **params,
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError("Failed to get chat completion") from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        txt = resp.choices[0].message.content or ""
        # The docstring promises a dict, and callers (logging, the metrics
        # panel) treat it as one — but the Groq SDK returns a pydantic model
        # here, so this returned an object that only looked dict-like.
        usage = _usage_as_dict(getattr(resp, "usage", None))
        return txt, latency_ms, usage
