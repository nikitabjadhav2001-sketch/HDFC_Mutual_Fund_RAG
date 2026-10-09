import asyncio
import logging
from typing import AsyncIterator, Dict, List

import openai
from openai import AsyncOpenAI

from app.config import get_settings
from app.core.metrics import LLM_FAILURES, observe_llm_usage
from app.core.telemetry import start_child_span

logger = logging.getLogger(__name__)

PUBLIC_ERROR_MESSAGE = (
    "The language model is temporarily unavailable. Please try again in a moment."
)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """Provider failure with a client-safe message (`public_message`)."""

    def __init__(self, message: str, *, public_message: str = PUBLIC_ERROR_MESSAGE) -> None:
        super().__init__(message)
        self.public_message = public_message


class LLMProvider:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )
        self.model = settings.llm_model

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream completion tokens with retry (open only), timeout, usage tracking.

        Retries apply to opening the stream; once tokens have been yielded a
        failure surfaces as `LLMError` instead of replaying the answer.
        """
        settings = get_settings()
        response = await self._open_stream(messages, settings)
        span = start_child_span("llm.stream", **{"llm.model": self.model})
        try:
            async with asyncio.timeout(settings.llm_timeout_seconds):
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        observe_llm_usage(
                            getattr(usage, "prompt_tokens", None),
                            getattr(usage, "completion_tokens", None),
                        )
        except asyncio.TimeoutError as exc:
            LLM_FAILURES.labels(reason="timeout").inc()
            raise LLMError(f"llm stream exceeded {settings.llm_timeout_seconds}s") from exc
        except openai.OpenAIError as exc:
            LLM_FAILURES.labels(reason="stream").inc()
            raise LLMError(f"llm stream failed: {exc}") from exc
        finally:
            span.end()

    async def _open_stream(self, messages: List[Dict[str, str]], settings):
        last_error: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        temperature=0.0,
                        stream_options={"include_usage": True},
                    ),
                    timeout=settings.llm_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
            except openai.APIStatusError as exc:
                if exc.status_code not in _RETRYABLE_STATUS:
                    LLM_FAILURES.labels(reason=f"status_{exc.status_code}").inc()
                    raise LLMError(f"llm returned HTTP {exc.status_code}") from exc
                last_error = exc
            except openai.APIConnectionError as exc:
                last_error = exc
            except openai.OpenAIError as exc:
                LLM_FAILURES.labels(reason="request").inc()
                raise LLMError(f"llm request failed: {exc}") from exc
            if attempt < settings.llm_max_retries:
                await asyncio.sleep(0.5 * (2**attempt))
        LLM_FAILURES.labels(reason="retries_exhausted").inc()
        message = f"llm unavailable after {settings.llm_max_retries} retries"
        raise LLMError(message) from last_error
