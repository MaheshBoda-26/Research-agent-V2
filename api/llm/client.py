"""The LLM client: one interface, several OpenAI-compatible backends (A.4).

Ported from V1's proven client and extended per the V2 plan: every attempt is
recorded (``llm_calls`` rows via an injected ``on_call`` sink), token usage and
a best-effort cost estimate are captured, refusals are detected, and
``response_format`` capability is *discovered* (a 400 naming it disables it for
the process instead of being assumed either way).

Backends and their structured-output paths:

* ``nim``        — ``extra_body={"nvext": {"guided_json": <schema>}}``.
* ``openrouter`` — ``response_format`` json_schema (``strict`` off) plus
  ``provider.require_parameters`` so the schema is actually honoured.
* ``openai`` / ``local`` — ``response_format`` json_schema, ``strict=True``.

Rather than trust any of these mechanisms, every response is validated against
the Pydantic schema and a bounded repair loop is run on failure. Constrained
decoding can still emit JSON that is syntactically fine and semantically wrong.

``complete_json`` never raises for a model-or-validation failure. It returns
``None`` and records the reason; the caller decides whether that is fatal.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from config import Settings

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: Response text sometimes arrives inside a markdown fence despite instructions.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

#: HTTP statuses worth retrying: rate limits and transient upstream failures.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Wrapper keys under which a model may nest the requested object.
_WRAPPER_KEYS = ("fields", "result", "output", "response", "data")

#: Best-effort cost table, USD per 1M tokens (input, output). Prices drift;
#: an unknown model records tokens with cost 0.0 rather than a made-up number.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "llama-3.3-70b": (0.23, 0.28),
    "llama-3.1-70b": (0.23, 0.28),
    "llama-3.1-8b": (0.03, 0.03),
    "qwen3-next-80b": (0.10, 0.60),
    "mixtral-8x7b": (0.24, 0.24),
    "deepseek-r1": (0.40, 2.00),
}

_REFUSAL_MARKERS = (
    "i can't assist",
    "i cannot assist",
    "i'm sorry, but",
    "as an ai language model",
    "against my guidelines",
)


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, ignoring braces inside strings.

    A plain ``text[text.find('{'):text.rfind('}')+1]`` breaks as soon as the
    payload contains a brace inside a string, which abstracts routinely do
    (inline JSON snippets, LaTeX, code fragments).
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def candidate_json_payloads(raw: str) -> list[str]:
    """Extraction candidates for a model response, most-likely first."""
    if not raw:
        return []
    candidates: list[str] = [raw.strip()]
    for match in _FENCE_RE.findall(raw):
        candidates.append(match.strip())
    block = _first_json_object(raw)
    if block:
        candidates.append(block)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _unwrap_nested_object(payload: str) -> str | None:
    """Flatten ``{"fields": {...}}``-style wrappers.

    Some chat-tuned models answer a schema request by nesting the requested
    object under a wrapper key (observed live in V1). Pydantic rejects that as
    missing fields, so before giving up, check whether exactly one top-level
    key holds an object that validates on its own.
    """
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    for wrapper in _WRAPPER_KEYS:
        inner = parsed.get(wrapper)
        if isinstance(inner, dict):
            return json.dumps(inner, ensure_ascii=False)
    return None


def parse_into(raw: str, schema: type[SchemaT]) -> SchemaT:
    """Validate a raw model response against ``schema``, trying candidates."""
    candidates = candidate_json_payloads(raw)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return schema.model_validate_json(candidate)
        except (ValidationError, ValueError, TypeError) as exc:
            last_error = exc
        unwrapped = _unwrap_nested_object(candidate)
        if unwrapped is not None and unwrapped != candidate:
            try:
                return schema.model_validate_json(unwrapped)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
    if isinstance(last_error, ValidationError):
        raise last_error
    raise ValidationError.from_exception_data(
        schema.__name__,
        [
            {
                "type": "value_error",
                "loc": (),
                "input": raw,
                "ctx": {"error": str(last_error or "unparseable response")},
            }
        ],
    )


def _brief(exc: Exception, limit: int = 400) -> str:
    text = (
        json.dumps(exc.errors()[:3], default=str)
        if isinstance(exc, ValidationError)
        else str(exc)
    )
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    lowered = model.lower()
    price = (0.0, 0.0)
    for known, known_price in _PRICE_PER_MTOK.items():
        if known in lowered:
            price = known_price
            break
    return round(prompt_tokens / 1e6 * price[0] + completion_tokens / 1e6 * price[1], 6)


def _is_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


class LLMError(RuntimeError):
    """Raised for construction-level failures (missing key)."""


class LLMClient:
    """Synchronous, thread-safe structured-output client.

    Synchronous because the pipeline is: the arXiv client is blocking and model
    inference is blocking, so a run happens in a worker thread and concurrency
    comes from a thread pool plus the semaphore here.

    ``on_call``, when given, receives one dict per HTTP attempt — the
    ``llm_calls`` audit row (plan A.4): stage, run_id, attempt, model, usage,
    cost, ok, and an error string. It must never raise.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        on_call: Any | None = None,
    ) -> None:
        self.settings = settings
        self.on_call: Any = on_call
        self.structured_disabled = False  # set when the provider rejects response_format
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            if not settings.llm_api_key:
                raise LLMError(
                    "No LLM API key configured for provider "
                    f"{settings.llm_provider!r}; run scripts/verify_llm.py and fill .env."
                )
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=float(settings.llm_timeout_seconds),
                max_retries=0,  # retries are handled here so they can be logged
            )
        # Bounds concurrent requests to the provider.
        self._semaphore = threading.BoundedSemaphore(max(1, settings.llm_concurrency))
        self.calls = 0
        self.failures = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    @property
    def model_name(self) -> str:
        return self.settings.llm_model

    def _response_format(self, schema: type[BaseModel]) -> dict[str, Any] | None:
        """Backend-specific structured-output request, or None to prompt+parse."""
        if self.structured_disabled:
            return None
        schema_json = schema.model_json_schema()
        if self.settings.llm_provider == "openrouter":
            # ``strict`` is deliberately False: strict mode requires every
            # property in ``required``, which would misrepresent genuinely
            # optional extraction fields. Validation and repair cover the gap.
            return {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": False, "schema": schema_json},
            }
        if self.settings.llm_provider in ("openai", "local"):
            return {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True, "schema": schema_json},
            }
        return None

    def _extra_body(self, schema: type[BaseModel]) -> dict[str, Any]:
        if self.settings.llm_provider == "nim":
            # vLLM's guided decoding: the documented NIM path for structured
            # output; response_format is not.
            return {"nvext": {"guided_json": schema.model_json_schema()}}
        if self.settings.llm_provider == "openrouter":
            # Only route to providers that actually honour the schema.
            return {"provider": {"require_parameters": True}}
        return {}

    def _usage(self, response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def _record(self, **fields: Any) -> None:
        if self.on_call is None:
            return
        try:
            self.on_call(**fields)
        except Exception:  # pragma: no cover - a sink must never break a run
            logger.exception("llm_calls recorder raised; ignoring")

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        stage: str = "",
        run_id: str = "",
        attempt: int = 1,
        temperature: float = 0.0,
        max_repairs: int | None = None,
    ) -> SchemaT | None:
        """Return a validated ``schema`` instance, or ``None`` on failure."""
        repairs = self.settings.llm_max_repairs if max_repairs is None else max_repairs
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for repair_index in range(1, repairs + 1):
            raw, usage, error = self._chat(messages, schema=schema, temperature=temperature)
            if error is not None:
                self._record(
                    stage=stage, run_id=run_id, attempt=attempt, repair=repair_index,
                    model=self.settings.llm_model, provider=self.settings.llm_provider,
                    ok=False, error=error, cost_usd=0.0,
                )
                self.failures += 1
                return None
            if _is_refusal(raw or ""):
                logger.warning("LLM refused the request (stage=%s)", stage)
                self._record(
                    stage=stage, run_id=run_id, attempt=attempt, repair=repair_index,
                    model=self.settings.llm_model, provider=self.settings.llm_provider,
                    ok=False, error="refusal", cost_usd=0.0,
                )
                self.failures += 1
                return None
            try:
                result = parse_into(raw or "", schema)
            except ValidationError as exc:
                if repair_index >= repairs:
                    logger.warning(
                        "Discarding response after %d repair attempts (stage=%s): %s",
                        repairs, stage, _brief(exc),
                    )
                    self._record(
                        stage=stage, run_id=run_id, attempt=attempt, repair=repair_index,
                        model=self.settings.llm_model, provider=self.settings.llm_provider,
                        ok=False, error=f"validation: {_brief(exc)}",
                        cost_usd=_cost_usd(self.settings.llm_model, *usage),
                    )
                    self.failures += 1
                    return None
                logger.info("Repairing response (attempt %d): %s", repair_index + 1, _brief(exc))
                self._record(
                    stage=stage, run_id=run_id, attempt=attempt, repair=repair_index,
                    model=self.settings.llm_model, provider=self.settings.llm_provider,
                    ok=False, error=f"validation: {_brief(exc)}",
                    prompt_tokens=usage[0], completion_tokens=usage[1],
                    cost_usd=_cost_usd(self.settings.llm_model, *usage),
                )
                messages.append({"role": "assistant", "content": raw or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response did not match the required JSON schema.\n"
                            f"Validation error: {_brief(exc)}\n"
                            "Reply with corrected JSON only. No prose, no code fences."
                        ),
                    }
                )
                continue
            self._record(
                stage=stage, run_id=run_id, attempt=attempt, repair=repair_index,
                model=self.settings.llm_model, provider=self.settings.llm_provider,
                ok=True, error="",
                prompt_tokens=usage[0], completion_tokens=usage[1],
                cost_usd=_cost_usd(self.settings.llm_model, *usage),
            )
            return result
        return None  # pragma: no cover - loop always returns or records

    def complete_text(self, *, system: str, user: str) -> str | None:
        """Plain completion (no schema). ``None`` on any failure."""
        raw, _usage, error = self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            schema=None,
            temperature=0.2,
        )
        if error is not None or _is_refusal(raw or ""):
            self.failures += 1
            return None
        return raw

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: type[BaseModel] | None,
        temperature: float,
    ) -> tuple[str | None, tuple[int, int], str | None]:
        """One logical request with its retry ladder.

        Returns ``(content, (prompt, completion), error)``; exactly one of
        ``content``/``error`` is set. Transport errors are retried with
        exponential backoff + jitter, then returned as an error — never raised.
        """
        request: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if schema is not None:
            response_format = self._response_format(schema)
            if response_format is not None:
                request["response_format"] = response_format
            extra_body = self._extra_body(schema)
            if extra_body:
                request["extra_body"] = extra_body

        last_error: Exception | None = None
        for transport_attempt in range(4):
            with self._semaphore:
                try:
                    self.calls += 1
                    response = self._client.chat.completions.create(**request)
                except Exception as exc:  # noqa: BLE001 - transport varies by provider
                    status = getattr(exc, "status_code", None)
                    if status is not None and status not in _RETRYABLE_STATUS:
                        # A 400 naming response_format means the endpoint does
                        # not support it: disable it and retry once via
                        # prompt+parse (capability discovery, not assumption).
                        if (
                            status == 400
                            and schema is not None
                            and not self.structured_disabled
                            and "response_format" in str(exc)
                        ):
                            self.structured_disabled = True
                            logger.info("Provider rejected response_format; using prompt+parse")
                            return self._chat(messages, schema=schema, temperature=temperature)
                        return None, (0, 0), f"HTTP {status}: {exc}"
                    last_error = exc
                else:
                    try:
                        content = response.choices[0].message.content
                    except (AttributeError, IndexError, KeyError) as exc:
                        return None, (0, 0), f"Malformed completion shape: {exc}"
                    start_time = time.perf_counter()
                    usage = self._usage(response)
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    cost_usd = _cost_usd(self.settings.llm_model, *usage)
                    self.total_prompt_tokens += usage[0]
                    self.total_completion_tokens += usage[1]
                    self.total_cost_usd += cost_usd
                    return content, usage, None

            # Exponential backoff with jitter, so parallel callers do not
            # synchronise their retries against a throttled provider.
            delay = min(30.0, (2**transport_attempt) * 1.0) * (0.5 + random.random())
            logger.info(
                "Retrying LLM call in %.1fs (attempt %d): %s",
                delay, transport_attempt + 1, last_error,
            )
            time.sleep(delay)

        return None, (0, 0), f"LLM request failed after retries: {last_error}"



    def _set_context(self, stage: str, run_id: str) -> None:
        """Set the current stage and run_id for callback tracking."""
        self._current_stage = stage
        self._current_run_id = run_id

    def _clear_context(self) -> None:
        """Clear the current stage and run_id."""
        self._current_stage = ''
        self._current_run_id = ''


class NullCompleter:
    """A completer for runs without a provider.

    Everything degrades by design (§3.3): retrieval and embedding still work,
    LLM stages report ``None`` and the run's ``narrative_status`` / ``degraded``
    flags surface it.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.failures = 0

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        stage: str = "",
        run_id: str = "",
        attempt: int = 1,
        **kwargs: object,
    ) -> BaseModel | None:
        self.failures += 1
        return None

    def complete_text(self, *, system: str, user: str) -> str | None:
        self.failures += 1
        return None


def build_completer(
    settings: Settings, *, on_call: Any | None = None
) -> LLMClient | NullCompleter:
    """Return a working completer, or ``NullCompleter`` when none is possible.

    Never raises: a missing key or model downgrades the run (D3) instead of
    preventing it.
    """
    if not settings.llm_api_key or not settings.llm_model:
        logger.warning("No LLM configured (key/model missing); runs will be degraded")
        return NullCompleter()
    try:
        return LLMClient(settings, on_call=on_call)
    except LLMError as exc:
        logger.warning("No LLM client (%s); continuing without one", exc)
        return NullCompleter()


__all__ = [
    "LLMClient",
    "LLMError",
    "NullCompleter",
    "build_completer",
    "candidate_json_payloads",
    "parse_into",
]
