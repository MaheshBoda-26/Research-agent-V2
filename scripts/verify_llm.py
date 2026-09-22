"""Live LLM verification (plan Task 1.5, `make verify-llm`).

1. Probe the provider's ``/models`` catalogue and record it as evidence.
2. Ask for a tiny structured object from the preferred model — one real
   completion, validated against a strict Pydantic schema.
3. On success write ``LLM_MODEL=<slug>`` into ``.env`` and print the evidence.

Exit 0 on success, 1 otherwise. This script is the ONLY thing that writes
``LLM_MODEL`` — a guessed slug is how a demo dies at minute zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "api"))

import httpx  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from config import Settings  # noqa: E402

ENV_PATH = PROJECT_ROOT / ".env"

#: Candidates in preference order: strong instruction-following, cheap,
#: and known to honour guided JSON on NIM. The first that passes the probe wins.
PREFERRED = (
    "openai/gpt-oss-20b",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "mistralai/mistral-large-2-instruct",
    "moonshotai/kimi-k2.6",
)


class _Probe(BaseModel):
    """A deliberately tiny strict schema for the live probe."""

    model_config = ConfigDict(extra="forbid")

    city: str
    population_millions: float


PROBE_SYSTEM = (
    "You are a JSON API. Reply with a single JSON object with exactly the "
    'fields "city" (string) and "population_millions" (number). No prose, '
    "no code fences."
)
PROBE_USER = "Return the record for Paris, France."


def _load_env_dotenv() -> None:
    """Minimal .env loader so this script works before anything else does."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.strip().strip('"')
            import os

            os.environ.setdefault(key.strip(), value)


def catalogue(base_url: str, key: str) -> list[str]:
    response = httpx.get(
        f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    ids = [entry.get("id", "") for entry in data.get("data", [])]
    return sorted(i for i in ids if i)


def pick(catalogue_ids: list[str]) -> list[str]:
    """Catalogue ids matching a preferred slug, in preference order."""
    return [cid for preferred in PREFERRED for cid in catalogue_ids if preferred in cid]


def probe(settings: Settings) -> str:
    """One real structured completion; raises on any failure."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=120,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": PROBE_SYSTEM},
            {"role": "user", "content": PROBE_USER},
        ],
        temperature=0.0,
        extra_body={"nvext": {"guided_json": _Probe.model_json_schema()}},
    )
    raw = response.choices[0].message.content or ""
    result = _Probe.model_validate_json(raw)
    usage = getattr(response, "usage", None)
    return (
        f"validated: {_Probe.model_validate(result).model_dump_json()} | "
        f"prompt={getattr(usage, 'prompt_tokens', '?')} "
        f"completion={getattr(usage, 'completion_tokens', '?')}"
    )


def write_model(slug: str) -> None:
    lines = ENV_PATH.read_text().splitlines()
    replaced = False
    out: list[str] = []
    for line in lines:
        if line.startswith("LLM_MODEL="):
            out.append(f"LLM_MODEL={slug}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"LLM_MODEL={slug}")
    ENV_PATH.write_text("\n".join(out) + "\n")


def main() -> int:
    _load_env_dotenv()
    settings = Settings.from_env()
    if not settings.llm_api_key:
        print("FAIL: no API key in environment (.env)")
        return 1

    print(f"catalogue: GET {settings.llm_base_url}/models ...")
    try:
        ids = catalogue(settings.llm_base_url, settings.llm_api_key)
    except Exception as exc:
        print(f"FAIL: catalogue request failed: {exc}")
        return 1
    print(json.dumps({"catalogue_count": len(ids), "sample": ids[:10]}, indent=2))
    if not ids:
        print("FAIL: empty catalogue")
        return 1

    candidates = pick(ids)
    if not candidates:
        print("FAIL: no preferred model found in catalogue; extend PREFERRED")
        return 1

    evidence = ""
    slug = None
    for candidate in candidates:
        print(f"probing: {candidate}")
        trial = type(settings)(
            **{
                **{f: getattr(settings, f) for f in settings.__dataclass_fields__},
                "llm_model": candidate,
            }
        )
        try:
            evidence = probe(trial)
        except Exception as exc:
            print(f"  probe failed: {str(exc)[:140]}")
            continue
        slug = candidate
        break
    if slug is None:
        print("FAIL: every candidate failed the live probe")
        return 1
    print(f"probe ok -> {evidence}")

    write_model(slug)
    print(f"wrote LLM_MODEL={slug} to {ENV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
