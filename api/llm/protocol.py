"""The LLM seam (plan Appendix A.4, §11.2).

Every pipeline stage depends on this Protocol, never on a concrete client, so
the offline test suite can inject a scripted completer and the pipeline can run
with no provider at all (``NullCompleter``). A hallucination is a validation
failure here: ``complete_json`` returns a validated Pydantic instance or
``None`` — never an unvalidated string, and never a raised exception for a
model-quality problem.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class JSONCompleter(Protocol):
    """A structured-output model call that cannot raise on bad model output."""

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        stage: str,
        run_id: str = "",
        attempt: int = 1,
        **kwargs: object,
    ) -> BaseModel | None: ...

    def complete_text(self, *, system: str, user: str) -> str | None: ...


__all__ = ["JSONCompleter"]
