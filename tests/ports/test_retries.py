"""Tests for llm_ops.retries.retry_until_valid."""

from __future__ import annotations

import pytest

from ports.retries import AttemptOutcome, retry_until_valid


class TestAcceptOnFirstAttempt:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        calls: list[int] = []

        async def _call(i: int) -> str:
            calls.append(i)
            return "hello"

        def _validate(text: str):
            return True, text.upper(), ""

        outcome = await retry_until_valid(_call, validate=_validate, max_retries=2)
        assert outcome == AttemptOutcome(accepted=True, value="HELLO", attempts=1, last_reason="")
        assert calls == [0]


class TestRejectThenAccept:
    @pytest.mark.asyncio
    async def test_accepts_on_second_try(self) -> None:
        rejects: list[tuple[int, str]] = []

        async def _call(i: int) -> int:
            return i  # returns 0, then 1, then 2...

        def _validate(n: int):
            if n < 1:
                return False, None, f"need >= 1, got {n}"
            return True, n, ""

        outcome = await retry_until_valid(
            _call,
            validate=_validate,
            max_retries=3,
            on_reject=lambda i, r: rejects.append((i, r)),
        )
        assert outcome.accepted is True
        assert outcome.value == 1
        assert outcome.attempts == 2
        assert outcome.last_reason == ""
        assert rejects == [(0, "need >= 1, got 0")]


class TestAllAttemptsFail:
    @pytest.mark.asyncio
    async def test_exhausts_and_reports_last_reason(self) -> None:
        rejects: list[tuple[int, str]] = []

        async def _call(i: int) -> int:
            return i

        def _validate(n: int):
            return False, None, f"reject-{n}"

        outcome = await retry_until_valid(
            _call,
            validate=_validate,
            max_retries=2,
            on_reject=lambda i, r: rejects.append((i, r)),
        )
        assert outcome.accepted is False
        assert outcome.value is None
        assert outcome.attempts == 3  # max_retries=2 → 3 total attempts
        assert outcome.last_reason == "reject-2"
        assert rejects == [(0, "reject-0"), (1, "reject-1"), (2, "reject-2")]


class TestExceptionHandling:
    @pytest.mark.asyncio
    async def test_exception_counts_as_attempt_and_continues(self) -> None:
        exceptions: list[tuple[int, str]] = []

        async def _call(i: int) -> str:
            if i == 0:
                raise RuntimeError("boom")
            return "ok"

        def _validate(text: str):
            return True, text, ""

        outcome = await retry_until_valid(
            _call,
            validate=_validate,
            max_retries=3,
            on_exception=lambda i, e: exceptions.append((i, str(e))),
        )
        assert outcome.accepted is True
        assert outcome.value == "ok"
        assert outcome.attempts == 2
        assert exceptions == [(0, "boom")]

    @pytest.mark.asyncio
    async def test_all_exceptions_report_last_reason(self) -> None:
        async def _call(i: int):
            raise RuntimeError(f"boom-{i}")

        def _validate(x):
            return True, x, ""  # never reached

        outcome = await retry_until_valid(
            _call,
            validate=_validate,
            max_retries=1,
            on_exception=lambda _i, _e: None,  # opt-in to swallow
        )
        assert outcome.accepted is False
        assert outcome.attempts == 2
        assert outcome.last_reason == "exception: RuntimeError('boom-1')"

    @pytest.mark.asyncio
    async def test_exception_propagates_when_no_handler(self) -> None:
        """Default behavior: without on_exception, exceptions propagate."""

        async def _call(_i: int):
            raise RuntimeError("boom")

        def _validate(x):
            return True, x, ""

        with pytest.raises(RuntimeError, match="boom"):
            await retry_until_valid(_call, validate=_validate, max_retries=3)


class TestPerAttemptStrategy:
    """Verify `attempt` index lets callers vary behavior per retry (prompt degradation)."""

    @pytest.mark.asyncio
    async def test_attempt_index_drives_call(self) -> None:
        prompts = ["A", "B", "C"]
        used: list[str] = []

        async def _call(i: int) -> str:
            used.append(prompts[i])
            return prompts[i]

        def _validate(p: str):
            if p == "C":
                return True, p, ""
            return False, None, f"not-C: {p}"

        outcome = await retry_until_valid(_call, validate=_validate, max_retries=3)
        assert outcome.accepted is True
        assert outcome.value == "C"
        assert used == ["A", "B", "C"]


class TestParameterValidation:
    @pytest.mark.asyncio
    async def test_negative_max_retries_rejected(self) -> None:
        async def _call(_i):
            return None

        def _validate(_x):
            return True, None, ""

        with pytest.raises(ValueError, match="max_retries"):
            await retry_until_valid(_call, validate=_validate, max_retries=-1)

    @pytest.mark.asyncio
    async def test_zero_retries_means_single_attempt(self) -> None:
        calls = 0

        async def _call(_i):
            nonlocal calls
            calls += 1
            return "x"

        def _validate(_x):
            return False, None, "nope"

        outcome = await retry_until_valid(_call, validate=_validate, max_retries=0)
        assert outcome.accepted is False
        assert outcome.attempts == 1
        assert calls == 1


class TestResolveOnFailure:
    def test_keep_returns_value(self) -> None:
        from ports.retries import resolve_on_failure

        assert resolve_on_failure("keep", keep_value=["fallback"], reason="why") == ["fallback"]

    def test_raise_raises_runtime_error(self) -> None:
        from ports.retries import resolve_on_failure

        with pytest.raises(RuntimeError, match="disk full"):
            resolve_on_failure("raise", keep_value=None, reason="disk full")

    def test_unknown_policy_rejected(self) -> None:
        from ports.retries import resolve_on_failure

        with pytest.raises(ValueError, match="unknown on_failure"):
            resolve_on_failure("bogus", keep_value=None, reason="x")
