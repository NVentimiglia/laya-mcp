"""
Jev client — TypeSafe System One, Laya-compatible predict().

Jev is TypeSafe's flagship System One model. It takes the same
state + typed questions shape as Laya (choice / noul / score) and
returns structured answers code can score without parsing prose.

Usage
-----
  pip install typesafe-sdk
  # gitignored jev.key at repo root, or TYPESAFE_API_KEY
  python -c "from jev_client import JevClient; ..."
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

_QUESTION_KEYS = ("type", "instructions", "criteria")
_ENV_NAMES = ("TYPESAFE_API_KEY", "TYPESAFE_DEFAULT_MODEL")


_HERE = Path(__file__).resolve().parent
_KEY_FILES = (_HERE.parent / "jev.key", _HERE / "jev.key")


def _parse_key(raw: str) -> str | None:
    """First line of a key file: a bare key or TYPESAFE_API_KEY=..."""
    line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    if not line or line.startswith("#"):
        return None
    if line.upper().startswith("TYPESAFE_API_KEY="):
        line = line.split("=", 1)[1].strip().strip('"').strip("'")
    return line or None


def _apply_typesafe_key(raw: str) -> None:
    if os.environ.get("TYPESAFE_API_KEY", "").strip():
        return
    key = _parse_key(raw)
    if key:
        os.environ["TYPESAFE_API_KEY"] = key


def jev_key_file() -> str | None:
    """Key from repo-root or laya-jev jev.key, or None."""
    for key_path in _KEY_FILES:
        if key_path.is_file():
            return _parse_key(key_path.read_text(encoding="utf-8"))
    return None


def _load_env_files() -> None:
    """Fill missing TypeSafe vars from jev.key, laya-jev/.env, or repo .env."""
    here = _HERE
    root = here.parent
    for key_path in _KEY_FILES:
        if key_path.is_file():
            _apply_typesafe_key(key_path.read_text(encoding="utf-8"))
            break
    for path in (here / ".env", root / ".env"):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in _ENV_NAMES:
                continue
            if os.environ.get(key, "").strip():
                continue
            os.environ[key] = value.strip().strip('"').strip("'")


_load_env_files()
DEFAULT_MODEL = os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")


def jev_available() -> bool:
    _load_env_files()
    return bool(os.environ.get("TYPESAFE_API_KEY", "").strip())


def normalize_questions(questions: dict[str, Any]) -> dict[str, Any]:
    """Map Laya/eval fixtures onto TypeSafe question dicts.

    - boolean → noul
    - allowed_values become criteria when criteria is missing
    - extra keys are dropped (TypeSafe rejects unknown fields)
    """
    out: dict[str, Any] = {}
    for key, raw in questions.items():
        q = copy.deepcopy(raw)
        allowed = q.pop("allowed_values", None)
        if q.get("type") == "boolean":
            q["type"] = "noul"
        qtype = q.get("type")
        if qtype == "choice" and not q.get("criteria"):
            if not allowed:
                raise ValueError(
                    f"choice {key!r} needs criteria or allowed_values"
                )
            q["criteria"] = {str(v): str(v) for v in allowed}
        if qtype == "score" and not q.get("criteria"):
            raise ValueError(f"score {key!r} needs criteria")
        out[key] = {k: q[k] for k in _QUESTION_KEYS if k in q}
    return out


def flatten_answers(result: Any) -> dict[str, Any]:
    """Flatten a TypeSafe response into Laya-shaped {answers: {...}}."""
    raw_answers = getattr(result, "answers", None)
    if raw_answers is None and isinstance(result, dict):
        raw_answers = result.get("answers", {})
    if raw_answers is None:
        raw_answers = {}

    answers: dict[str, Any] = {}
    for key, ans in raw_answers.items():
        answers[key] = _answer_to_dict(ans)

    model = getattr(result, "model", None)
    usage = getattr(result, "usage", None)
    if usage is not None and hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return {
        "model": model or DEFAULT_MODEL,
        "answers": answers,
        "usage": usage,
    }


def _answer_to_dict(ans: Any) -> dict[str, Any]:
    if hasattr(ans, "model_dump"):
        data = ans.model_dump()
        return {k: v for k, v in data.items() if v is not None}
    if isinstance(ans, dict):
        return dict(ans)
    data: dict[str, Any] = {}
    for field in (
        "type",
        "choice",
        "noul",
        "score",
        "probabilities",
        "confidence",
        "legend",
    ):
        if hasattr(ans, field):
            data[field] = getattr(ans, field)
    return data


def _make_http_client(timeout: float = 60.0) -> Any:
    """httpx2 + certifi. Default truststore TLS recurses on Windows."""
    import ssl  # noqa: PLC0415

    import certifi  # noqa: PLC0415
    import httpx2  # noqa: PLC0415

    ctx = ssl.create_default_context(cafile=certifi.where())
    return httpx2.Client(verify=ctx, timeout=timeout)


class JevClient:
    """Thin TypeSafe wrapper with Laya's predict(state, questions) seam."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "").strip() or None
        self._client = self._connect(self._api_key)

    def _connect(self, api_key: str | None) -> Any:
        from typesafe_sdk import TypeSafeClient  # noqa: PLC0415

        return TypeSafeClient(
            api_key=api_key,
            model=self.model,
            http_client=_make_http_client(),
        )

    def _switch_to_key_file(self) -> bool:
        """Reconnect with the jev.key key when it differs from the one in use."""
        key = jev_key_file()
        if not key or key == self._api_key:
            return False
        print("Jev rejected the API key; retrying with jev.key.", file=sys.stderr, flush=True)
        self._client.close()
        self._api_key = key
        self._client = self._connect(key)
        return True

    def predict(
        self,
        state: Any,
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        from typesafe_sdk import (  # noqa: PLC0415
            TypeSafeAuthenticationError,
            TypeSafePermissionDeniedError,
        )

        if state is None:
            raise ValueError("Jev state cannot be None")
        normalized = normalize_questions(questions)
        try:
            result = self._client.system_one(state=state, questions=normalized)
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError):
            # A stale env or .env key can shadow jev.key; fall back to it once.
            if not self._switch_to_key_file():
                raise
            result = self._client.system_one(state=state, questions=normalized)
        return flatten_answers(result)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
