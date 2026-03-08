from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator


def default_schema_path() -> Path:
    # src/storage/schema_validation.py 기준으로 프로젝트 루트의 schemas 폴더 찾기
    return Path(__file__).resolve().parents[2] / "schemas" / "result.schema.json"


def load_result_schema(schema_path: Path) -> Dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_result(result: Dict[str, Any], schema_path: Path | None = None) -> None:
    """
    Validate `result.json` against JSON Schema.

    Raises jsonschema.ValidationError on failure.
    """
    schema_path = schema_path or default_schema_path()
    schema = load_result_schema(schema_path)
    Draft202012Validator(schema).validate(result)