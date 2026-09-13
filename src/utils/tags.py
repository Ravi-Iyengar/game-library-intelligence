"""Shared helper: games.genres/themes/... are stored as JSON-array text."""
from __future__ import annotations

import json


def parse_tag_list(raw_json: str | None) -> list[str]:
    if not raw_json:
        return []
    try:
        return json.loads(raw_json) or []
    except json.JSONDecodeError:
        return []
