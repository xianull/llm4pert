"""Shared utilities for LLM response parsing."""

import json
import re


def extract_json_from_llm_response(raw: str) -> dict:
    """Extract a JSON object from an LLM response with multiple fallbacks.

    Tries in order:
      1. Direct json.loads on the raw string
      2. Extract from markdown code block (```json ... ``` or ``` ... ```)
      3. Regex search for the first top-level { ... } block

    Args:
        raw: Raw LLM response string.

    Returns:
        Parsed dict.

    Raises:
        json.JSONDecodeError: If all parsing strategies fail.
    """
    text = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: markdown code block extraction
    # Match ```json ... ``` or ``` ... ```
    block_pattern = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
    match = block_pattern.search(text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find the first top-level { ... } block via brace matching
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # All strategies failed
    raise json.JSONDecodeError(
        "Could not extract JSON from LLM response", text, 0
    )
