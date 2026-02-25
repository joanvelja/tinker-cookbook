"""Core XML parser, extraction, normalization, format instructions."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .fields import FieldSpec

_log = logging.getLogger(__name__)

_XML_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)

# Used by mcq.py and format.py for stripping XML tags from text
_XML_STRIP_RE = re.compile(r"</?[^>]+>")

# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------

_COERCE: dict[type, Callable[[str], Any]] = {
    str: lambda v: v,
    int: int,
    float: float,
    bool: lambda v: v.strip().lower() in ("true", "yes", "1"),
    list: lambda v: [item.strip() for item in v.split(",")],
}


def _coerce(value: str, target_type: type) -> Any:
    """Coerce a string value to the target type."""
    fn = _COERCE.get(target_type)
    if fn is None:
        _log.debug("No coercion for type %s", target_type.__name__)
        return None
    try:
        return fn(value)
    except (ValueError, TypeError) as exc:
        _log.debug("Field coerce to %s failed: %s", target_type.__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


def parse(text: str, schema: dict[str, type]) -> dict[str, Any] | None:
    """Parse text into structured fields via XML tag extraction."""
    xml_matches = _XML_TAG_RE.findall(text)
    if not xml_matches:
        return None
    result: dict[str, Any] = {}
    for tag, content in xml_matches:
        if tag in schema:
            coerced = _coerce(content.strip(), schema[tag])
            if coerced is not None:
                result[tag] = coerced
    return result or None


# ---------------------------------------------------------------------------
# Normalization at extraction boundary
# ---------------------------------------------------------------------------


def normalize_fields(
    raw: dict[str, Any], specs: dict[str, FieldSpec]
) -> dict[str, Any]:
    """Apply per-field normalizers from FieldSpecs to raw values."""
    result = dict(raw)
    for key, value in raw.items():
        spec = specs.get(key)
        if spec is not None and spec.normalizer is not None:
            result[key] = spec.normalizer(value)
    return result


def extract_fields(text: str, specs: dict[str, FieldSpec]) -> dict[str, Any] | None:
    """Parse + normalize via FieldSpec normalizers."""
    type_map = {k: v.type for k, v in specs.items()}
    raw = parse(text, type_map)
    if raw is None:
        return None
    return normalize_fields(raw, specs)


# ---------------------------------------------------------------------------
# Format instructions
# ---------------------------------------------------------------------------


def generate_format_instructions(fields: Mapping[str, FieldSpec]) -> str:
    """Generate XML format instruction text from field specs."""
    from .fields import BinaryScoring, EnumScoring, NumericScoring

    lines = ["You MUST include the following XML tags in your response:"]
    for name, spec in fields.items():
        scoring = spec.scoring
        if isinstance(scoring, BinaryScoring):
            lines.append(f"<{name}>{scoring.true_value} or {scoring.false_value}</{name}>")
        elif isinstance(scoring, EnumScoring):
            vals = ", ".join(scoring.values)
            lines.append(f"<{name}>{vals}</{name}>")
        elif isinstance(scoring, NumericScoring):
            lines.append(f"<{name}>number between {scoring.min_val} and {scoring.max_val}</{name}>")
        elif spec.description:
            lines.append(f"<{name}>{spec.description}</{name}>")
        else:
            lines.append(f"<{name}>your {name} here ({spec.type.__name__})</{name}>")
    return "\n".join(lines)
