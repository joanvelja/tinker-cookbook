"""YAML-driven prompt system for debate environments."""

from __future__ import annotations

import functools
import hashlib
import logging
import re
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jinja2
import jinja2.meta
import jinja2.sandbox
import yaml

from ..types import (
    DebateState,
    PHASE_DONE,
    Phase,
    Role,
    ThinkVisibility,
    TRIGGER_BOUNDARY,
    TRIGGER_FINAL,
    current_phase,
)
from ..scoring.fields import (
    EnumScoring,
    FieldSpec,
    _resolve_fields,
    validate_type_scoring,
    _TYPE_MAP,
)
from ..scoring.parsing import generate_format_instructions

_PROMPTS_DIR = Path(__file__).parent

# Sentinel tokens for two-phase rendering (template injection prevention).
_SENTINEL_PREFIX = "__SENTINEL_"
_SENTINEL_SUFFIX = f"_{uuid.uuid4().hex[:8]}__"
_INJECTABLE_KEYS = ("task_prompt", "answer", "answer_a", "answer_b")

_ROLE_NAMES = {"judge", "debater_a", "debater_b"}
_TAG_NAME_RE = re.compile(r"^\w+$")
_UTILITY_BLOCK_NAMES = ("_matcher", "_grader")

# Every key that render-time lookups may request.  'default' is expanded into
# these at compile time so no runtime fallback path exists.
_ALL_LOOKUP_KEYS: tuple[str, ...] = (
    *(p.value for p in Phase),
    PHASE_DONE,
    TRIGGER_FINAL,
    TRIGGER_BOUNDARY,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinaryJudgeTemplate:
    system: str
    user: jinja2.Template
    positive: str
    negative: str


@dataclass(frozen=True)
class ThinkConfig:
    """Per-phase thinking configuration for a role."""

    visibility: ThinkVisibility
    tag: str = "thinking"


_VISIBILITY_DESCRIPTIONS: dict[ThinkVisibility, str] = {
    ThinkVisibility.PRIVATE: "Your reasoning is private — other participants will NOT see it.",
    ThinkVisibility.VISIBLE_TO_JUDGE: "The judge will see your reasoning but your opponent will NOT.",
    ThinkVisibility.OPEN: "All reasoning is visible to all participants.",
}


@dataclass(frozen=True)
class DebatePrompts:
    system: dict[str, dict[str, jinja2.Template]]
    user: dict[str, dict[str, jinja2.Template]]
    question: dict[str, jinja2.Template]
    think: dict[str, dict[str, ThinkConfig]]
    prefill: dict[str, dict[str, jinja2.Template]]
    fields: dict[str, dict[str, dict[str, FieldSpec]]]
    content_hash: str
    source_ref: str
    binary_judges: dict[str, BinaryJudgeTemplate] = field(default_factory=dict)
    opponent_wrap: dict[str, jinja2.Template] | None = None

    def render_system(self, state: DebateState, viewer: Role) -> str:
        """Render system template for viewer. Strict lookup, no fallback."""
        role = viewer.value
        phase = current_phase(state)
        templates = self.system[role]
        if phase not in templates:
            raise KeyError(
                f"No system template for role={role}, phase={phase} "
                f"in {self.source_ref}. Available keys: {sorted(templates.keys())}"
            )
        ctx = _build_context(state, viewer)
        return _two_phase_render(templates[phase], ctx, state, viewer)

    def render_question(self, state: DebateState, viewer: Role) -> str | None:
        """Render question template. None if role not in section."""
        role = viewer.value
        if role not in self.question:
            return None
        tmpl = self.question[role]
        ctx = _build_context(state, viewer)
        result = _two_phase_render(tmpl, ctx, state, viewer)
        return result if result.strip() else None

    def get_think_instruction(
        self, state: DebateState, viewer: Role, trigger: str | None = None
    ) -> str | None:
        """Resolve think config -> instruction string or None."""
        role = viewer.value
        if role not in self.think:
            return None
        cfg = self.think[role]
        phase = trigger or current_phase(state)
        if phase not in cfg:
            return None
        tc = cfg[phase]
        if tc.visibility == ThinkVisibility.DISABLED:
            return None
        desc = _VISIBILITY_DESCRIPTIONS.get(tc.visibility, "")
        return f"Use <{tc.tag}>...</{tc.tag}> tags for your reasoning. {desc}"

    def get_think_visibility(self) -> dict[Role, ThinkVisibility]:
        """Extract uniform non-DISABLED visibility per role from think config."""
        result: dict[Role, ThinkVisibility] = {}
        for role_str, phases in self.think.items():
            role = Role(role_str)
            visibilities = {
                tc.visibility for tc in phases.values() if tc.visibility != ThinkVisibility.DISABLED
            }
            if visibilities:
                assert len(visibilities) == 1, (
                    f"Mixed visibility for {role_str} should have been caught at load time"
                )
                result[role] = next(iter(visibilities))
        return result

    def render_prefill(
        self, state: DebateState, viewer: Role, trigger: str | None = None
    ) -> str | None:
        """Render prefill template. None if empty/absent."""
        role = viewer.value
        if role not in self.prefill:
            return None
        cfg = self.prefill[role]
        phase = trigger or current_phase(state)
        if phase not in cfg:
            return None
        ctx = _build_context(state, viewer)
        result = _two_phase_render(cfg[phase], ctx, state, viewer)
        return result.strip() if result.strip() else None

    def render_user(
        self, state: DebateState, viewer: Role, trigger: str | None = None
    ) -> str | None:
        """Render user template. Returns None if result is empty (skip message)."""
        role = viewer.value

        # 1. Phase instruction
        phase_text = None
        if role in self.user:
            triggers = self.user[role]
            phase = trigger or current_phase(state)
            if phase in triggers:
                ctx = _build_context(state, viewer)
                rendered = _two_phase_render(triggers[phase], ctx, state, viewer)
                if rendered.strip():
                    phase_text = rendered.strip()

        # 2. Think instruction
        think_text = self.get_think_instruction(state, viewer, trigger=trigger)

        # 3. Field instructions
        field_text = self.get_field_instructions(role, trigger) if trigger else None

        parts = [p for p in [phase_text, think_text, field_text] if p]
        return "\n\n".join(parts) if parts else None

    def get_field_instructions(self, role: str, trigger: str) -> str | None:
        """Auto-generate XML format instructions from fields. None if no fields."""
        role_fields = self.fields.get(role)
        if not role_fields:
            return None
        trigger_fields = role_fields.get(trigger)
        if not trigger_fields:
            return None
        return generate_format_instructions(trigger_fields)

    def get_field_specs(self, role: str, trigger: str) -> dict[str, FieldSpec] | None:
        """Get FieldSpec dict for a role/trigger. None if no fields defined."""
        role_fields = self.fields.get(role)
        if not role_fields:
            return None
        trigger_fields = role_fields.get(trigger)
        if not trigger_fields:
            return None
        return dict(trigger_fields)

    def get_field_names(self, role: str, trigger: str) -> list[str] | None:
        """Get field names for a role/trigger. None if no fields defined."""
        role_fields = self.fields.get(role)
        if not role_fields:
            return None
        trigger_fields = role_fields.get(trigger)
        if not trigger_fields:
            return None
        return list(trigger_fields.keys())

    def render_opponent_wrap(self, text: str, label: str, phase: str, viewer: Role) -> str:
        """Render opponent wrap template for the given viewer.

        Uses "debater" template for DEBATER_A/DEBATER_B viewers,
        "judge" template for JUDGE viewer. Variables: text, label, phase.
        """
        if self.opponent_wrap is None:
            raise ValueError("render_opponent_wrap called but opponent_wrap is None")
        key = "judge" if viewer == Role.JUDGE else "debater"
        if key not in self.opponent_wrap:
            raise KeyError(
                f"No opponent_wrap template for key={key} "
                f"in {self.source_ref}. Available keys: {sorted(self.opponent_wrap.keys())}"
            )
        return self.opponent_wrap[key].render(text=text, label=label, phase=phase)

    def get_binary_judge_template(
        self, name: Literal["matcher", "grader"]
    ) -> BinaryJudgeTemplate | None:
        return self.binary_judges.get(name)


def check_ab_symmetry(prompts: DebatePrompts) -> list[str]:
    """Return warning messages if debater_a and debater_b have asymmetric phase keys."""
    warnings: list[str] = []

    # question (flat): check both roles present
    q_roles = set(prompts.question.keys())
    for r in ("debater_a", "debater_b"):
        if r not in q_roles:
            warnings.append(f"question: missing role '{r}'")

    # system, user, prefill (nested): compare phase key sets
    for section_name, section in [
        ("system", prompts.system),
        ("user", prompts.user),
        ("prefill", prompts.prefill),
    ]:
        a_keys = set(section.get("debater_a", {}).keys())
        b_keys = set(section.get("debater_b", {}).keys())
        if a_keys != b_keys:
            warnings.append(
                f"{section_name}: debater_a phases {sorted(a_keys)} != debater_b phases {sorted(b_keys)}"
            )

    # think: compare phase key sets (values are now ThinkConfig)
    if prompts.think:
        a_think = prompts.think.get("debater_a", {})
        b_think = prompts.think.get("debater_b", {})
        if set(a_think.keys()) != set(b_think.keys()):
            warnings.append(
                f"think: debater_a phases {sorted(a_think.keys())} != debater_b phases {sorted(b_think.keys())}"
            )

    # fields: compare trigger→field-name sets per role
    if prompts.fields:
        a_fields = prompts.fields.get("debater_a", {})
        b_fields = prompts.fields.get("debater_b", {})
        a_triggers = set(a_fields.keys())
        b_triggers = set(b_fields.keys())
        if a_triggers != b_triggers:
            warnings.append(
                f"fields: debater_a triggers {sorted(a_triggers)} != debater_b triggers {sorted(b_triggers)}"
            )
        for trigger in a_triggers & b_triggers:
            a_names = set(a_fields[trigger].keys())
            b_names = set(b_fields[trigger].keys())
            if a_names != b_names:
                warnings.append(
                    f"fields.{trigger}: debater_a fields {sorted(a_names)} != debater_b fields {sorted(b_names)}"
                )

    return warnings


def _sentinel(key: str) -> str:
    return f"{_SENTINEL_PREFIX}{key}{_SENTINEL_SUFFIX}"


def _build_context(state: DebateState, viewer: Role) -> dict:
    schedule = state.spec.schedule
    phase = current_phase(state)
    if state.slot_index < len(schedule):
        round_index = schedule[state.slot_index].round_index
    else:
        round_index = -1
    num_rounds = max((s.round_index for s in schedule), default=0) + 1

    answer_by_role = state.spec.problem.answer_by_role
    has_assigned = bool(answer_by_role and viewer in answer_by_role and answer_by_role[viewer])

    _tv = state.spec.think_visibility
    _viewer_vis = _tv.get(viewer, ThinkVisibility.DISABLED)

    return {
        "task_prompt": _sentinel("task_prompt"),
        "viewer_role": viewer.value,
        "phase": phase,
        "round_index": round_index,
        "num_rounds": num_rounds,
        "is_first_round": round_index == 0,
        "is_last_round": round_index == num_rounds - 1,
        "protocol_kind": state.spec.protocol_kind,
        "answer": _sentinel("answer"),
        "answer_a": _sentinel("answer_a"),
        "answer_b": _sentinel("answer_b"),
        "has_assigned_answer": has_assigned,
        "think_visibility": _viewer_vis.value,
    }


def _actual_values(state: DebateState, viewer: Role) -> dict[str, str]:
    """Map sentinel keys to their real values for post-Jinja replacement."""
    answer_by_role = state.spec.problem.answer_by_role or {}
    return {
        "task_prompt": state.spec.problem.task_prompt,
        "answer": answer_by_role.get(viewer, ""),
        "answer_a": answer_by_role.get(Role.DEBATER_A, ""),
        "answer_b": answer_by_role.get(Role.DEBATER_B, ""),
    }


def _two_phase_render(tmpl: jinja2.Template, ctx: dict, state: DebateState, viewer: Role) -> str:
    """Render template in two phases to prevent template injection via user data."""
    # Phase 1: Jinja renders with sentinels standing in for user-supplied values.
    rendered = tmpl.render(ctx)
    # Phase 2: Replace sentinels with actual values (plain string substitution).
    actuals = _actual_values(state, viewer)
    for key in _INJECTABLE_KEYS:
        rendered = rendered.replace(_sentinel(key), actuals[key])
    return rendered


# ---------------------------------------------------------------------------
# Migration lint: detect removed V1 variables before compilation
# ---------------------------------------------------------------------------


def _check_migration_lint(d: dict) -> None:
    """Scan raw template strings for removed V1 variables before compilation."""
    for section_name in ("system", "user", "question", "prefill"):
        block = d.get(section_name, {})
        _scan_block_for_removed_vars(block, section_name)


def _scan_block_for_removed_vars(block: object, section_name: str, path: str = "") -> None:
    if isinstance(block, str):
        ast = _jinja_env.parse(block)
        undeclared = jinja2.meta.find_undeclared_variables(ast)
        if "reasoning_instruction" in undeclared:
            raise ValueError(
                f"Template at {section_name}{path} uses removed variable 'reasoning_instruction'. "
                "In V2, thinking mode is controlled by the renderer, not prompts."
            )
    elif isinstance(block, dict):
        for key, val in block.items():
            _scan_block_for_removed_vars(val, section_name, f".{key}")


def normalize_binary_verdict_token(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    token = stripped.split()[0].rstrip(string.punctuation)
    return token.upper() if token else None


def _validate_binary_judge_blocks(d: dict) -> None:
    for block_name in _UTILITY_BLOCK_NAMES:
        block = d.get(block_name)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"{block_name}: expected mapping, got {type(block).__name__}")
        required_keys = {"system", "user", "positive", "negative"}
        missing = required_keys - set(block)
        extra = set(block) - required_keys
        if missing:
            raise ValueError(f"{block_name}: missing required keys {sorted(missing)}")
        if extra:
            raise ValueError(f"{block_name}: unknown keys {sorted(extra)}")
        if not isinstance(block["system"], str):
            raise ValueError(f"{block_name}.system: expected str")
        if not isinstance(block["user"], str):
            raise ValueError(f"{block_name}.user: expected str")
        if not isinstance(block["positive"], str):
            raise ValueError(f"{block_name}.positive: expected str")
        if not isinstance(block["negative"], str):
            raise ValueError(f"{block_name}.negative: expected str")

        positive = normalize_binary_verdict_token(block["positive"])
        negative = normalize_binary_verdict_token(block["negative"])
        if positive is None:
            raise ValueError(f"{block_name}.positive: expected non-empty verdict token")
        if negative is None:
            raise ValueError(f"{block_name}.negative: expected non-empty verdict token")
        if positive == negative:
            raise ValueError(f"{block_name}: positive/negative must normalize to distinct verdicts")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_no_default_mixing(mapping: dict, label: str) -> None:
    """Raise if 'default' coexists with phase-specific keys."""
    if "default" not in mapping:
        return
    non_default = {k for k in mapping if k != "default"}
    if non_default:
        raise ValueError(
            f"{label}: 'default' cannot coexist with phase-specific "
            f"keys {sorted(non_default)}. Use only 'default' or explicit keys."
        )


def _validate(d: dict) -> None:
    if d.get("version") != 2:
        raise ValueError(f"Unsupported prompt version: {d.get('version')} (expected 2)")

    _validate_binary_judge_blocks(d)

    for section in ("system", "user"):
        block = d.get(section, {})
        for role in block:
            if role not in _ROLE_NAMES:
                raise ValueError(f"Unknown role '{role}' in {section}")

    # Phase-key mixing: 'default' cannot coexist with phase-specific keys.
    for section_name in ("system", "user", "prefill"):
        block = d.get(section_name, {})
        for role, phases in block.items():
            if isinstance(phases, dict):
                _check_no_default_mixing(phases, f"{section_name}.{role}")

    # System must have at least one template per role.
    for role, phases in d.get("system", {}).items():
        if not phases:
            raise ValueError(f"system.{role}: no templates defined")

    # Question must have debater_a AND debater_b (hard error). Judge is optional.
    question_block = d.get("question", {})
    for required_role in ("debater_a", "debater_b"):
        if required_role not in question_block:
            raise ValueError(f"question section missing required role '{required_role}'")
    for role in question_block:
        if role not in _ROLE_NAMES:
            raise ValueError(f"Unknown role '{role}' in question")

    # Validate think config types.
    think_block = d.get("think", {})
    _think_reserved = {"tag", "visibility"}
    for role, val in think_block.items():
        if role not in _ROLE_NAMES:
            raise ValueError(f"Unknown role '{role}' in think")
        if val is True:
            raise ValueError(
                f"think.{role}: bare `true` is not allowed — "
                "specify a ThinkVisibility string (private, visible_to_judge, open) or false"
            )
        if val is False or isinstance(val, str):
            pass  # scalar OK
        elif isinstance(val, dict):
            phase_keys = {k for k in val if k not in _think_reserved}
            if phase_keys:
                _check_no_default_mixing({k: val[k] for k in phase_keys}, f"think.{role}")
            for phase, v in val.items():
                if phase in _think_reserved:
                    continue  # reserved keys validated in _normalize_think
                if v is True:
                    raise ValueError(
                        f"think.{role}.{phase}: bare `true` is not allowed — "
                        "specify a ThinkVisibility string or false"
                    )
                if not isinstance(v, (bool, str)):
                    raise ValueError(
                        f"think.{role}.{phase}: expected false or str, got {type(v).__name__}"
                    )
        else:
            raise ValueError(
                f"think.{role}: expected false, str, or dict, got {type(val).__name__}"
            )

    # Validate field tag names.
    for role, triggers in d.get("fields", {}).items():
        if role not in _ROLE_NAMES:
            raise ValueError(f"Unknown role '{role}' in fields")
        for trigger, field_defs in triggers.items():
            for tag_name in field_defs:
                if not _TAG_NAME_RE.match(tag_name):
                    raise ValueError(f"Invalid field tag name: '{tag_name}'")

    # Validate field scoring compatibility.
    for role, triggers in d.get("fields", {}).items():
        for trigger, field_defs in triggers.items():
            for tag_name, props in field_defs.items():
                if isinstance(props, dict) and "scoring" in props:
                    type_str = props.get("type", "str")
                    if type_str in _TYPE_MAP:
                        from ..scoring.fields import resolve_scoring

                        scoring = resolve_scoring(props["scoring"])
                        if scoring is not None:
                            validate_type_scoring(tag_name, _TYPE_MAP[type_str], scoring)

    # Validate judge final fields when present: must contain exactly one EnumScoring
    # field whose values map to known roles covering both debaters.
    final_fields = d.get("fields", {}).get("judge", {}).get("final")
    if final_fields is not None:
        from ..scoring.fields import resolve_scoring as _resolve_scoring
        from ..scoring.judge import _ENUM_TO_ROLE

        allowed_values = set(_ENUM_TO_ROLE.keys())
        role_mapping = {k: v for k, v in _ENUM_TO_ROLE.items() if v is not None}

        enum_fields: list[tuple[str, dict]] = []
        for tag_name, props in final_fields.items():
            if isinstance(props, dict) and "scoring" in props:
                scoring = _resolve_scoring(props["scoring"])
                if isinstance(scoring, EnumScoring):
                    enum_fields.append((tag_name, props))
        if len(enum_fields) == 0:
            raise ValueError(
                "fields.judge.final must contain exactly one field with EnumScoring "
                "(the decision field), found 0"
            )
        if len(enum_fields) > 1:
            names = [n for n, _ in enum_fields]
            raise ValueError(
                f"fields.judge.final must contain exactly one field with EnumScoring, "
                f"found {len(enum_fields)}: {names}"
            )
        enum_tag, enum_props = enum_fields[0]
        enum_values = (
            enum_props["scoring"].get("values", [])
            if isinstance(enum_props["scoring"], dict)
            else []
        )
        bad_values = set(enum_values) - allowed_values
        if bad_values:
            raise ValueError(
                f"fields.judge.final.{enum_tag}: EnumScoring values {sorted(bad_values)} "
                f"are not recognized. Allowed: {sorted(allowed_values)}"
            )
        covered_roles = {role_mapping[v] for v in enum_values if v in role_mapping}
        if Role.DEBATER_A not in covered_roles or Role.DEBATER_B not in covered_roles:
            raise ValueError(
                f"fields.judge.final.{enum_tag}: EnumScoring values must map to both "
                f"DEBATER_A and DEBATER_B. Covered roles: {covered_roles}"
            )

    # Validate opponent_wrap section.
    ow = d.get("opponent_wrap")
    if ow is not None:
        if not isinstance(ow, dict):
            raise ValueError(f"opponent_wrap: expected mapping, got {type(ow).__name__}")
        valid_keys = {"debater", "judge"}
        extra = set(ow) - valid_keys
        if extra:
            raise ValueError(
                f"opponent_wrap: unknown keys {sorted(extra)} (expected 'debater' and/or 'judge')"
            )
        for key, val in ow.items():
            if not isinstance(val, str):
                raise ValueError(f"opponent_wrap.{key}: expected str, got {type(val).__name__}")


_jinja_env = jinja2.sandbox.SandboxedEnvironment(undefined=jinja2.StrictUndefined)


def _compile_templates(block: dict) -> dict[str, dict[str, jinja2.Template]]:
    result: dict[str, dict[str, jinja2.Template]] = {}
    for role, phases in block.items():
        result[role] = {}
        for phase_name, template_str in phases.items():
            result[role][phase_name] = _jinja_env.from_string(template_str)
    return result


def _compile_flat_templates(block: dict[str, str]) -> dict[str, jinja2.Template]:
    return {role: _jinja_env.from_string(tmpl_str) for role, tmpl_str in block.items()}


def _parse_think_leaf(val: object, path: str, tag: str = "thinking") -> ThinkConfig:
    """Parse a leaf think value into ThinkConfig."""
    if val is False:
        return ThinkConfig(visibility=ThinkVisibility.DISABLED, tag=tag)
    if isinstance(val, str):
        try:
            vis = ThinkVisibility(val)
        except ValueError:
            raise ValueError(f"{path}: unknown ThinkVisibility '{val}'") from None
        return ThinkConfig(visibility=vis, tag=tag)
    raise ValueError(f"{path}: expected false or ThinkVisibility string, got {type(val).__name__}")


def _normalize_think(block: dict) -> dict[str, dict[str, ThinkConfig]]:
    _RESERVED_KEYS = {"tag", "visibility"}

    result: dict[str, dict[str, ThinkConfig]] = {}
    for role, val in block.items():
        if val is True:
            raise ValueError(
                f"think.{role}: bare `true` is not allowed — "
                "specify a ThinkVisibility string (private, visible_to_judge, open) or false"
            )
        if val is False:
            result[role] = {"default": ThinkConfig(visibility=ThinkVisibility.DISABLED)}
            continue
        if isinstance(val, str):
            tc = _parse_think_leaf(val, f"think.{role}")
            result[role] = {"default": tc}
            continue
        if isinstance(val, dict):
            # Extract reserved keys
            tag = val.get("tag", "thinking")
            if not isinstance(tag, str):
                raise ValueError(f"think.{role}.tag: expected str, got {type(tag).__name__}")
            role_visibility_str = val.get("visibility")
            role_visibility: ThinkVisibility | None = None
            if role_visibility_str is not None:
                if not isinstance(role_visibility_str, str):
                    raise ValueError(
                        f"think.{role}.visibility: expected str, got {type(role_visibility_str).__name__}"
                    )
                try:
                    role_visibility = ThinkVisibility(role_visibility_str)
                except ValueError:
                    raise ValueError(
                        f"think.{role}.visibility: unknown ThinkVisibility '{role_visibility_str}'"
                    ) from None

            phase_keys = {k for k in val if k not in _RESERVED_KEYS}

            if not phase_keys and role_visibility is not None:
                # Role-level visibility, no per-phase overrides → apply to all phases
                result[role] = {"default": ThinkConfig(visibility=role_visibility, tag=tag)}
                continue

            if not phase_keys and role_visibility is None:
                # Empty dict with just tag → disabled
                result[role] = {
                    "default": ThinkConfig(visibility=ThinkVisibility.DISABLED, tag=tag)
                }
                continue

            # Per-phase overrides
            if role_visibility is not None:
                # Pre-populate ALL lookup keys with role-level default, then override
                result[role] = {
                    k: ThinkConfig(visibility=role_visibility, tag=tag) for k in _ALL_LOOKUP_KEYS
                }
            else:
                result[role] = {}

            for phase_key in phase_keys:
                phase_val = val[phase_key]
                if phase_val is True:
                    raise ValueError(
                        f"think.{role}.{phase_key}: bare `true` is not allowed — "
                        "specify a ThinkVisibility string or false"
                    )
                if role_visibility is not None:
                    # Role-level visibility set: phase values are enablement bools/strings
                    if phase_val is False:
                        result[role][phase_key] = ThinkConfig(
                            visibility=ThinkVisibility.DISABLED, tag=tag
                        )
                    elif isinstance(phase_val, str):
                        # Phase-specific visibility override
                        tc = _parse_think_leaf(phase_val, f"think.{role}.{phase_key}", tag=tag)
                        result[role][phase_key] = tc
                    else:
                        raise ValueError(
                            f"think.{role}.{phase_key}: expected false or ThinkVisibility string, "
                            f"got {type(phase_val).__name__}"
                        )
                else:
                    # No role-level visibility — infer from phase values
                    result[role][phase_key] = _parse_think_leaf(
                        phase_val, f"think.{role}.{phase_key}", tag=tag
                    )
            continue

        raise ValueError(f"think.{role}: expected false, str, or dict, got {type(val).__name__}")

    # Post-parse validation: uniform visibility per role, uniform tag per role
    for role_str, phases in result.items():
        visibilities = {
            tc.visibility for tc in phases.values() if tc.visibility != ThinkVisibility.DISABLED
        }
        if len(visibilities) > 1:
            raise ValueError(
                f"think.{role_str}: mixed non-DISABLED visibility values {sorted(v.value for v in visibilities)}. "
                "All enabled phases for a role must share the same visibility."
            )
        tags = {tc.tag for tc in phases.values()}
        if len(tags) > 1:
            raise ValueError(
                f"think.{role_str}: mixed tag values {sorted(tags)}. "
                "All phases for a role must use the same tag."
            )

    return result


def _expand_defaults(compiled: dict[str, dict], section_name: str) -> None:
    """Expand sole 'default' key into all lookup keys at compile time.

    After expansion no 'default' key remains — render-time lookups are strict.
    Precondition: _validate() already rejected mixed default + specific keys.
    """
    for role, phases in compiled.items():
        if "default" not in phases:
            continue
        assert len(phases) == 1, f"_validate should have caught mixed keys in {section_name}.{role}"
        default_val = phases.pop("default")
        for key in _ALL_LOOKUP_KEYS:
            phases[key] = default_val


def _parse_fields(block: dict) -> dict[str, dict[str, dict[str, FieldSpec]]]:
    result: dict[str, dict[str, dict[str, FieldSpec]]] = {}
    for role, triggers in block.items():
        result[role] = {}
        for trigger, field_defs in triggers.items():
            result[role][trigger] = _resolve_fields(field_defs)
    return result


def _compile_binary_judge_blocks(d: dict) -> dict[str, BinaryJudgeTemplate]:
    result: dict[str, BinaryJudgeTemplate] = {}
    for block_name, short_name in (("_matcher", "matcher"), ("_grader", "grader")):
        block = d.get(block_name)
        if block is None:
            continue
        positive = normalize_binary_verdict_token(block["positive"])
        negative = normalize_binary_verdict_token(block["negative"])
        assert positive is not None
        assert negative is not None
        result[short_name] = BinaryJudgeTemplate(
            system=block["system"],
            user=_jinja_env.from_string(block["user"]),
            positive=positive,
            negative=negative,
        )
    return result


@functools.lru_cache(maxsize=32)
def resolve_prompts(ref: str) -> DebatePrompts:
    """Load and compile a prompt template set.

    If ref contains no '/' or '.', treat as built-in name (looked up in prompts/ subdir).
    Otherwise treat as file path.
    """
    ref = ref.strip()
    if "/" not in ref and "." not in ref:
        path = _PROMPTS_DIR / f"{ref}.yaml"
    else:
        path = Path(ref)

    raw = path.read_text()
    content_hash = hashlib.sha256(raw.encode()).hexdigest()
    d = yaml.safe_load(raw)

    # Migration lint BEFORE validation (catches removed vars with helpful message).
    _check_migration_lint(d)
    _validate(d)

    system = _compile_templates(d.get("system", {}))
    _expand_defaults(system, "system")
    user = _compile_templates(d.get("user", {}))
    _expand_defaults(user, "user")
    question = _compile_flat_templates(d.get("question", {}))
    think = _normalize_think(d.get("think", {}))
    _expand_defaults(think, "think")
    prefill = _compile_templates(d.get("prefill", {}))
    _expand_defaults(prefill, "prefill")
    fields = _parse_fields(d.get("fields", {}))
    binary_judges = _compile_binary_judge_blocks(d)

    # Compile opponent_wrap templates.
    raw_ow = d.get("opponent_wrap")
    opponent_wrap: dict[str, jinja2.Template] | None = None
    if raw_ow is not None:
        opponent_wrap = {key: _jinja_env.from_string(val) for key, val in raw_ow.items()}

    prompts = DebatePrompts(
        system=system,
        user=user,
        question=question,
        think=think,
        prefill=prefill,
        fields=fields,
        binary_judges=binary_judges,
        content_hash=content_hash,
        source_ref=str(path),
        opponent_wrap=opponent_wrap,
    )

    warnings = check_ab_symmetry(prompts)
    for w in warnings:
        _log.warning(f"A/B asymmetry: {w}")

    return prompts
