"""YAML-driven prompt system for debate environments."""

from __future__ import annotations

import functools
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import jinja2
import jinja2.meta
import jinja2.sandbox
import yaml

from ..types import (
    DebateState,
    PHASE_DONE,
    Phase,
    Role,
    TRIGGER_BOUNDARY,
    TRIGGER_FINAL,
    current_phase,
)
from ..scoring.fields import FieldSpec, _resolve_fields, validate_type_scoring, _TYPE_MAP
from ..scoring.parsing import generate_format_instructions

_PROMPTS_DIR = Path(__file__).parent

# Sentinel tokens for two-phase rendering (template injection prevention).
_SENTINEL_PREFIX = "__SENTINEL_"
_SENTINEL_SUFFIX = f"_{uuid.uuid4().hex[:8]}__"
_INJECTABLE_KEYS = ("task_prompt", "answer", "answer_a", "answer_b")

_ROLE_NAMES = {"judge", "debater_a", "debater_b"}
_TAG_NAME_RE = re.compile(r"^\w+$")

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
class DebatePrompts:
    system: dict[str, dict[str, jinja2.Template]]
    user: dict[str, dict[str, jinja2.Template]]
    question: dict[str, jinja2.Template]
    think: dict[str, dict[str, bool | jinja2.Template]]
    prefill: dict[str, dict[str, jinja2.Template]]
    fields: dict[str, dict[str, dict[str, FieldSpec]]]
    content_hash: str
    source_ref: str

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
        """Resolve think config + open_reasoning -> instruction string or None."""
        role = viewer.value
        if role not in self.think:
            return None
        cfg = self.think[role]
        phase = trigger or current_phase(state)
        if phase not in cfg:
            return None
        val = cfg[phase]
        if val is False or val is None:
            return None
        if val is True:
            if state.spec.open_reasoning:
                return "Use <thinking>...</thinking> tags for your reasoning. Note: all reasoning is visible to all participants."
            else:
                return "Use <thinking>...</thinking> tags for private reasoning that your opponent will NOT see."
        # Custom template string
        assert isinstance(val, jinja2.Template)
        ctx = _build_context(state, viewer)
        return _two_phase_render(val, ctx, state, viewer)

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

    # think (heterogeneous): normalize shape before comparing
    if prompts.think:
        a_think = prompts.think.get("debater_a", {})
        b_think = prompts.think.get("debater_b", {})
        a_shape = {k: type(v).__name__ for k, v in a_think.items()}
        b_shape = {k: type(v).__name__ for k, v in b_think.items()}
        if set(a_shape.keys()) != set(b_shape.keys()):
            warnings.append(
                f"think: debater_a phases {sorted(a_shape.keys())} != debater_b phases {sorted(b_shape.keys())}"
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

    answer_by_role = state.spec.answer_by_role
    has_assigned = bool(answer_by_role and viewer in answer_by_role and answer_by_role[viewer])

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
        "open_reasoning": state.spec.open_reasoning,
        "has_assigned_answer": has_assigned,
        "reasoning_is_private": not state.spec.open_reasoning,
    }


def _actual_values(state: DebateState, viewer: Role) -> dict[str, str]:
    """Map sentinel keys to their real values for post-Jinja replacement."""
    answer_by_role = state.spec.answer_by_role or {}
    return {
        "task_prompt": state.spec.task_prompt,
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
    for role, val in think_block.items():
        if role not in _ROLE_NAMES:
            raise ValueError(f"Unknown role '{role}' in think")
        if isinstance(val, (bool, str)):
            pass  # scalar OK
        elif isinstance(val, dict):
            _check_no_default_mixing(val, f"think.{role}")
            for phase, v in val.items():
                if not isinstance(v, (bool, str)):
                    raise ValueError(
                        f"think.{role}.{phase}: expected bool or str, got {type(v).__name__}"
                    )
        else:
            raise ValueError(f"think.{role}: expected bool, str, or dict, got {type(val).__name__}")

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


def _normalize_think(block: dict) -> dict[str, dict[str, bool | jinja2.Template]]:
    result: dict[str, dict[str, bool | jinja2.Template]] = {}
    for role, val in block.items():
        if isinstance(val, bool):
            result[role] = {"default": val}
        elif isinstance(val, str):
            result[role] = {"default": _jinja_env.from_string(val)}
        elif isinstance(val, dict):
            result[role] = {}
            for phase, v in val.items():
                if isinstance(v, bool):
                    result[role][phase] = v
                elif isinstance(v, str):
                    result[role][phase] = _jinja_env.from_string(v) if v else False
                else:
                    raise ValueError(
                        f"think.{role}.{phase}: expected bool or str, got {type(v).__name__}"
                    )
        else:
            raise ValueError(f"think.{role}: expected bool, str, or dict, got {type(val).__name__}")
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

    prompts = DebatePrompts(
        system=system,
        user=user,
        question=question,
        think=think,
        prefill=prefill,
        fields=fields,
        content_hash=content_hash,
        source_ref=str(path),
    )

    warnings = check_ab_symmetry(prompts)
    for w in warnings:
        _log.warning(f"A/B asymmetry: {w}")

    return prompts
