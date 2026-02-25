"""Token usage tracking and cost estimation for Tinker API calls.

Records per-call usage events from completers, supports cursor-based delta
queries for per-step attribution, and prints end-of-run cost breakdowns.

Pricing data from https://thinkingmachines.ai/tinker/ (USD per million tokens).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens for a Tinker model."""

    prefill_usd_per_mtok: float  # Input / prompt tokens (sampling)
    sample_usd_per_mtok: float  # Output / completion tokens (sampling)
    train_usd_per_mtok: float  # Training forward+backward tokens


# ---------------------------------------------------------------------------
# Pricing table: all Tinker-supported models.
#
# Source: https://thinkingmachines.ai/tinker/
# Last updated: 2026-02-24
#
# Keyed by full HuggingFace model path (org/model).  Variants at the same
# parameter count / architecture share pricing (confirmed: Tinker bills by
# compute, not by variant label).
# ---------------------------------------------------------------------------

_Q4 = ModelPricing(0.07, 0.22, 0.22)
_Q8 = ModelPricing(0.13, 0.40, 0.40)
_Q30_MOE = ModelPricing(0.12, 0.30, 0.36)
_Q30_VL = ModelPricing(0.18, 0.44, 0.53)
_Q32 = ModelPricing(0.49, 1.47, 1.47)
_Q235 = ModelPricing(0.68, 1.70, 2.04)
_Q235_VL = ModelPricing(1.02, 2.56, 3.07)

_L1B = ModelPricing(0.03, 0.09, 0.09)
_L3B = ModelPricing(0.06, 0.18, 0.18)
_L8B = ModelPricing(0.13, 0.40, 0.40)
_L70B = ModelPricing(1.05, 3.16, 3.16)

_DS_V3 = ModelPricing(1.13, 2.81, 3.38)

_GPT20B = ModelPricing(0.12, 0.30, 0.36)
_GPT120B = ModelPricing(0.18, 0.44, 0.52)

_KIMI_K2 = ModelPricing(0.98, 2.44, 2.93)
_KIMI_K25 = ModelPricing(1.47, 3.66, 4.40)

PRICING: dict[str, ModelPricing] = {
    # --- Qwen 3 (dense, 4B) ---
    "Qwen/Qwen3-4B": _Q4,
    "Qwen/Qwen3-4B-Base": _Q4,
    "Qwen/Qwen3-4B-Instruct-2507": _Q4,
    # --- Qwen 3 (dense, 8B) ---
    "Qwen/Qwen3-8B": _Q8,
    "Qwen/Qwen3-8B-Base": _Q8,
    # --- Qwen 3 (MoE, 30B-A3B) ---
    "Qwen/Qwen3-30B-A3B": _Q30_MOE,
    "Qwen/Qwen3-30B-A3B-Base": _Q30_MOE,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": _Q30_MOE,
    # --- Qwen 3 VL (MoE, 30B-A3B) ---
    "Qwen/Qwen3-VL-30B-A3B-Instruct": _Q30_VL,
    # --- Qwen 3 (dense, 32B) ---
    "Qwen/Qwen3-32B": _Q32,
    # --- Qwen 3 (MoE, 235B-A22B) ---
    "Qwen/Qwen3-235B-A22B-Instruct-2507": _Q235,
    # --- Qwen 3 VL (MoE, 235B-A22B) ---
    "Qwen/Qwen3-VL-235B-A22B-Instruct": _Q235_VL,
    # --- Llama 3.2 (dense, 1B) ---
    "meta-llama/Llama-3.2-1B": _L1B,
    "meta-llama/Llama-3.2-1B-Instruct": _L1B,
    # --- Llama 3.2 (dense, 3B) ---
    "meta-llama/Llama-3.2-3B": _L3B,
    "meta-llama/Llama-3.2-3B-Instruct": _L3B,
    # --- Llama 3.1 (dense, 8B) ---
    "meta-llama/Llama-3.1-8B": _L8B,
    "meta-llama/Llama-3.1-8B-Instruct": _L8B,
    # --- Llama (dense, 70B) ---
    "meta-llama/Llama-3.1-70B": _L70B,
    "meta-llama/Llama-3.3-70B-Instruct": _L70B,
    # --- DeepSeek V3.1 (MoE, 671B-A37B) ---
    "deepseek-ai/DeepSeek-V3.1": _DS_V3,
    "deepseek-ai/DeepSeek-V3.1-Base": _DS_V3,
    # --- OpenAI GPT-OSS (MoE) ---
    "openai/gpt-oss-20b": _GPT20B,
    "openai/gpt-oss-120b": _GPT120B,
    # --- Moonshot Kimi (MoE, 1T-A32B) ---
    "moonshotai/Kimi-K2-Thinking": _KIMI_K2,
    "moonshotai/Kimi-K2.5": _KIMI_K25,
}


def get_pricing(model_name: str) -> ModelPricing | None:
    """Look up pricing for a model by full HF path. Returns None if unknown."""
    return PRICING.get(model_name)


# ---------------------------------------------------------------------------
# Usage events and tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageEvent:
    """A single API call's token usage."""

    actor: str  # "trained", "opponent", "judge", or custom label
    model_name: str  # Full HF path, e.g. "Qwen/Qwen3-4B-Instruct-2507"
    input_tokens: int  # Prompt / prefill tokens (sampling) or total tokens (training)
    output_tokens: int  # Sampled / completion tokens (0 for training)
    call_type: str = "sample"  # "sample" or "train"


@dataclass
class _ActorModelBucket:
    """Accumulated tokens for one (actor, model) pair."""

    input_tokens: int = 0
    output_tokens: int = 0
    train_tokens: int = 0
    n_calls: int = 0


class UsageTracker:
    """Accumulates usage events with cursor-based delta queries.

    Safe for asyncio (single-threaded event loop). The cursor pattern lets
    consumers ask "what happened since I last checked?" without any mutable
    state on their side beyond an int.
    """

    def __init__(self) -> None:
        self._events: list[UsageEvent] = []

    def record(self, event: UsageEvent) -> None:
        self._events.append(event)

    def cursor(self) -> int:
        return len(self._events)

    def events_since(self, cursor: int) -> list[UsageEvent]:
        return self._events[cursor:]

    def all_events(self) -> list[UsageEvent]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _buckets(
        self, events: list[UsageEvent] | None = None,
    ) -> dict[tuple[str, str], _ActorModelBucket]:
        """Aggregate events into (actor, model) → bucket."""
        evts = events if events is not None else self._events
        buckets: dict[tuple[str, str], _ActorModelBucket] = defaultdict(
            _ActorModelBucket
        )
        for e in evts:
            b = buckets[(e.actor, e.model_name)]
            if e.call_type == "train":
                b.train_tokens += e.input_tokens
            else:
                b.input_tokens += e.input_tokens
                b.output_tokens += e.output_tokens
            b.n_calls += 1
        return dict(buckets)

    @staticmethod
    def _bucket_cost(b: _ActorModelBucket, p: ModelPricing) -> float:
        """Compute USD cost for a single bucket given its pricing."""
        return (
            b.input_tokens * p.prefill_usd_per_mtok
            + b.output_tokens * p.sample_usd_per_mtok
            + b.train_tokens * p.train_usd_per_mtok
        ) / 1_000_000

    def total_cost_usd(self) -> float:
        """Compute total estimated cost across all events."""
        total = 0.0
        for (_, model), b in self._buckets().items():
            p = get_pricing(model)
            if p is not None:
                total += self._bucket_cost(b, p)
        return total

    def format_cost_report(self) -> str:
        """Format a human-readable cost breakdown."""
        buckets = self._buckets()
        if not buckets:
            return "No usage recorded."

        w = 110
        lines: list[str] = []
        lines.append("")
        lines.append("Tinker Usage Cost Report")
        lines.append("=" * w)
        lines.append(
            f"{'Actor':<10} {'Model':<38} {'Calls':>6} "
            f"{'Prefill Tok':>13} {'Sample Tok':>12} {'Train Tok':>12} {'Cost (USD)':>12}"
        )
        lines.append("-" * w)

        grand_input = 0
        grand_output = 0
        grand_train = 0
        grand_cost = 0.0

        for (actor, model), b in sorted(buckets.items()):
            p = get_pricing(model)
            if p is not None:
                cost = self._bucket_cost(b, p)
                cost_str = f"${cost:.4f}"
            else:
                cost = 0.0
                cost_str = "unknown"

            lines.append(
                f"{actor:<10} {model:<38} {b.n_calls:>6} "
                f"{b.input_tokens:>13,} {b.output_tokens:>12,} "
                f"{b.train_tokens:>12,} {cost_str:>12}"
            )
            grand_input += b.input_tokens
            grand_output += b.output_tokens
            grand_train += b.train_tokens
            grand_cost += cost

        lines.append("-" * w)
        lines.append(
            f"{'TOTAL':<10} {'':<38} {'':<6} "
            f"{grand_input:>13,} {grand_output:>12,} "
            f"{grand_train:>12,} {'$' + f'{grand_cost:.4f}':>12}"
        )
        lines.append("")
        return "\n".join(lines)

    def as_metrics(self, prefix: str = "usage") -> dict[str, float | int]:
        """Export usage as a flat metrics dict for ml_log / wandb."""
        metrics: dict[str, float | int] = {}
        for (actor, model), b in self._buckets().items():
            short_model = model.split("/")[-1]
            key = f"{prefix}/{actor}/{short_model}"
            metrics[f"{key}/input_tokens"] = b.input_tokens
            metrics[f"{key}/output_tokens"] = b.output_tokens
            metrics[f"{key}/train_tokens"] = b.train_tokens
            metrics[f"{key}/n_calls"] = b.n_calls
            p = get_pricing(model)
            if p is not None:
                metrics[f"{key}/cost_usd"] = self._bucket_cost(b, p)
        metrics[f"{prefix}/total_cost_usd"] = self.total_cost_usd()
        return metrics
