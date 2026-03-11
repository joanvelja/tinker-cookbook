# Thinking Architecture Compendium

How `<think>` tags flow through the Tinker SDK, tinker-cookbook renderers, the debate recipe, and RL training. Reference document for contributors working on thinking-model integration.

---

## 1. Tinker Backend: Thinking-Unaware

The Tinker SDK has no concept of thinking. There is no thinking API, no thinking field, no thinking parameter.

- `SampledSequence.tokens` is `list[int]` — a flat token list. If the model emits `<think>reasoning</think>answer`, those tokens appear inline with no structural separation.
- `SamplingParams` has: `max_tokens`, `seed`, `stop`, `temperature`, `top_k`, `top_p`. Zero thinking-related params.
- `<think>` and `</think>` are just tokens to the backend. They get sampled, returned, and trained on like any other token.

**Source:** `docs/api-reference/types.md` lines 209-223 (SampledSequence), 840-865 (SamplingParams).

---

## 2. Thinking Toggle = Renderer Choice

Thinking mode is controlled entirely by which renderer you pick. There is no runtime toggle.

| Renderer name | Class | Thinking | Notes |
|---|---|---|---|
| `qwen3` | `Qwen3Renderer` | ON | Default `strip_thinking_from_history=True` |
| `qwen3_disable_thinking` | `Qwen3DisableThinkingRenderer` | OFF | Injects empty `<think>\n\n</think>\n\n` in header |
| `qwen3_instruct` | `Qwen3InstructRenderer` | N/A | 2507 models, no `<think>` tag at all |

**How it works:**
- `Qwen3Renderer` uses `parse_content_blocks()` during `parse_response()` to split raw text into `ThinkingPart` / `TextPart` structured content. During `render_message()`, it re-renders these as `<think>...</think>` text.
- `Qwen3DisableThinkingRenderer` extends `Qwen3Renderer` by prepending an empty think block (`<think>\n\n</think>\n\n`) to the header of the last assistant message, signaling the model to skip reasoning.
- `Qwen3InstructRenderer` inherits from `Qwen3Renderer` but targets 2507 models that never use `<think>` tags. `has_extension_property` returns `True` unconditionally.

**`strip_thinking_from_history`:** Controls whether `ThinkingPart` entries are removed from historical (non-last) assistant messages during `render_message()`. Only operates on list-based (structured) content — string content passes through unstripped. Defaults to `True` for all Qwen3 renderers.

**Source:** `tinker_cookbook/renderers/qwen3.py` lines 60-228 (Qwen3Renderer), 328-364 (DisableThinking), 367-383 (Instruct). Registry: `tinker_cookbook/renderers/__init__.py` lines 95-155.

---

## 3. Content Representation Duality

Message content exists in two forms:

1. **String**: `message["content"] = "raw text including <think>...</think> inline"`
2. **Structured**: `message["content"] = [ThinkingPart(...), TextPart(...)]`

Key functions that navigate this duality (all in `tinker_cookbook/renderers/base.py`):

| Function | Input | Output | Behavior |
|---|---|---|---|
| `parse_content_blocks` (line 492) | `str` | `(list[ContentPart], list[ToolCall])` or `None` | Parses `<think>` and `<tool_call>` tags from raw string. Returns `None` if no tags found. |
| `get_text_content` (line 416) | `Message` | `str` | Extracts only `TextPart` text, **silently drops ThinkingPart**. String content passes through as-is. |
| `format_content_as_string` (line 428) | `Content` | `str` | Preserves all parts: wraps `ThinkingPart` back into `<think>...</think>` tags. String content passes through. |
| `remove_thinking` (line 411) | `list[ContentPart]` | `list[ContentPart]` | Filters out entries where `type == "thinking"`. |

The critical distinction: `get_text_content` discards thinking; `format_content_as_string` preserves it.

**Source:** `tinker_cookbook/renderers/base.py` lines 411-457.

---

## 4. Thinking in RL Training

RL training is thinking-unaware at the loss level. The gradient treats think tokens identically to answer tokens.

**Mask construction** (`tinker_cookbook/rl/data_processing.py` line 166):
```python
SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))
```

All action tokens (the model's sampled output) get `mask=1.0` during data assembly. Observation tokens get `mask=0.0`. Since `<think>...</think>` tokens are part of the sampled action, they receive `mask=1.0`.

Note: This mask is used during datum construction for prefix-detection and splitting, but is stripped via `_remove_mask()` (`rl/train.py:188`) before being sent to the Tinker backend. The RL loss functions handle observation-vs-action distinction through the `target_tokens` and `logprobs` inputs, not an explicit mask. The net effect is the same — think tokens in the action span receive gradient — but the mechanism is indirect.

**What controls context, not loss:** `strip_thinking_from_history` affects what tokens appear in the *observation* (the prompt for the next turn). When `True`, past thinking blocks are removed from context, making observations shorter but breaking the extension property.

**Extension property** (`docs/rl/sequence-extension.mdx`):
- `strip_thinking_from_history=False` → `has_extension_property=True` → O(T) compute (KV-cache reuse)
- `strip_thinking_from_history=True` (default) → `has_extension_property=False` → O(T^2) compute (re-encode from scratch each turn)

The RL data processing code auto-detects extension by checking if each observation is a prefix of the next (`_is_prefix` in `data_processing.py`). When extension holds, observations share a single growing sequence; when it doesn't, each turn becomes a separate datum.

**Source:** `tinker_cookbook/rl/data_processing.py` lines 130-171, `tinker_cookbook/renderers/qwen3.py` lines 104-114, `docs/rl/sequence-extension.mdx`.

---

## 5. `strip_thinking_from_history` Interaction with Debate

The `strip_thinking_from_history` flag operates on **structured `ContentPart` lists**, not raw strings. This has a subtle interaction with the debate recipe:

**Debate stores raw text.** `Utterance.text` is always a `str`. After the Wave 1 fix, `env.py` uses `format_content_as_string(msg["content"])` which preserves `<think>` blocks as inline text. The debate transcript never contains structured `ContentPart` — it's string all the way down.

**Consequence:** When the renderer sees historical assistant messages from debate, they have `content = "plain text with <think>...</think> inline"`. The `strip_thinking_from_history` logic in `Qwen3Renderer.render_message()` checks `isinstance(content, list)` before stripping. String content passes through unstripped:

```python
# qwen3.py lines 157-162
else:
    # String content - pass through as-is.
    # Note: strip_thinking_from_history only works with list-based content.
    output_content = content
```

**Therefore:** For the debate recipe, `strip_thinking_from_history` is **inert** because debate stores string content (not structured `ContentPart`). Think blocks in the transcript are handled by the debate visibility layer (`_strip_reasoning`) instead. The renderer flag would only matter if debate stored structured `ContentPart` content.

**Source:** `tinker_cookbook/recipes/multiplayer_rl/debate/env.py` lines 118-120, `tinker_cookbook/renderers/qwen3.py` lines 138-162.

---

## 6. Tag Handling Inconsistencies

Three independent `<think>` stripping implementations exist across the codebase, each with a different regex:

| Location | Regex | Accepts `<thinking>`? | Case insensitive? | Handles unclosed? |
|---|---|---|---|---|
| `visibility.py` line 29 | `<think(?:ing)?[^>]*>.*?</think(?:ing)?>` | Yes | Yes (`re.IGNORECASE`) | No |
| `mcq.py` line 13 | `<think(?:ing)?[^>]*>(.*?)</think(?:ing)?>` | Yes | Yes (`re.IGNORECASE`) | Yes (line 14) |
| `renderers/base.py` line 492 | `parse_content_blocks` (stateful parser) | No (literal `<think>`) | No | No |

**Post-fix state (Wave 1):** `visibility.py` and `mcq.py` regexes were unified — both now accept `<think>` and `<thinking>`, are case-insensitive, and allow attributes via `[^>]*`. The remaining divergence is:
- `visibility._strip_reasoning` does not handle unclosed tags (truncated output).
- `mcq.strip_think` handles unclosed `<think>` via a fallback regex (line 14).
- `parse_content_blocks` in the renderer is a stateful parser that only recognizes literal `<think>` (no variants). This remains un-unified since it's in the shared renderer, not debate-specific.

**Source:** `tinker_cookbook/recipes/multiplayer_rl/debate/core/visibility.py` line 29, `tinker_cookbook/recipes/multiplayer_rl/debate/scoring/mcq.py` lines 13-29, `tinker_cookbook/renderers/base.py` lines 492+.

---

## 7. The `get_text_content` Bug — FIXED (Wave 1)

**Problem:** `env.py` originally called `get_text_content(msg)` on the parsed response, which strips `ThinkingPart` unconditionally. `Utterance.text` never contained think blocks, making `open_reasoning=True` broken.

**Fix applied in Wave 1:**
1. `env.py` lines 69, 118: Replaced `get_text_content(msg)` with `format_content_as_string(msg["content"], separator="")` at both the trained agent and opponent paths. Think blocks now survive into `Utterance.text`.
2. `runtime.py` line 137: Added `strip_think(text)` before `extract_fields()` so field extraction works on clean text even though `Utterance.text` now contains think blocks.
3. `judge.py` line 33: Same fix — `format_content_as_string` + `strip_think` before `extract_fields`.

**Data flow after fix:**
```
tokens → parse_response → Message → format_content_as_string(separator="") → full text w/ <think>
  → Utterance.text (stores everything including think blocks)
  → runtime: strip_think → extract_fields (on clean text)
  → visibility: _strip_reasoning when open_reasoning=False (strips think from opponent view)
  → visibility: preserves when open_reasoning=True (opponent sees reasoning)
```

**Source:** `tinker_cookbook/recipes/multiplayer_rl/debate/env.py` lines 69, 117-118, `core/runtime.py` line 137, `scoring/judge.py` line 33.

---

## 8. Private Reasoning in Debate via Think Tags

Think tags enable private scratchpad reasoning in debate, even with non-thinking models. This is an RL training strategy, not a model capability.

**Mechanism:**
1. The YAML prompt system's `think` section controls per-role, per-phase think instructions. When `think: true` for a role, `get_think_instruction()` (prompts/__init__.py line 65) appends an instruction like "Use `<think>...</think>` tags for private reasoning that your opponent will NOT see."
2. `open_reasoning` determines the wording: when `False` (default), reasoning is "private"; when `True`, reasoning is "visible to all participants."
3. Non-thinking models (e.g., via `qwen3_disable_thinking` or `qwen3_instruct`) can still be *prompted* to use `<think>` tags. The model treats them as text output. The renderer's `parse_content_blocks` picks them up and structures them as `ThinkingPart`.

**RL gradient implications:**
- All tokens in the action get `mask=1.0`, including think tokens. The RL gradient shapes *both* the reasoning and the visible answer.
- `max_tokens` in `SamplingParams` covers the entire generation including think blocks. A model that reasons extensively in `<think>` has fewer tokens for its visible argument. Token budget allocation between reasoning and presentation is itself a learned behavior.

**Visibility flow (when working correctly):**
1. Model generates `<think>reasoning</think>visible argument`
2. `parse_response` structures this into `[ThinkingPart, TextPart]`
3. Transcript stores the full text including `<think>` blocks (fixed in Wave 1 — see Section 7)
4. `_strip_reasoning` in visibility.py removes think blocks when `open_reasoning=False`
5. Opponents see only the visible argument
6. The model being trained sees its own full output (think + visible) in its history

**Source:** `tinker_cookbook/recipes/multiplayer_rl/debate/prompts/__init__.py` lines 65-91, `tinker_cookbook/rl/data_processing.py` line 166, `tinker_cookbook/recipes/multiplayer_rl/debate/core/visibility.py` lines 38-53.
