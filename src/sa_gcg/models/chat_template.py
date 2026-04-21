"""Chat templating + suffix-span detection.

Why this is its own module: prior codebases (SGCG, AGCG, refusal_direction)
each templated differently and their results were not directly comparable.
EVAL_PLAN.md §14.1 enumerates the resulting bugs. We therefore pin the
template per model family in one place and resolve the suffix span by anchor
search rather than position arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Per Arditi 2024 Appendix F.2: Llama-2 ASR is highly sensitive to system
# prompt. We pin the published default for all reported runs.
LLAMA2_SYSTEM_DEFAULT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe. Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. Please ensure that your responses are socially "
    "unbiased and positive in nature.\n\nIf a question does not make sense, or is not factually "
    "coherent, explain why instead of answering something not correct. If you don't know the "
    "answer to a question, please don't share false information."
)


@dataclass(frozen=True)
class ChatSpec:
    """Per-family templating spec.

    The template is rendered as ``prefix + user + " " + suffix + postfix``
    where ``suffix`` is the adversarial token sequence.  ``suffix_anchor`` is
    a *string* that we tokenize with the model's tokenizer to derive the
    anchor token sequence at runtime; using the string keeps us agnostic to
    tokenizer-specific id remappings.
    """

    name: str
    prefix_tmpl: str  # rendered with .format(system=...)
    user_postfix: str  # text inserted between user message and suffix
    suffix_postfix: str  # text inserted between suffix and assistant turn
    suffix_anchor: str  # string that follows the suffix span exactly once
    default_system: str | None


LLAMA2_SPEC = ChatSpec(
    name="llama-2",
    prefix_tmpl="<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n",
    user_postfix=" ",  # user message + space + suffix
    suffix_postfix=" [/INST]",
    suffix_anchor=" [/INST]",
    default_system=LLAMA2_SYSTEM_DEFAULT,
)

LLAMA3_SPEC = ChatSpec(
    name="llama-3",
    prefix_tmpl=(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    ),
    user_postfix=" ",
    suffix_postfix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    suffix_anchor="<|eot_id|>",
    default_system="You are a helpful assistant.",
)

VICUNA_SPEC = ChatSpec(
    name="vicuna",
    prefix_tmpl=(
        "{system} "
        "USER: "
    ),
    user_postfix=" ",
    suffix_postfix=" ASSISTANT:",
    suffix_anchor=" ASSISTANT:",
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
)

MISTRAL_SPEC = ChatSpec(
    name="mistral",
    prefix_tmpl="<s>[INST] ",
    user_postfix=" ",
    suffix_postfix=" [/INST]",
    suffix_anchor=" [/INST]",
    default_system=None,
)


@dataclass(frozen=True)
class SuffixSlot:
    """Tokenization of a templated prompt with the suffix span identified."""

    input_ids: list[int]      # full sequence
    suffix_start: int         # inclusive
    suffix_end: int           # exclusive
    prefix_ids: list[int]
    suffix_ids: list[int]
    postfix_ids: list[int]

    @property
    def suffix_len(self) -> int:
        return self.suffix_end - self.suffix_start


def build_input_ids(
    tokenizer,
    spec: ChatSpec,
    user_message: str,
    suffix_str: str,
    *,
    system: str | None = None,
    add_special_tokens: bool = False,
) -> SuffixSlot:
    """Render a templated prompt and locate the suffix span by anchor search.

    The anchor approach is robust to subword tokenization edge cases that
    would break naive offset arithmetic (which previous codebases tried and
    miscounted on for many sentences containing apostrophes / unicode).
    """
    sys_text = system if system is not None else spec.default_system
    prefix = spec.prefix_tmpl.format(system=sys_text or "")
    full_text = (
        prefix + user_message + spec.user_postfix + suffix_str + spec.suffix_postfix
    )
    full_ids = tokenizer(full_text, add_special_tokens=add_special_tokens).input_ids
    return _slot_from_anchor(tokenizer, spec, full_ids, suffix_str)


def find_suffix_span(
    tokenizer, spec: ChatSpec, full_ids: Sequence[int], suffix_str: str
) -> tuple[int, int]:
    """Locate (start, end) of suffix tokens in ``full_ids`` by anchor search."""
    slot = _slot_from_anchor(tokenizer, spec, list(full_ids), suffix_str)
    return slot.suffix_start, slot.suffix_end


def _slot_from_anchor(
    tokenizer, spec: ChatSpec, full_ids: list[int], suffix_str: str
) -> SuffixSlot:
    anchor_ids = tokenizer(spec.suffix_anchor, add_special_tokens=False).input_ids
    if not anchor_ids:
        raise ValueError(f"Anchor {spec.suffix_anchor!r} tokenized to empty.")

    end = _find_subseq_last(full_ids, anchor_ids)
    if end is None:
        raise ValueError(
            f"Could not locate anchor {spec.suffix_anchor!r} (ids {anchor_ids}) in input. "
            f"Tokenizer or template likely diverged from spec."
        )
    suffix_end = end  # anchor starts where suffix ends
    suffix_ids = tokenizer(suffix_str, add_special_tokens=False).input_ids
    suffix_start = suffix_end - len(suffix_ids)
    if suffix_start < 0:
        raise ValueError("Suffix span starts before token 0; spec is wrong.")
    if full_ids[suffix_start:suffix_end] != suffix_ids:
        # Tokenizer may merge across boundaries (e.g. preceding space).
        # Search for the suffix tokens directly.
        candidate = _find_subseq_last(full_ids[:suffix_end], suffix_ids)
        if candidate is None:
            raise ValueError(
                "Suffix tokens not found in expected position. "
                "Likely a tokenizer-merge issue across the user/suffix boundary."
            )
        suffix_start = candidate - len(suffix_ids)
        suffix_end = candidate
    return SuffixSlot(
        input_ids=list(full_ids),
        suffix_start=suffix_start,
        suffix_end=suffix_end,
        prefix_ids=list(full_ids[:suffix_start]),
        suffix_ids=list(full_ids[suffix_start:suffix_end]),
        postfix_ids=list(full_ids[suffix_end:]),
    )


def _find_subseq_last(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    """Return the index *just past* the last occurrence of ``needle``, or None."""
    if not needle:
        return None
    n = len(needle)
    needle_t = tuple(needle)
    for i in range(len(haystack) - n, -1, -1):
        if tuple(haystack[i : i + n]) == needle_t:
            return i + n
    return None
